from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from collections import defaultdict
from typing import TYPE_CHECKING, Callable
from typing import List
from typing import Dict

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.evaluation import SequentialEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.util import cos_sim, dot_score

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


def ndcg_at_k_batch(
    relevances: Tensor,
    scores: Tensor,
    k: int,
) -> tuple:
    """Mean NDCG@k for a batch of queries using graded relevance values.

    Args:
        relevances: (batch_size, num_candidates) graded relevance values.
        scores:     (batch_size, num_candidates) predicted scores.
        k:          cutoff.

    Returns:
        (mean_ndcg_at_k, per_sample_ndcg_at_k).
    """
    batch_size, n = relevances.shape
    k = min(k, n)

    _, indices = torch.sort(scores, dim=1, descending=True)
    sorted_rels = torch.gather(relevances, 1, indices)

    denom = torch.log2(torch.arange(n, device=scores.device).float() + 2.0)
    denom = denom.unsqueeze(0).expand(batch_size, -1)
    dcg = (sorted_rels[:, :k] / denom[:, :k]).sum(dim=1)

    ideal_sorted_rels, _ = torch.sort(relevances, dim=1, descending=True)
    idcg = (ideal_sorted_rels[:, :k] / denom[:, :k]).sum(dim=1)

    ndcgs = torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))
    return ndcgs.mean().item(), ndcgs.tolist()


class CustomInformationRetrievalEvaluator(SentenceEvaluator):
    """
    This class evaluates an Information Retrieval (IR) setting.

    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document. It measures
    Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)

    Example:
        ::

            import random
            from sentence_transformers import SentenceTransformer
            from sentence_transformers.evaluation import InformationRetrievalEvaluator
            from datasets import load_dataset

            # Load a model
            model = SentenceTransformer('all-mpnet-base-v2')

            # Load the Quora IR dataset (https://huggingface.co/datasets/BeIR/quora, https://huggingface.co/datasets/BeIR/quora-qrels)
            corpus = load_dataset("BeIR/quora", "corpus", split="corpus")
            queries = load_dataset("BeIR/quora", "queries", split="queries")
            relevant_contexts_data = load_dataset("BeIR/quora-qrels", split="validation")

            # Shrink the corpus size heavily to only the relevant documents + 10,000 random documents
            required_context_ids = list(map(str, relevant_contexts_data["corpus-id"]))
            required_context_ids += random.sample(corpus["_id"], k=10_000)
            corpus = corpus.filter(lambda x: x["_id"] in required_context_ids)

            # Convert the datasets to dictionaries
            corpus = dict(zip(corpus["_id"], corpus["text"]))  # Our corpus (cid => document)
            queries = dict(zip(queries["_id"], queries["text"]))  # Our queries (qid => question)
            relevant_contexts = {}  # Query ID to relevant documents (qid => set([relevant_cids])
            for qid, context_ids in zip(relevant_contexts_data["query-id"], relevant_contexts_data["corpus-id"]):
                qid = str(qid)
                context_ids = str(context_ids)
                if qid not in relevant_contexts:
                    relevant_contexts[qid] = set()
                relevant_contexts[qid].add(context_ids)

            # Given queries, a corpus and a mapping with relevant documents, the InformationRetrievalEvaluator computes different IR metrics.
            ir_evaluator = InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_contexts=relevant_contexts,
                name="BeIR-quora-dev",
            )
            results = ir_evaluator(model)
            '''
            Information Retrieval Evaluation of the model on the BeIR-quora-dev dataset:
            Queries: 5000
            Corpus: 17476

            Score-Function: cosine
            Accuracy@1: 96.26%
            Accuracy@3: 99.38%
            Accuracy@5: 99.74%
            Accuracy@10: 99.94%
            Precision@1: 96.26%
            Precision@3: 43.01%
            Precision@5: 27.66%
            Precision@10: 14.58%
            Recall@1: 82.93%
            Recall@3: 96.28%
            Recall@5: 98.38%
            Recall@10: 99.55%
            MRR@10: 0.9782
            NDCG@10: 0.9807
            MAP@100: 0.9732
            Score-Function: dot
            Accuracy@1: 96.26%
            Accuracy@3: 99.38%
            Accuracy@5: 99.74%
            Accuracy@10: 99.94%
            Precision@1: 96.26%
            Precision@3: 43.01%
            Precision@5: 27.66%
            Precision@10: 14.58%
            Recall@1: 82.93%
            Recall@3: 96.28%
            Recall@5: 98.38%
            Recall@10: 99.55%
            MRR@10: 0.9782
            NDCG@10: 0.9807
            MAP@100: 0.9732
            '''
            print(ir_evaluator.primary_metric)
            # => "BeIR-quora-dev_cosine_map@100"
            print(results[ir_evaluator.primary_metric])
            # => 0.9732046108457585
    """

    def __init__(
        self,
        queries: List[dict[str, str]],  # qid => query
        contexts: List[dict[str, str]],  # cid => doc
        relevant_contexts: List[dict[str, set[str]]],  # qid => Set[cid]
        mrr_at_k: list[int] = [10],
        ndcg_at_k: list[int] = [10],
        accuracy_at_k: list[int] = [1, 3, 5, 10],
        precision_recall_at_k: list[int] = [1, 3, 5, 10],
        map_at_k: list[int] = [100],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = "",
        write_csv: bool = True,
        truncate_dim: int | None = None,
        score_functions: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
            SimilarityFunction.COSINE.value: cos_sim,
            SimilarityFunction.DOT_PRODUCT.value: dot_score,
        },  # Score function, higher=more similar
        main_score_function: str | SimilarityFunction | None = None,
        query_prompt: str | None = None,
        query_prompt_name: str | None = None,
        context_prompt: str | None = None,
        context_prompt_name: str | None = None,
        graded_ndcg_queries: List[str] | None = None,
        graded_ndcg_positives: List[str] | None = None,
        graded_ndcg_negatives: List[List[str]] | None = None,
        graded_ndcg_relevancies: List[list] | None = None,
        graded_ndcg_at_k: list[int] = [5, 10, 20, 30],
    ) -> None:
        """
        Initializes the InformationRetrievalEvaluator.

        Args:
            queries (List[Dict[str, str]]): A dictionary mapping query IDs to queries.
            corpus (List[Dict[str, str]]): A dictionary mapping document IDs to contexts.
            relevant_contexts (List([Dict[str, Set[str]]]): A dictionary mapping query IDs to a set of relevant document IDs.
            mrr_at_k (List[int]): A list of integers representing the values of k for MRR calculation. Defaults to [10].
            ndcg_at_k (List[int]): A list of integers representing the values of k for NDCG calculation. Defaults to [10].
            accuracy_at_k (List[int]): A list of integers representing the values of k for accuracy calculation. Defaults to [1, 3, 5, 10].
            precision_recall_at_k (List[int]): A list of integers representing the values of k for precision and recall calculation. Defaults to [1, 3, 5, 10].
            map_at_k (List[int]): A list of integers representing the values of k for MAP calculation. Defaults to [100].
            show_progress_bar (bool): Whether to show a progress bar during evaluation. Defaults to False.
            batch_size (int): The batch size for evaluation. Defaults to 32.
            name (str): A name for the evaluation. Defaults to "".
            write_csv (bool): Whether to write the evaluation results to a CSV file. Defaults to True.
            truncate_dim (int, optional): The dimension to truncate the embeddings to. Defaults to None.
            score_functions (Dict[str, Callable[[Tensor, Tensor], Tensor]]): A dictionary mapping score function names to score functions. Defaults to {SimilarityFunction.COSINE.value: cos_sim, SimilarityFunction.DOT_PRODUCT.value: dot_score}.
            main_score_function (Union[str, SimilarityFunction], optional): The main score function to use for evaluation. Defaults to None.
            query_prompt (str, optional): A prompt to use for the queries. Defaults to None.
            query_prompt_name (str, optional): A name for the query prompt. Defaults to None.
            context_prompt (str, optional): A prompt to use for the corpus. Defaults to None.
            context_prompt_name (str, optional): A name for the corpus prompt. Defaults to None.
        """
        super().__init__()
        self.queries = queries
        self.contexts = contexts

        self.relevant_contexts = relevant_contexts
        assert len(queries) == len(contexts) == len(relevant_contexts)

        self.query_prompt = query_prompt
        self.query_prompt_name = query_prompt_name
        self.context_prompt = context_prompt
        self.context_prompt_name = context_prompt_name

        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys()))
        self.main_score_function = SimilarityFunction(main_score_function) if main_score_function else None
        self.truncate_dim = truncate_dim

        self.graded_ndcg_queries = graded_ndcg_queries
        self.graded_ndcg_positives = graded_ndcg_positives
        self.graded_ndcg_negatives = graded_ndcg_negatives
        self.graded_ndcg_relevancies = graded_ndcg_relevancies
        self.graded_ndcg_at_k = graded_ndcg_at_k

        if name:
            name = "_" + name

        self.csv_file: str = "Information-Retrieval_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]

        for score_name in self.score_function_names:
            for k in accuracy_at_k:
                self.csv_headers.append(f"{score_name}-Accuracy@{k}")

            for k in precision_recall_at_k:
                self.csv_headers.append(f"{score_name}-Precision@{k}")
                self.csv_headers.append(f"{score_name}-Recall@{k}")

            for k in mrr_at_k:
                self.csv_headers.append(f"{score_name}-MRR@{k}")

            for k in ndcg_at_k:
                self.csv_headers.append(f"{score_name}-NDCG@{k}")

            for k in map_at_k:
                self.csv_headers.append(f"{score_name}-MAP@{k}")

            if self.graded_ndcg_queries:
                for k in graded_ndcg_at_k:
                    self.csv_headers.append(f"{score_name}-Graded-NDCG@{k}")

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
        *args,
        **kwargs,
    ) -> dict[str, float]:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""

        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(f"Information Retrieval Evaluation of the model on the {self.name} dataset{out_txt}:")

        scores = self.compute_metrices(model, *args, **kwargs)

        # Write results to disc
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.accuracy_at_k:
                    output_data.append(scores[name]["accuracy@k"][k])

                for k in self.precision_recall_at_k:
                    output_data.append(scores[name]["precision@k"][k])
                    output_data.append(scores[name]["recall@k"][k])

                for k in self.mrr_at_k:
                    output_data.append(scores[name]["mrr@k"][k])

                for k in self.ndcg_at_k:
                    output_data.append(scores[name]["ndcg@k"][k])

                for k in self.map_at_k:
                    output_data.append(scores[name]["map@k"][k])

                if self.graded_ndcg_queries and "graded_ndcg@k" in scores[name]:
                    for k in self.graded_ndcg_at_k:
                        output_data.append(scores[name]["graded_ndcg@k"][k])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        if not self.primary_metric:
            if self.main_score_function is None:
                score_function = max(
                    [(name, scores[name]["map@k"][max(self.map_at_k)]) for name in self.score_function_names],
                    key=lambda x: x[1],
                )[0]
                self.primary_metric = f"{score_function}_map@{max(self.map_at_k)}"
            else:
                self.primary_metric = f"{self.main_score_function.value}_map@{max(self.map_at_k)}"

        metrics = {
            f"{score_function}_{metric_name.replace('@k', '@' + str(k))}": value
            for score_function, values_dict in scores.items()
            for metric_name, values in values_dict.items()
            for k, value in values.items()
        }
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics)
        return metrics

    def compute_metrices(
        self, model: SentenceTransformer, context_model=None, context_embeddings: Tensor = None
    ) -> dict[str, float]:
        if context_model is None:
            context_model = model

        max_k = max(
            max(self.mrr_at_k),
            max(self.ndcg_at_k),
            max(self.accuracy_at_k),
            max(self.precision_recall_at_k),
            max(self.map_at_k),
        )

        queries_results = defaultdict(dict)
        queries_gold_result = defaultdict(dict)
        for queries, contexts, relevant_contexts in tqdm(
            zip(self.queries, self.contexts, self.relevant_contexts),
            desc="Document Levele Evaluation",
            disable=not self.show_progress_bar,
        ):
            queries_ids = list(queries.keys())
            queries = [queries[queries_id] for queries_id in queries_ids]

            gold_contexts = {query_id: relevant_contexts[query_id] for query_id in queries_ids}
            queries_gold_result.update(gold_contexts)

            # Compute embeddings for the queries
            with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(
                self.truncate_dim
            ):
                query_embeddings = model.encode(
                    queries,
                    prompt_name=self.query_prompt_name,
                    prompt=self.query_prompt,
                    show_progress_bar=self.show_progress_bar,
                    batch_size=self.batch_size,
                    convert_to_tensor=True,
                )

            contexts_ids = list(contexts.keys())
            contexts = list(contexts.values())
            # compute embeddings for contexts
            with nullcontext() if self.truncate_dim is None else context_model.truncate_sentence_embeddings(
                self.truncate_dim
            ):
                context_embeddings = context_model.encode(
                    contexts,
                    prompt_name=self.context_prompt_name,
                    prompt=self.context_prompt,
                    show_progress_bar=False,
                    batch_size=self.batch_size,
                    convert_to_tensor=True,
                )

            # Compute cosine similarites
            for name, score_function in self.score_functions.items():
                pair_scores = score_function(query_embeddings, context_embeddings)

                # Get top-k values
                pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(
                    pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=False
                )
                pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

                for query_idx in range(len(queries)):
                    matching_contexts = []
                    for context_idx, score in zip(
                        pair_scores_top_k_idx[query_idx], pair_scores_top_k_values[query_idx]
                    ):
                        matching_contexts.append({"context_id": contexts_ids[context_idx], "score": score})

                    queries_results[name][queries_ids[query_idx]] = matching_contexts

        # Compute scores
        scores = {
            name: self.compute_metrics(
                queries_results=queries_results[name], queries_gold_result=queries_gold_result
            )
            for name in self.score_functions
        }

        # Graded NDCG evaluation (per-sample with negatives and relevancy scores)
        if self.graded_ndcg_queries:
            graded_ndcg_scores = self.compute_graded_ndcg(model, context_model or model)
            for name in self.score_functions:
                scores[name]["graded_ndcg@k"] = graded_ndcg_scores[name]

        # Output
        for name in self.score_function_names:
            logger.info(f"Score-Function: {name}")
            self.output_scores(scores[name])

        return scores

    def compute_graded_ndcg(
        self, model: "SentenceTransformer", context_model: "SentenceTransformer"
    ) -> dict:
        """Compute graded NDCG using per-sample negatives and relevancy scores.

        Each sample has 1 positive and N negatives with graded relevancy.
        The positive is assigned relevance 1.0; negatives use their relevancy
        score (with a -0.1 offset, floored at 0).
        """
        # Batch-encode all queries
        with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(
            self.truncate_dim
        ):
            all_query_embeddings = model.encode(
                self.graded_ndcg_queries,
                prompt_name=self.query_prompt_name,
                prompt=self.query_prompt,
                show_progress_bar=self.show_progress_bar,
                batch_size=self.batch_size,
                convert_to_tensor=True,
            )

        # Batch-encode all candidates (positive + negatives for every sample)
        all_candidates = []
        offsets = []
        for i in range(len(self.graded_ndcg_queries)):
            start = len(all_candidates)
            all_candidates.append(self.graded_ndcg_positives[i])
            all_candidates.extend(self.graded_ndcg_negatives[i])
            offsets.append((start, len(all_candidates)))

        with nullcontext() if self.truncate_dim is None else context_model.truncate_sentence_embeddings(
            self.truncate_dim
        ):
            all_candidate_embeddings = context_model.encode(
                all_candidates,
                prompt_name=self.context_prompt_name,
                prompt=self.context_prompt,
                show_progress_bar=self.show_progress_bar,
                batch_size=self.batch_size,
                convert_to_tensor=True,
            )

        results = {}
        for name, score_function in self.score_functions.items():
            ndcg_per_k = {k: [] for k in self.graded_ndcg_at_k}

            for i in range(len(self.graded_ndcg_queries)):
                start, end = offsets[i]
                query_emb = all_query_embeddings[i:i+1]
                cand_embs = all_candidate_embeddings[start:end]

                pair_scores = score_function(query_emb, cand_embs)

                relevancy = self.graded_ndcg_relevancies[i]
                graded_rel = torch.tensor(
                    [1.0] + [max(0.0, float(r) - 0.1) for r in relevancy],
                    dtype=torch.float32,
                ).unsqueeze(0).to(pair_scores.device)

                for k in self.graded_ndcg_at_k:
                    mean_ndcg, _ = ndcg_at_k_batch(graded_rel, pair_scores, k=k)
                    ndcg_per_k[k].append(mean_ndcg)

            results[name] = {k: float(np.mean(v)) if v else 0.0 for k, v in ndcg_per_k.items()}

        logger.info(f"Graded NDCG evaluation: {len(self.graded_ndcg_queries)} samples")
        for name in self.score_function_names:
            logger.info(f"  Score-Function: {name}")
            for k in self.graded_ndcg_at_k:
                logger.info(f"    Graded-NDCG@{k}: {results[name][k]:.4f}")

        return results

    def compute_metrics(
        self, queries_results: Dict[str, List[Dict]], queries_gold_result: Dict[str, set[str]]
    ):
        # Init score computation values
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}

        # Compute scores on results
        for query_id, query_results in queries_results.items():
            # Sort scores
            top_hits = sorted(query_results, key=lambda x: x["score"], reverse=True)
            query_relevant_contexts = queries_gold_result[query_id]

            # Accuracy@k - We count the result correct, if at least one relevant doc is across the top-k documents
            for k_val in self.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    if hit["context_id"] in query_relevant_contexts:
                        num_hits_at_k[k_val] += 1
                        break

            # Precision and Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit["context_id"] in query_relevant_contexts:
                        num_correct += 1

                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_contexts))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["context_id"] in query_relevant_contexts:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [
                    1 if top_hit["context_id"] in query_relevant_contexts else 0
                    for top_hit in top_hits[0:k_val]
                ]
                true_relevances = [1] * len(query_relevant_contexts)

                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(
                    true_relevances, k_val
                )
                ndcg[k_val].append(ndcg_value)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["context_id"] in query_relevant_contexts:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(k_val, len(query_relevant_contexts))
                AveP_at_k[k_val].append(avg_precision)

        # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(queries_results)

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k])

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])

        for k in MRR:
            MRR[k] /= len(queries_results)

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k])

        return {
            "accuracy@k": num_hits_at_k,
            "precision@k": precisions_at_k,
            "recall@k": recall_at_k,
            "ndcg@k": ndcg,
            "mrr@k": MRR,
            "map@k": AveP_at_k,
        }

    def output_scores(self, scores):
        for k in scores["accuracy@k"]:
            logger.info("Accuracy@{}: {:.2f}%".format(k, scores["accuracy@k"][k] * 100))

        for k in scores["precision@k"]:
            logger.info("Precision@{}: {:.2f}%".format(k, scores["precision@k"][k] * 100))

        for k in scores["recall@k"]:
            logger.info("Recall@{}: {:.2f}%".format(k, scores["recall@k"][k] * 100))

        for k in scores["mrr@k"]:
            logger.info("MRR@{}: {:.4f}".format(k, scores["mrr@k"][k]))

        for k in scores["ndcg@k"]:
            logger.info("NDCG@{}: {:.4f}".format(k, scores["ndcg@k"][k]))

        for k in scores["map@k"]:
            logger.info("MAP@{}: {:.4f}".format(k, scores["map@k"][k]))

        if "graded_ndcg@k" in scores:
            for k in scores["graded_ndcg@k"]:
                logger.info("Graded-NDCG@{}: {:.4f}".format(k, scores["graded_ndcg@k"][k]))

    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
        return dcg


def get_evaluator(dataset):
    # triplet_evaluator = TripletEvaluator(
    #     anchors=dataset.triplet_evaluation_dataset.queries,
    #     positives=dataset.triplet_evaluation_dataset.positives,
    #     negatives=dataset.triplet_evaluation_dataset.negatives,
    #     name="triplet",
    # )

    graded_kwargs = {}
    if hasattr(dataset, "graded_ndcg_evaluation_dataset") and dataset.graded_ndcg_evaluation_dataset.queries:
        graded_kwargs = dict(
            graded_ndcg_queries=dataset.graded_ndcg_evaluation_dataset.queries,
            graded_ndcg_positives=dataset.graded_ndcg_evaluation_dataset.positives,
            graded_ndcg_negatives=dataset.graded_ndcg_evaluation_dataset.negatives,
            graded_ndcg_relevancies=dataset.graded_ndcg_evaluation_dataset.relevancies,
        )

    document_level_evaluator = CustomInformationRetrievalEvaluator(
        queries=dataset.document_level_evaluation_dataset.queries,
        contexts=dataset.document_level_evaluation_dataset.contexts,
        relevant_contexts=dataset.document_level_evaluation_dataset.relevant_contexts,
        **graded_kwargs,
    )
    evaluator = SequentialEvaluator(evaluators=[document_level_evaluator])
    return evaluator
