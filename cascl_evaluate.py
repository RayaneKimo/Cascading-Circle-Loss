"""
Standalone evaluation for CasCL-trained models (produces the reported results).

Computes, on the test set:
  - Document-level retrieval: Recall@k, MRR@k, MAP@k. For each document
    (doc_id), the candidates are all its paragraphs + the samples'
    ``negative_contexts``; queries are grouped by ``query_id`` so one
    query can have several relevant paragraphs.
  - Graded NDCG@k over [positive(s) + ``negatives``], using the per-negative
    ``relevency`` scores as gains (positive = 1.0).

Usage:
    python cascl_evaluate.py \
        --model_path models/<run_name>/checkpoint-2000 \
        --test_dataset_path data/test.json \
        --document_dataset_path data/documents.json \
        --output_file eval/<run_name>_checkpoint-2000.json
"""

import json
import argparse
import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple
from tqdm import tqdm

from sentence_transformers import SentenceTransformer


# ═══════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════

def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    r = relevances[:k]
    if len(r) == 0:
        return 0.0
    return float(np.sum((2**r - 1) / np.log2(np.arange(len(r)) + 2)))

def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    actual = dcg_at_k(relevances, k)
    ideal = dcg_at_k(np.sort(relevances)[::-1], k)
    return actual / ideal if ideal > 0 else 0.0

def precision_at_k(relevances: np.ndarray, k: int, threshold: float = 0.5) -> float:
    topk = relevances[:k]
    return float(np.sum(topk >= threshold)) / k if len(topk) > 0 else 0.0

def recall_at_k(relevances: np.ndarray, k: int, total_relevant: int, threshold: float = 0.5) -> float:
    if total_relevant == 0:
        return 0.0
    return float(np.sum(relevances[:k] >= threshold)) / total_relevant

def mrr_at_k(relevances: np.ndarray, k: int, threshold: float = 0.5) -> float:
    for i, r in enumerate(relevances[:k]):
        if r >= threshold:
            return 1.0 / (i + 1)
    return 0.0

def map_at_k(relevances: np.ndarray, k: int, total_relevant: int, threshold: float = 0.5) -> float:
    topk = relevances[:k]
    hits = 0
    sum_p = 0.0
    for i, r in enumerate(topk):
        if r >= threshold:
            hits += 1
            sum_p += hits / (i + 1)
    denom = min(k, total_relevant)
    return sum_p / denom if denom > 0 else 0.0

def pairwise_accuracy(sims: np.ndarray, rels: np.ndarray) -> float:
    n = len(sims)
    concordant = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if rels[i] == rels[j]:
                continue
            total += 1
            if (rels[i] > rels[j]) == (sims[i] > sims[j]):
                concordant += 1
    return concordant / total if total > 0 else 1.0


# ═══════════════════════════════════════════════════════════════
#  Graded NDCG evaluation (uses the per-negative `relevency` scores)
# ═══════════════════════════════════════════════════════════════
#
# Convention:
#   - Each test sample has 1 query, 1 positive, N negatives and a parallel
#     list of N `relevency` scores in [0, 1] (higher = more relevant).
#   - Gains: positive → 1.0, negative j → max(0, relevency_j).
# ═══════════════════════════════════════════════════════════════

def ndcg_at_k_batch(
    relevances: torch.Tensor,
    scores: torch.Tensor,
    k: int,
) -> Tuple[float, List[float]]:
    """Mean NDCG@k for a batch of queries.

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




# ═══════════════════════════════════════════════════════════════
#  Document-Level Retrieval Evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate_document_retrieval(
    model,
    test_samples: List[dict],
    documents: List[dict],
    ks: List[int] = [1, 3, 5, 10],
    k_ndcg: List[int] = [5, 10, 20, 30],
    batch_size: int = 128,
    threshold: float = 0.5,
    positive_relevance: float = 1.0,
) -> dict:
    """
    Document-level retrieval evaluation supporting multiple relevant contexts
    per query.

    Samples are grouped by ``query_id`` (or by ``query`` if ``query_id`` is
    absent) within each document.  All ``positive`` values in a group become
    the set of relevant contexts for that query, so one query can have several
    correct paragraphs.

    Binary metrics (recall, mrr, map) are computed over the candidate pool:
    all paragraphs of the document + ``negative_contexts``.

    Graded NDCG is computed over: positives + ``negatives``, using the
    ``relevency`` scores as gains.
    """
    doc_id_to_doc = {}
    for doc in documents:
        doc_id_to_doc[doc["doc_id"]] = doc

    doc_id_to_samples = defaultdict(list)
    for s in test_samples:
        doc_id_to_samples[s["doc_id"]].append(s)

    metric_names = ["recall", "mrr", "map"]
    all_metrics = {f"{m}@{k}": [] for m in metric_names for k in ks}
    all_metrics.update({f"graded_ndcg@{k}": [] for k in k_ndcg})

    total_queries = 0

    for doc_id, samples in tqdm(doc_id_to_samples.items(), desc="Document-level eval"):
        if doc_id not in doc_id_to_doc:
            continue

        doc = doc_id_to_doc[doc_id]
        paragraphs = doc.get("paragraphs", [])
        if not paragraphs:
            continue

        # ── Group samples by query_id: one query, possibly several positives ──
        key_to_samples: Dict[str, List[dict]] = defaultdict(list)
        for sample in samples:
            key = sample.get("query_id", sample["query"])
            key_to_samples[key].append(sample)

        # ── Collect every unique text to encode once per document ──
        # Start with ALL document paragraphs
        all_texts_set: set = set(paragraphs)
        for key_samples in key_to_samples.values():
            for s in key_samples:
                all_texts_set.add(s["positive"])
                for nc in (s.get("negative_contexts") or []):
                    all_texts_set.add(nc)
                for neg in (s.get("negatives") or []):
                    all_texts_set.add(neg)

        all_texts_list = list(all_texts_set)

        text_embs = model.encode(
            all_texts_list, normalize_embeddings=True,
            batch_size=batch_size, show_progress_bar=False,
        )
        text_to_emb = dict(zip(all_texts_list, text_embs))

        # ── Encode deduplicated queries ──
        query_groups = list(key_to_samples.items())
        query_texts = [gs[0]["query"] for _, gs in query_groups]
        q_embs = model.encode(
            query_texts, normalize_embeddings=True,
            batch_size=batch_size, show_progress_bar=False,
        )

        for qi, (key, key_samples) in enumerate(query_groups):
            q_emb = q_embs[qi]
            total_queries += 1

            # Multiple relevant contexts
            positive_set = set(s["positive"] for s in key_samples)
            positive_texts = list(positive_set)

            # ── Binary metrics using negative_contexts ──
            # Context pool = all document paragraphs + any negative_contexts
            # not already in paragraphs
            neg_ctx_set: set = set()
            for s in key_samples:
                for nc in (s.get("negative_contexts") or []):
                    neg_ctx_set.add(nc)

            candidate_set = set(paragraphs) | neg_ctx_set
            candidate_texts = list(candidate_set)
            if len(candidate_texts) < 2:
                continue

            cand_embs = np.array([text_to_emb[t] for t in candidate_texts])
            sims = np.dot(cand_embs, q_emb)
            rels = np.array([1.0 if t in positive_set else 0.0
                             for t in candidate_texts])
            total_relevant = len(positive_texts)

            ranked_idx = np.argsort(-sims)
            ranked_rels = rels[ranked_idx]

            for k in ks:
                all_metrics[f"recall@{k}"].append(
                    recall_at_k(ranked_rels, k, total_relevant, threshold))
                all_metrics[f"mrr@{k}"].append(
                    mrr_at_k(ranked_rels, k, threshold))
                all_metrics[f"map@{k}"].append(
                    map_at_k(ranked_rels, k, total_relevant, threshold))

            # ── Graded NDCG using negatives + relevancy ──
            for s in key_samples:
                negatives = s.get("negatives") or []
                relevancy = s.get("relevency")
                if not (relevancy
                        and len(negatives) > 0
                        and len(relevancy) == len(negatives)):
                    continue

                filtered_negatives = []
                filtered_relevancy = []
                for neg, rel in zip(negatives, relevancy):
                    if neg not in positive_set:
                        filtered_negatives.append(neg)
                        filtered_relevancy.append(rel)

                if not filtered_negatives:
                    continue

                ndcg_texts = positive_texts + filtered_negatives
                ndcg_embs = np.array([text_to_emb[t] for t in ndcg_texts])
                ndcg_sims = np.dot(ndcg_embs, q_emb)

                scores_t = torch.tensor(
                    ndcg_sims, dtype=torch.float32).unsqueeze(0)
                graded_rels = (
                    [float(positive_relevance)] * len(positive_texts)
                    + [max(0.0, float(r)) for r in filtered_relevancy]
                )
                graded_rel_t = torch.tensor(
                    graded_rels, dtype=torch.float32).unsqueeze(0)

                for k in k_ndcg:
                    mean_ndcg, _ = ndcg_at_k_batch(graded_rel_t, scores_t, k=k)
                    all_metrics[f"graded_ndcg@{k}"].append(mean_ndcg)

    results = {}
    for name, values in all_metrics.items():
        results[name] = float(np.mean(values)) if values else 0.0
    results["num_queries"] = total_queries
    results["num_graded_queries"] = len(all_metrics.get(
        f"graded_ndcg@{k_ndcg[0]}", []))

    return results


# ═══════════════════════════════════════════════════════════════
#  Pretty print
# ═══════════════════════════════════════════════════════════════

def print_results(results: dict, title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    metric_groups = defaultdict(dict)
    other = {}
    for key, value in sorted(results.items()):
        if "@" in key:
            name, k = key.rsplit("@", 1)
            metric_groups[name][int(k)] = value
        else:
            other[key] = value

    if metric_groups:
        ks = sorted(set(k for g in metric_groups.values() for k in g))
        header = f"  {'Metric':<15}" + "".join(f"{'@'+str(k):>10}" for k in ks)
        print(header)
        print("  " + "-" * (15 + 10 * len(ks)))
        for m in ["ndcg", "graded_ndcg", "recall", "mrr", "map"]:
            if m in metric_groups:
                row = f"  {m:<15}" + "".join(f"{metric_groups[m].get(k,0):>10.4f}" for k in ks)
                print(row)

    if other:
        print()
        for k, v in sorted(other.items()):
            print(f"  {k:<30} {v:.4f}" if isinstance(v, float) else f"  {k:<30} {v}")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate CasCL-trained model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Trained checkpoint directory or HF model id (e.g. Qwen/Qwen3-Embedding-0.6B)")
    parser.add_argument("--test_dataset_path", type=str, required=True)
    parser.add_argument("--document_dataset_path", type=str, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--k_ndcg", type=int, nargs="+", default=[5, 10, 20, 30, 40])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output_file", type=str, default="eval_results.json")
    parser.add_argument("--no_unsloth", action="store_true",
                        help="Skip the Unsloth loader and use SentenceTransformers directly")
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        help="'flash_attention_2' or 'sdpa'")
    args = parser.parse_args()

    # Load model
    try:
        if args.no_unsloth:
            raise ImportError("Unsloth disabled")
        from unsloth import FastSentenceTransformer
        model = FastSentenceTransformer.from_pretrained(args.model_path, for_inference=True)
        print(f"✓ Loaded via Unsloth: {args.model_path}")
    except Exception:
        model = SentenceTransformer(args.model_path,
                                trust_remote_code=True,
                                tokenizer_kwargs={"model_max_length": args.max_seq_length,  "padding_side": "left"},
                                model_kwargs={"attn_implementation": args.attn_implementation, "device_map": "auto", "torch_dtype": torch.bfloat16},)
        print(f"✓ Loaded via SentenceTransformers: {args.model_path}")

    # Load test data
    with open(args.test_dataset_path) as f:
        test_samples = json.load(f)
    print(f"Test samples: {len(test_samples)}")

    all_results = {}

    # Document-level retrieval eval
    if args.document_dataset_path:
        print("\n▶ Document-level retrieval evaluation")
        with open(args.document_dataset_path) as f:
            documents = json.load(f)
        print(f"Documents: {len(documents)}")

        retrieval_results = evaluate_document_retrieval(
            model, test_samples, documents, ks=args.ks, k_ndcg=args.k_ndcg, batch_size=args.batch_size
        )
        all_results["retrieval"] = retrieval_results
        print_results(retrieval_results, "Document-Level Retrieval Results")

    # Save
    output_file = args.output_file or "eval_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
