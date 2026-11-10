"""Dataset loading: JSON files → HF training Dataset + evaluation sets.

File formats are documented in the README ("Data format").
"""
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List

from datasets import Dataset
from omegaconf import DictConfig

from cascl.io_utils import load_file


@dataclass
class TrainSample:
    query: str
    positive: str
    hard_negatives: List[str]
    hard_relevencies: List[float]
    soft_negatives: List[str]
    soft_relevencies: List[float]

    @staticmethod
    def from_dict(sample: Dict):
        return TrainSample(
            query=sample["query"],
            positive=sample["positive"],
            hard_negatives=sample.get("hard_negatives"),
            hard_relevencies=sample.get("hard_relevencies"),
            soft_negatives=sample.get("soft_negatives"),
            soft_relevencies=sample.get("soft_relevencies"),
        )


@dataclass
class TestSample:
    doc_id: str
    query_id: str
    query: str
    positive: str
    negatives: List[str]
    relevency: List[float]

    @staticmethod
    def from_dict(sample: Dict):
        return TestSample(
            doc_id=sample["doc_id"],
            # samples sharing a query_id within a document are one query with several positives
            query_id=sample.get("query_id", sample["query"]),
            query=sample["query"],
            positive=sample["positive"],
            negatives=sample.get("negatives"),
            relevency=sample.get("relevency"),
        )


@dataclass
class DocumentLevelEvaluationDataset:
    queries: List[dict[str, str]] = field(default_factory=lambda: [])
    contexts: List[dict[str, str]] = field(default_factory=lambda: [])
    relevant_contexts: List[dict[str, set[str]]] = field(default_factory=lambda: [])


@dataclass
class GradedNDCGEvaluationDataset:
    queries: List[str] = field(default_factory=list)
    positives: List[str] = field(default_factory=list)
    negatives: List[List[str]] = field(default_factory=list)
    relevancies: List[list] = field(default_factory=list)


@dataclass
class TrainingDataset:
    num_negatives: int
    queries: List[str] = field(default_factory=lambda: [])
    positive_contexts: List[str] = field(default_factory=lambda: [])
    hard_negative_contexts: List[List[str]] = field(default_factory=lambda: [])
    soft_negative_contexts: List[List[str]] = field(default_factory=lambda: [])
    hard_relevency_scores: List[List[float]] = field(default_factory=lambda: [])
    soft_relevency_scores: List[List[float]] = field(default_factory=lambda: [])

    @property
    def training_dataset(self):
        # Column order matters: anchor, positive, hard negatives, soft negatives
        # (the losses split sentence_features by position).
        hard_negatives_dict = {
            f"hard_negative_{n_index}": [hard_negatives[n_index] for hard_negatives in self.hard_negative_contexts]
            for n_index in range(self.num_negatives)
        }
        soft_negatives_dict = {
            f"soft_negative_{n_index}": [soft_negatives[n_index] for soft_negatives in self.soft_negative_contexts]
            for n_index in range(self.num_negatives)
        }
        return Dataset.from_dict(
            {
                "anchor": self.queries,
                "positive": self.positive_contexts,
                "hard_relevency": self.hard_relevency_scores,
                "soft_relevency": self.soft_relevency_scores,
                **hard_negatives_dict,
                **soft_negatives_dict,
            }
        )


class RetrieverDataset:
    def __init__(self, config: DictConfig):
        self.dataset_config = config

        # set seed
        random.seed(self.dataset_config.seed)

        # load dataset
        self.samples, self.doc_id_to_paragraphs, self.test_samples = self.load_dataset()

        # prepare training/evaluation datasets
        self.training_dataset = TrainingDataset(num_negatives=self.dataset_config.num_negatives)
        self.document_level_evaluation_dataset = DocumentLevelEvaluationDataset()
        self.graded_ndcg_evaluation_dataset = GradedNDCGEvaluationDataset()

        self.process_dataset()

        del self.samples, self.doc_id_to_paragraphs, self.test_samples

    def load_dataset(self):
        samples = [TrainSample.from_dict(s) for s in load_file(file_path=self.dataset_config.train_dataset_path)]
        test_samples = [TestSample.from_dict(s) for s in load_file(file_path=self.dataset_config.test_dataset_path)]
        documents = load_file(file_path=self.dataset_config.document_dataset_path)
        doc_id_to_paragraphs = {document["doc_id"]: document["paragraphs"] for document in documents}
        return samples, doc_id_to_paragraphs, test_samples

    def process_dataset(self):
        num_negatives = self.dataset_config.num_negatives

        # ── Training set: keep samples with exactly num_negatives hard and soft negatives ──
        for sample in self.samples:
            if (
                sample.hard_negatives
                and sample.soft_negatives
                and len(sample.hard_relevencies) == num_negatives
                and len(sample.soft_relevencies) == num_negatives
            ):
                self.training_dataset.queries.append(sample.query)
                self.training_dataset.positive_contexts.append(sample.positive)
                self.training_dataset.hard_negative_contexts.append(sample.hard_negatives)
                self.training_dataset.soft_negative_contexts.append(sample.soft_negatives)
                self.training_dataset.hard_relevency_scores.append(sample.hard_relevencies)
                self.training_dataset.soft_relevency_scores.append(sample.soft_relevencies)
        print(f"Training samples: {len(self.training_dataset.queries)} / {len(self.samples)} kept")

        # ── Graded NDCG set: test samples with graded negatives ──
        doc_id_to_test_samples = defaultdict(list)
        for sample in self.test_samples:
            doc_id_to_test_samples[sample.doc_id].append(sample)

            if (sample.negatives is not None
                    and sample.relevency is not None
                    and len(sample.negatives) > 0
                    and len(sample.relevency) == len(sample.negatives)):
                self.graded_ndcg_evaluation_dataset.queries.append(sample.query)
                self.graded_ndcg_evaluation_dataset.positives.append(sample.positive)
                self.graded_ndcg_evaluation_dataset.negatives.append(sample.negatives)
                self.graded_ndcg_evaluation_dataset.relevancies.append(sample.relevency)

        max_ndcg = getattr(self.dataset_config, "max_graded_ndcg_samples", None)
        if max_ndcg and len(self.graded_ndcg_evaluation_dataset.queries) > max_ndcg:
            rng = random.Random(self.dataset_config.seed)
            indices = rng.sample(range(len(self.graded_ndcg_evaluation_dataset.queries)), max_ndcg)
            indices.sort()
            self.graded_ndcg_evaluation_dataset.queries = [self.graded_ndcg_evaluation_dataset.queries[i] for i in indices]
            self.graded_ndcg_evaluation_dataset.positives = [self.graded_ndcg_evaluation_dataset.positives[i] for i in indices]
            self.graded_ndcg_evaluation_dataset.negatives = [self.graded_ndcg_evaluation_dataset.negatives[i] for i in indices]
            self.graded_ndcg_evaluation_dataset.relevancies = [self.graded_ndcg_evaluation_dataset.relevancies[i] for i in indices]
            print(f"Graded NDCG: subsampled to {max_ndcg} samples")

        # ── Document-level set: a random subset of test documents ──
        evaluation_doc_ids = random.sample(
            list(doc_id_to_test_samples.keys()),
            min(self.dataset_config.num_eval_documents, len(doc_id_to_test_samples)),
        )
        for doc_id in evaluation_doc_ids:
            if doc_id not in self.doc_id_to_paragraphs:
                print("No paragraphs available for doc_id", doc_id)
                continue

            document_contexts = self.doc_id_to_paragraphs[doc_id]

            key_to_query_id = defaultdict(lambda: str(uuid.uuid4()))
            context_to_context_id = defaultdict(lambda: str(uuid.uuid4()))

            id_to_queries = {}
            id_to_relevant_contexts = defaultdict(set)
            for sample in doc_id_to_test_samples[doc_id]:
                if sample.positive not in document_contexts:
                    continue

                query_id = key_to_query_id[sample.query_id]
                id_to_queries[query_id] = sample.query
                id_to_relevant_contexts[query_id].add(context_to_context_id[sample.positive])

            if not id_to_queries:
                continue
            self.document_level_evaluation_dataset.queries.append(id_to_queries)
            self.document_level_evaluation_dataset.relevant_contexts.append(id_to_relevant_contexts)
            self.document_level_evaluation_dataset.contexts.append(
                {context_to_context_id[context]: context for context in document_contexts}
            )
