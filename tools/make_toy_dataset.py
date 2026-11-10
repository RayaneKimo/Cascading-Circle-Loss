#!/usr/bin/env python3
"""
Generate a tiny synthetic dataset in the format expected by train.py and
cascl_evaluate.py, to check that the pipeline runs end to end.

The texts are meaningless templates: use it for smoke tests, not to measure quality.

Usage:
    python tools/make_toy_dataset.py --output_dir data/
"""
import argparse
import json
import os
import random

TOPICS = ["payment terms", "termination notice", "delivery date", "warranty period",
          "governing law", "confidentiality", "liability cap", "renewal conditions"]
RELEVANCE_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def paragraph(doc_idx: int, topic: str, variant: int) -> str:
    return f"Document {doc_idx}, section on {topic} (version {variant}): the parties agree on the {topic}."


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", default="data")
    p.add_argument("--num_docs", type=int, default=20)
    p.add_argument("--num_negatives", type=int, default=10, help="hard and soft negatives per training sample")
    p.add_argument("--num_test_negatives", type=int, default=40, help="graded negatives per test sample")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    documents, train, test = [], [], []
    for d in range(args.num_docs):
        doc_id = f"doc-{d:04d}"
        paragraphs = [paragraph(d, t, v) for t in TOPICS for v in range(2)]
        documents.append({"doc_id": doc_id, "paragraphs": paragraphs})

        for t_idx, topic in enumerate(TOPICS):
            query = f"What does document {d} say about the {topic}?"
            positive = paragraph(d, topic, 0)
            others = [x for x in paragraphs if x != positive]

            if d % 4:  # 3/4 of the documents for training, 1/4 for test
                soft = [paragraph(d, topic, 1)] + random.sample(others, args.num_negatives - 1)
                train.append({
                    "query": query,
                    "positive": positive,
                    "hard_negatives": [f"Unrelated passage {i} about {random.choice(TOPICS)} in another document."
                                       for i in range(args.num_negatives)],
                    "hard_relevencies": [0.0] * args.num_negatives,
                    "soft_negatives": soft,
                    "soft_relevencies": [0.8] + [random.choice(RELEVANCE_GRID) for _ in soft[1:]],
                })
            else:
                negatives = [f"Candidate passage {i} for document {d} on {random.choice(TOPICS)}."
                             for i in range(args.num_test_negatives)]
                test.append({
                    "doc_id": doc_id,
                    "query_id": f"{doc_id}-{t_idx}",
                    "query": query,
                    "positive": positive,
                    "negative_contexts": random.sample(others, 5),
                    "negatives": negatives,
                    "relevency": [random.choice(RELEVANCE_GRID) for _ in negatives],
                })

    for name, content in [("train.json", train), ("test.json", test), ("documents.json", documents)]:
        with open(os.path.join(args.output_dir, name), "w") as f:
            json.dump(content, f, indent=1, ensure_ascii=False)
    print(f"Wrote {len(train)} train samples, {len(test)} test samples, {len(documents)} documents to {args.output_dir}/")


if __name__ == "__main__":
    main()
