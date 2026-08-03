"""UPSTREAM REFERENCE, NOT RUNNABLE FROM THIS BUNDLE.

This file needs finexam_eval, torch, transformers, which are not part of this release. It is shipped so
the procedure can be read and checked, not executed. Nothing in the reproduction path
imports it. See code/selector/README.md for what is and is not re-executable.
"""

from __future__ import annotations

"""Build the frozen BUPT function graph artifact.

The graph is built ONLY from the function corpus. It never reads FinExam items, answers,
model outputs, or correctness. Two relations are unioned:

  1. shared `article_title`. Editorial grouping, orthogonal to embedding similarity, so it
     recovers functions that dense retrieval ranks far apart. 1,864 titles, of which 823
     hold two or more functions.
  2. Contriever nearest neighbours. Needed because 1,041 functions are the only member of
     their title and would otherwise have degree zero, making graph expansion a no-op for a
     third of the corpus.

Encoding matches the retrieval path exactly: CLS pooling, no explicit L2 normalisation,
inner product, pinned revision, offline.

Usage:
  PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    .venv-contriever/bin/python scripts/build_bupt_function_graph.py \
      assets/functions-article-all.json artifacts/bupt-function-graph.json \
      --finexam-evaluation-ids artifacts/finexam-evaluation-ids.json --knn 4
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path

from finexam_eval.bupt_table5_protocol import function_catalog_hash
from finexam_eval.core import canonical_hash, load_corpus
from finexam_eval.graph_protocol import GRAPH_ARTIFACT_KIND, GRAPH_SOURCE_DATASET
from finexam_eval.retrieval import (BUPT_REPO_COMMIT, CONTRIEVER_MODEL_ID,
                                    CONTRIEVER_REVISION)

BUILDER = ("shared_article_title_union_contriever_knn; CLS pooling; no explicit L2 "
           "normalisation; inner product; corpus only, no FinExam or answer signal")


def encode(texts: list[str], batch_size: int = 32):
    import torch  # torch must load before faiss on macOS arm64
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        CONTRIEVER_MODEL_ID, revision=CONTRIEVER_REVISION, local_files_only=True)
    model = AutoModel.from_pretrained(
        CONTRIEVER_MODEL_ID, revision=CONTRIEVER_REVISION, local_files_only=True).eval()
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                          max_length=512, return_tensors="pt")
        with torch.no_grad():
            chunks.append(model(**batch).last_hidden_state[:, 0, :])
    return torch.cat(chunks, dim=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--finexam-evaluation-ids", type=Path, required=True)
    parser.add_argument("--knn", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.knn <= 16:
        raise SystemExit("knn must be 1..16")

    records = load_corpus(args.corpus)
    ids = [str(record.id) for record in records]
    catalog_ids = set(ids)
    if len(catalog_ids) != len(ids):
        raise SystemExit("duplicate function id in corpus")

    finexam_ids = set(json.loads(args.finexam_evaluation_ids.read_text(encoding="utf-8"))["ids"])
    overlap = len(catalog_ids & finexam_ids)
    if overlap:
        raise SystemExit(f"FinExam_overlap_must_be_zero: {overlap}")

    neighbours: dict[str, set[str]] = {item: set() for item in ids}

    by_title: dict[str, list[str]] = collections.defaultdict(list)
    for record in records:
        by_title[str(getattr(record, "title", ""))].append(str(record.id))
    title_edges = 0
    for members in by_title.values():
        if len(members) < 2:
            continue
        for source in members:
            for target in members:
                if source != target:
                    neighbours[source].add(target)
                    title_edges += 1

    import torch
    matrix = encode([str(record.function) for record in records], args.batch_size)
    scores = matrix @ matrix.T
    scores.fill_diagonal_(float("-inf"))
    knn_edges = 0
    for index, row in enumerate(torch.topk(scores, args.knn, dim=1).indices.tolist()):
        source = ids[index]
        for position in row:
            target = ids[position]
            if target != source and target not in neighbours[source]:
                neighbours[source].add(target)
                knn_edges += 1

    adjacency = {item: sorted(neighbours[item]) for item in ids if neighbours[item]}
    artifact = {
        "schema_version": 1,
        "artifact_kind": GRAPH_ARTIFACT_KIND,
        "function_catalog_hash": function_catalog_hash(records),
        "provenance": {
            "source_dataset": GRAPH_SOURCE_DATASET,
            "source_commit": BUPT_REPO_COMMIT,
            "builder": BUILDER,
            "source_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
            "finexam_overlap_audit": {"count": 0, "method": "code_computed_id_intersection"},
        },
        "adjacency": adjacency,
    }
    artifact["artifact_sha256"] = canonical_hash(
        {key: value for key, value in artifact.items() if key != "artifact_sha256"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=1, sort_keys=True),
                           encoding="utf-8")
    degrees = sorted(len(value) for value in adjacency.values())
    print(json.dumps({
        "artifact_sha256": artifact["artifact_sha256"],
        "file_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "function_catalog_hash": artifact["function_catalog_hash"],
        "nodes_with_neighbours": len(adjacency),
        "isolated_nodes": len(ids) - len(adjacency),
        "title_edges": title_edges,
        "knn_edges": knn_edges,
        "degree_min": degrees[0], "degree_median": degrees[len(degrees) // 2],
        "degree_max": degrees[-1],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
