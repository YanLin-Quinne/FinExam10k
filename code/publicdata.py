"""The single data entry point for everything in this bundle.

Why this module exists. An earlier revision of these scripts kept private path constants built from
the curation machine's home directory, and read the full 10,198 item corpus plus the raw per-run
output shards. Neither ships with the release, so the scripts imported cleanly, compiled cleanly,
and then failed at their first read on any other machine. Nothing here reaches outside the bundle
root, and no home directory appears anywhere in the reproduction path.

What this changes about the numbers. The paper reports over all 10,198 items. This release contains
the 5,110 public mock and practice items, so a script driven by this module recomputes the same
quantity over the public partition and prints a public-partition count. Those counts are smaller
than the paper's, deliberately. `scope_banner()` prints the scope of each run so its output cannot
be mistaken for a full-corpus reproduction.

One property makes the subsetting sound. The difficulty score s_i is computed per item from the
frozen 17-model response matrix, so an item's band does not depend on which other items sit
alongside it. Restricting to the public partition selects a subset of each band rather than
recomputing the bands over fewer items.

Raw runs. Scripts written during the study read per-condition JSONL shards straight from the
inference run directories. The release ships the derived intervention matrix instead, which carries
the parsed prediction for all nine conditions over all 5,110 items. `run()` accepts either the
historical shard filename or the condition key, so those scripts keep their original shape.
"""
from __future__ import annotations

import functools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paths as PATHS  # noqa: E402

PARTITION = "public_mock_practice"
N_PUBLIC = 5110
N_FULL = 10198

# The inference shard each condition was parsed from, kept so study-era scripts resolve unchanged.
RUN_ALIAS = {
    "gpt4o-direct-pot-full-10198.jsonl": "pot_direct",
    "gpt4o-bupt-table5-official-full-10198.jsonl": "pot_function",
    "gpt4o-learned-graph-full-10198.jsonl": "pot_graph",
    "gpt4o-graph-verifier-full-10198.jsonl": "pot_verifier",
    "deepseek_r1_baseline_mcq.jsonl": "cot_direct",
    "deepseek_r1_function_rag_mcq.jsonl": "cot_function_bm25",
    "deepseek_r1_fr_llm_instruct_judge_function_rag_mcq.jsonl": "cot_function_judge",
    "deepseek_r1_fr_all_learned_graph_rag_mcq.jsonl": "cot_graph",
    "deepseek_r1_fr_all_learned_graph_rag_nojudge_mcq.jsonl": "cot_graph_nojudge",
    "deepseek_r1_graph_rag_fr_all_mcq.jsonl": "cot_graph_nojudge",
}


def _read(p: pathlib.Path):
    if not p.is_file():
        raise SystemExit(
            f"missing bundled file: {p}\n"
            f"Every file this module needs ships with the release. If one is absent the checkout "
            f"is incomplete: verify it with `python tests/test_bundle.py`."
        )
    return json.loads(p.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def items() -> dict[str, dict]:
    """The 5,110 released items keyed by id, with full stem, options and metadata."""
    return {r["id"]: r for r in _read(PATHS.PUBLIC_ITEMS)}


@functools.lru_cache(maxsize=1)
def gold() -> dict[str, str]:
    return {i: str(r["answer"]).strip().upper() for i, r in items().items()}


@functools.lru_cache(maxsize=1)
def answerable() -> set[str]:
    """Ids an adjudicator judged answerable from the item's own text.

    3,406 of the 5,110 released items. The complement carries a defect code in
    `answerability_defect`, so a caller can restrict to one failure mode rather than to the whole
    complement. This is the public-partition counterpart of the clean-partition list the study-era
    scripts read from a file that is not part of the release.
    """
    return {i for i, r in items().items() if r["answerable"]}


@functools.lru_cache(maxsize=1)
def difficulty() -> dict[str, dict]:
    """Per-item difficulty_v1: `s` and `band`, plus the two sensitivity variants."""
    return _read(PATHS.DIFFICULTY)["labels"]


def band(name: str = "hard") -> list[str]:
    """Sorted ids in one difficulty band over the public partition."""
    return sorted(i for i, v in difficulty().items() if v["band"] == name)


@functools.lru_cache(maxsize=1)
def context_complete_hard() -> list[str]:
    """Hard-band ids an adjudicator judged answerable from their own text.

    138 of the 372 that define the set over the full corpus. The other 234 are held out.
    """
    return sorted(r["id"] for r in _read(PATHS.DIAG_HARD)["items"])


@functools.lru_cache(maxsize=1)
def zero_solve() -> list[str]:
    """Ids no system in the frozen 17-model panel answered correctly.

    118 of the 188 that define the set over the full corpus. The other 70 are held out.
    """
    return sorted(r["id"] for r in _read(PATHS.DIAG_ZERO)["items"])


@functools.lru_cache(maxsize=1)
def _matrix():
    return _read(PATHS.INTERVENTION_MATRIX)


def run(which: str) -> dict[str, dict]:
    """Parsed predictions for one intervention condition, shaped like the study-era loader.

    `which` is either a condition key such as `pot_graph` or the historical shard filename such
    as `gpt4o-learned-graph-full-10198.jsonl`. The return value is `{id: {"prediction": letter}}`,
    which is the subset of the raw record that every analysis script actually reads.
    """
    key = RUN_ALIAS.get(pathlib.Path(which).name, which)
    m = _matrix()
    if key not in m["conditions"]:
        raise SystemExit(
            f"unknown condition {key!r}. The release carries these nine: "
            f"{', '.join(m['conditions'])}."
        )
    k = m["conditions"].index(key)
    return {i: {"prediction": row[k]} for i, row in m["predictions"].items()}


@functools.lru_cache(maxsize=1)
def sidecar() -> dict[str, dict]:
    """Per-item Direct-call state: prediction, parser and executor status, tokens, latency."""
    return _read(PATHS.SIDECAR)["items"]


def scope_banner(what: str, n: int, n_full: int | None = None) -> None:
    """Print a run's scope so its output cannot be read as a full-corpus number."""
    tail = (f" The paper reports this over all {n_full:,} items."
            if n_full else "")
    print(f"[scope] {what}: {n:,} items from the {PARTITION} partition, "
          f"{N_PUBLIC:,} released of {N_FULL:,}.{tail}")
