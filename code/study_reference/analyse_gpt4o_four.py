"""STUDY RECORD, NOT PART OF THE REPRODUCTION PATH.

This script ran during the study against the working tree, which held the full 10,198 item corpus
and the raw per-condition inference shards. Neither is part of the release, so this file cannot
execute here and is not imported by anything that can. It ships because the procedure it encodes is
worth reading: it is the four-condition PoT comparison, run over the raw shards before the intervention matrix was derived from them.

Nothing in `code/analysis`, `code/figures`, `code/selector` or `code/router` depends on this file.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py


import json
import math
import pathlib
import statistics
import sys
from collections import Counter, defaultdict

RUNS = BUNDLE / "runs" / "graph-pot"
SWITCH_EPOCH = pathlib.Path("/tmp/route_switch_epoch")
LABELS = pathlib.Path(__file__).resolve().parent / "difficulty_v1.json"

CONDITIONS = {
    2: "gpt4o-bupt-table5-official-full-10198.jsonl",
    3: "gpt4o-learned-graph-full-10198.jsonl",
    4: "gpt4o-graph-verifier-full-10198.jsonl",
}


def load_arm(name: str) -> dict[str, dict]:
    """Union the materialised jsonl with the per-item record files, newest wins."""
    base = RUNS / name
    out: dict[str, dict] = {}
    if base.exists():
        for line in base.open(encoding="utf-8"):
            if line.strip():
                value = json.loads(line)
                out[str(value["id"])] = value
    records = base.with_suffix(base.suffix + ".records")
    if records.is_dir():
        for path in records.glob("*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            out[str(value["id"])] = value | {"_mtime": path.stat().st_mtime}
    return out


def mcnemar(a: dict, b: dict, ids: list[str]) -> tuple[int, int, float]:
    """Exact two-sided McNemar. b_count = a right and b wrong, c_count = the reverse."""
    right = lambda rec: bool(rec.get("correct"))
    b_count = sum(1 for i in ids if right(a[i]) and not right(b[i]))
    c_count = sum(1 for i in ids if not right(a[i]) and right(b[i]))
    n = b_count + c_count
    if n == 0:
        return b_count, c_count, 1.0
    tail = sum(math.comb(n, k) for k in range(min(b_count, c_count) + 1))
    return b_count, c_count, min(1.0, 2 * tail / 2 ** n)


def benjamini_hochberg(pairs: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(pairs, key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 1.0
    for rank in range(total, 0, -1):
        name, p = ordered[rank - 1]
        running = min(running, p * total / rank)
        adjusted[name] = running
    return adjusted


def main() -> int:
    questions: dict[str, dict] = {}
    for shard in sorted(DATA.glob("*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                questions[str(row["id"])] = row
    bands = json.load(LABELS.open())["labels"] if LABELS.exists() else {}

    conditions = {index: load_arm(name) for index, name in CONDITIONS.items()}
    conditions[1] = {str(json.loads(l)["id"]): json.loads(l)
               for l in DIRECT_POT.open(encoding="utf-8") if l.strip()} if DIRECT_POT.exists() else {}

    print("condition coverage")
    for index in sorted(conditions):
        print(f"  condition {index}: {len(conditions[index]):5d}/10198")
    complete = [i for i in sorted(conditions) if len(conditions[i]) == 10198]
    if len(complete) < 2:
        print("\nfewer than two complete conditions, nothing to compare yet")
        return 0

    print("\naccuracy")
    for index in complete:
        rows = conditions[index]
        acc = sum(1 for v in rows.values() if v.get("correct")) / len(rows)
        print(f"  condition {index}: {acc:.4f}")

    print("\npaired McNemar, BH corrected across the family")
    tests = []
    for left in complete:
        for right in complete:
            if left < right:
                ids = sorted(set(conditions[left]) & set(conditions[right]))
                b, c, p = mcnemar(conditions[left], conditions[right], ids)
                delta = (sum(1 for i in ids if conditions[right][i].get("correct"))
                         - sum(1 for i in ids if conditions[left][i].get("correct"))) / len(ids)
                tests.append((f"condition{left}_vs_arm{right}", p, b, c, delta, len(ids)))
    adjusted = benjamini_hochberg([(name, p) for name, p, *_ in tests])
    for name, p, b, c, delta, n in tests:
        print(f"  {name:16s} delta {delta*100:+6.2f}pp  rescued {c:4d}  hurt {b:4d}  "
              f"net {c-b:+5d}  p {p:.4f}  q {adjusted[name]:.4f}  n={n}")

    if 2 in complete and 3 in complete:
        print("\nroute stratification, the control for the mid-run provider change")
        switch = float(SWITCH_EPOCH.read_text().strip()) if SWITCH_EPOCH.exists() else None
        if switch is None:
            print("  no switch timestamp recorded")
        else:
            groups = defaultdict(list)
            for i in sorted(set(conditions[2]) & set(conditions[3])):
                stamp = conditions[2][i].get("_mtime")
                groups["openrouter" if stamp and stamp < switch else "openai_direct"].append(i)
            for route, ids in groups.items():
                if len(ids) < 30:
                    print(f"  {route}: n={len(ids)} too small, suppressed")
                    continue
                b, c, p = mcnemar(conditions[2], conditions[3], ids)
                a2 = sum(1 for i in ids if conditions[2][i].get("correct")) / len(ids)
                a3 = sum(1 for i in ids if conditions[3][i].get("correct")) / len(ids)
                print(f"  {route:14s} n={len(ids):5d}  arm2 {a2:.4f}  arm3 {a3:.4f}  "
                      f"delta {(a3-a2)*100:+.2f}pp  p {p:.4f}")
            print("  Conditions 2 and 3 share a route per item, so a route effect cancels in the")
            print("  paired difference. Agreement across the two strata is the check.")

        print("\nretrieval reach, the ceiling that bounds condition 2")
        injected = Counter()
        for i, rec in conditions[2].items():
            injected[min(int(rec.get("selected_count") or 0), 3)] += 1
        for k in sorted(injected):
            print(f"  {k} functions injected: {injected[k]:5d} ({injected[k]/len(conditions[2]):.1%})")

    if bands:
        print("\nby difficulty band")
        for index in complete:
            line = [f"  condition {index}:"]
            for band in ("easy", "medium", "hard"):
                ids = [i for i in conditions[index] if bands.get(i, {}).get("band") == band]
                if ids:
                    acc = sum(1 for i in ids if conditions[index][i].get("correct")) / len(ids)
                    line.append(f"{band} {acc:.3f} (n={len(ids)})")
            print("  ".join(line))

    if 4 in complete:
        print("\nverifier")
        fired = [v for v in conditions[4].values() if v.get("verifier_called")]
        print(f"  fired on {len(fired)} items ({len(fired)/len(conditions[4]):.1%})")
        if fired:
            print(f"  verdicts: {dict(Counter(str(v.get('verdict')) for v in fired))}")
            base = conditions[3] if 3 in complete else conditions[2]
            rescued = sum(1 for v in fired if v.get("correct") and not base[str(v["id"])].get("correct"))
            hurt = sum(1 for v in fired if not v.get("correct") and base[str(v["id"])].get("correct"))
            print(f"  rescued {rescued}  hurt {hurt}  net {rescued-hurt:+d}")

    print("\nexecution outcomes, the PoT sandbox")
    for index in complete:
        codes = Counter(str(v.get("error_code")) for v in conditions[index].values() if v.get("error_code"))
        blocked = sum(codes.values())
        print(f"  condition {index}: {blocked} programs rejected ({blocked/len(conditions[index]):.1%}), "
              f"top {dict(codes.most_common(3))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
