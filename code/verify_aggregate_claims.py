"""Verify arithmetic in aggregate-only held-out and exploratory result manifests."""
from __future__ import annotations

import json
import math
from pathlib import Path
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "aggregates"


def exact_mcnemar(rescue: int, harm: int) -> float:
    n = rescue + harm
    if n == 0:
        return 1.0
    k = min(rescue, harm)
    return min(1.0, 2.0 * float(binom.cdf(k, n, 0.5)))


def verify(name: str, n: int, block: dict, p_tolerance: float = 0.01) -> None:
    delta = 100.0 * (block["rescue"] - block["harm"]) / n
    p_value = exact_mcnemar(block["rescue"], block["harm"])
    if abs(delta - block["delta_pp"]) > 0.015:
        raise AssertionError(f"{name}: delta mismatch {delta} versus {block['delta_pp']}")
    reported_p = float(block["mcnemar_p"])
    if reported_p >= 0.001 and abs(p_value - reported_p) > p_tolerance:
        raise AssertionError(f"{name}: McNemar mismatch {p_value} versus {reported_p}")
    print(f"{name:<42} delta={delta:+.4f} p={p_value:.6g}")


def main() -> int:
    gate = json.loads((DATA / "gate_results.json").read_text(encoding="utf-8"))
    verify("gate heldout", gate["heldout"]["n"], gate["heldout"])
    verify(
        "gate heldout context-complete",
        gate["heldout_context_complete"]["n"],
        gate["heldout_context_complete"],
    )

    rq2 = json.loads((DATA / "rq2_results.json").read_text(encoding="utf-8"))
    for scope in ("full_coverage", "context_complete"):
        n = rq2[scope]["n"]
        for chain in ("cot_deepseek_r1", "pot_gpt4o"):
            for method, block in rq2[scope][chain].items():
                if isinstance(block, dict) and "rescue" in block:
                    verify(f"RQ2 {scope} {chain} {method}", n, block)
    for row in rq2["full_coverage"]["judge_strata"]:
        verify(f"RQ2 judge stratum {row['retained']}", row["n"], row)

    probes = json.loads((DATA / "agentic_probes.json").read_text(encoding="utf-8"))
    verify("post-execution verification", probes["post_execution_verification"]["n"], probes["post_execution_verification"])
    verify("plan-then-solve", probes["plan_then_solve"]["n"], probes["plan_then_solve"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
