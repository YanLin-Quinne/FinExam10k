"""Every path the bundle needs, in one place.

Context-complete by construction: nothing here reaches outside the bundle root. Set FINEXAM_ROOT
if you relocate the directory.
"""
from __future__ import annotations

import os
import pathlib

ROOT = pathlib.Path(os.environ.get("FINEXAM_ROOT",
                                   pathlib.Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
FIGURES = ROOT / "figures"   # 图脚本首次运行时自建，渲染结果不随包发
TABLES = ROOT / "tex"

PUBLIC_ITEMS = DATA / "finexam10k_public_5110.json"
RESPONSE_MATRIX = DATA / "response_matrix_public_5110.json"
INTERVENTION_MATRIX = DATA / "intervention_matrix_public_5110.json"
DIFFICULTY = DATA / "difficulty_labels_public_5110.json"
CONTEXT_DEMO = DATA / "context_completeness_public.json"

DIAG_HARD = DATA / "diagnostic_context_complete_hard.json"
DIAG_ZERO = DATA / "diagnostic_zero_solve.json"

SELECTOR = DATA / "selector"
ROUTER = DATA / "router"
SIDECAR = SELECTOR / "per_item_sidecar_public_5110.json"
GATE_FROZEN = ROUTER / "gate_frozen.json"
HELDOUT_MANIFEST = ROUTER / "heldout_decision_manifest.json"
PUBLIC_DECISION_MANIFEST = ROUTER / "public_decision_manifest.json"
