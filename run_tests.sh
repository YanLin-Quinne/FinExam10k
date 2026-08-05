#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

TMP_ROOT="${TMPDIR:-/tmp}/finexam10k_public_repro"
rm -rf "$TMP_ROOT"
mkdir -p "$TMP_ROOT"

echo "== 1/8 compile =="
python -m compileall -q code tests

echo "== 2/8 unit tests =="
python -m unittest discover -s tests -v

echo "== 3/8 public data export path =="
python code/data/build_public_exports.py --out "$TMP_ROOT/public_exports"
python code/tables/make_public_tables.py --out "$TMP_ROOT/public_tables"
python code/figures/make_public_figures.py --out "$TMP_ROOT/public_figures"

echo "== 4/8 reproduce public gate selection =="
python code/router/train_gate.py

echo "== 5/8 deterministic public gate inference =="
python code/router/gate_infer.py

echo "== 6/8 selector audit =="
python code/selector/audit_frozen.py

echo "== 7/8 aggregate arithmetic =="
python code/verify_aggregate_claims.py

echo "== 8/8 public analysis smoke tests =="
python code/diagnostics/difficulty_public.py >/dev/null
python code/analysis/rq1_public_diagnostics.py >/dev/null
python code/analysis/rq2_public.py >/dev/null

echo "all checks passed"
