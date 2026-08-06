#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/finexam10k_public_repro.XXXXXX")"
cleanup() {
  find "$TMP_ROOT" -depth -delete
}
trap cleanup EXIT

echo "== 1/5 compile =="
python -m compileall -q code tests

echo "== 2/5 requirement-driven tests =="
python -m unittest discover -s tests -v

echo "== 3/5 public data export path =="
python code/data/build_public_exports.py --out "$TMP_ROOT/public_exports"
python code/tables/make_public_tables.py --out "$TMP_ROOT/public_tables"
python code/figures/make_public_figures.py --out "$TMP_ROOT/public_figures"

echo "== 4/5 documented public reproduction entry point =="
python code/public_reproduce.py

echo "== 5/5 aggregate arithmetic =="
python code/verify_aggregate_claims.py

echo "all checks passed"
