#!/usr/bin/env bash
# One-click smoke test. Offline, no GPU, seconds.
set -euo pipefail
cd "$(dirname "$0")"
echo "== 1/3  every python file compiles =="
python -m compileall -q code tests
echo "== 2/3  bundle self-test =="
python tests/test_bundle.py
echo "== 3/3  deterministic gate inference on the public partition =="
python code/router/gate_infer.py
echo
echo "all checks passed"
