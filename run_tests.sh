#!/usr/bin/env bash
# One-click check. Offline, no GPU, under a minute.
#
# This runs the same things the CI workflow runs, in the same order: it compiles every file, runs
# the self-test, and then actually executes the commands the README's Quick Start documents. The
# last part is the point. Compiling and hashing a script proves nothing about whether it can read
# what it needs, and an earlier revision of this bundle passed both while every documented command
# failed on any machine but the curation one.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/4  every python file compiles =="
python -m compileall -q code tests

echo "== 2/4  bundle self-test, 22 assertions =="
python tests/test_bundle.py

echo "== 3/4  the documented quick start, actually executed =="
( cd code/analysis && python why_hard372.py        >/dev/null && echo "  ok  why_hard372.py" )
( cd code/analysis && python where_graph_wins.py   >/dev/null && echo "  ok  where_graph_wins.py" )
( cd code/analysis && python rq2_error_analysis.py >/dev/null && echo "  ok  rq2_error_analysis.py" )
( cd code/figures  && python make_taxonomy_figure.py >/dev/null && echo "  ok  make_taxonomy_figure.py" )

echo "== 4/4  deterministic gate inference on the public partition =="
python code/router/gate_infer.py

echo
echo "all checks passed"
