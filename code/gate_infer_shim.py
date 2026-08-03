"""Import shim so the test suite can reach build_features without a package layout."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "router"))

from gate_infer import build_features  # noqa: E402,F401
