"""One-command public reproduction of structural, leaderboard, RQ1, RQ2, and gate checks."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = [
    [sys.executable, str(ROOT / "code" / "diagnostics" / "difficulty_public.py")],
    [sys.executable, str(ROOT / "code" / "analysis" / "rq1_public_diagnostics.py")],
    [sys.executable, str(ROOT / "code" / "analysis" / "rq2_public.py")],
    [sys.executable, str(ROOT / "code" / "router" / "train_gate.py")],
    [sys.executable, str(ROOT / "code" / "router" / "gate_infer.py")],
    [sys.executable, str(ROOT / "code" / "selector" / "audit_frozen.py")],
]


def main() -> int:
    for command in COMMANDS:
        print("\n==>", " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
