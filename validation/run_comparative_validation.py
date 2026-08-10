"""Run the publication representative figure and final accuracy comparison."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=100)
    arguments = parser.parse_args()
    subprocess.run(
        [sys.executable, str(ROOT / "validation" / "run_representative_fit.py")],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "validation" / "run_final_accuracy_comparison.py"),
            "--num-trials",
            str(arguments.num_trials),
        ],
        check=True,
    )
