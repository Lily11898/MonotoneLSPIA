"""Reproduce all experiments and figures reported in the SoftwareX manuscript.

The default mode runs the full publication configuration.  ``--smoke`` keeps
the same code paths but uses one accuracy trial, one timing warm-up, one timing
repeat, and the smallest scaling condition.  Smoke outputs are for workflow
verification only and must not replace the archived publication results.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parents[1]
VALIDATION = ROOT / "validation"
DEFAULT_RESULTS = ROOT / "results"

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def run_command(
    label: str,
    arguments: list[str],
    *,
    environment: dict[str, str],
    records: list[dict[str, object]],
) -> None:
    """Run one documented workflow step and append auditable status metadata."""
    command = [sys.executable, *arguments]
    record: dict[str, object] = {
        "label": label,
        "command": command,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    records.append(record)
    try:
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    except subprocess.CalledProcessError as error:
        record["status"] = "failed"
        record["return_code"] = error.returncode
        raise
    else:
        record["status"] = "completed"
        record["return_code"] = 0
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Directory that receives all generated result subdirectories",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a fast workflow check; outputs are not publication results",
    )
    parser.add_argument(
        "--reuse-focused-results",
        type=Path,
        default=None,
        help="Optional compatible accuracy_trials.csv used by the full accuracy script",
    )
    parser.add_argument(
        "--skip-nasa",
        action="store_true",
        help="Skip NASA only when B0005.mat is unavailable",
    )
    parser.add_argument(
        "--download-nasa",
        action="store_true",
        help="Explicitly download and verify B0005.mat when it is not present",
    )
    parser.add_argument(
        "--nasa-data",
        type=Path,
        default=ROOT / "data" / "NASA_B0005" / "B0005.mat",
    )
    arguments = parser.parse_args()
    if arguments.skip_nasa and arguments.download_nasa:
        parser.error("--skip-nasa and --download-nasa cannot be used together")

    output_root = arguments.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    for variable in THREAD_VARIABLES:
        environment[variable] = "1"

    accuracy_trials = 1 if arguments.smoke else 100
    timing_warmups = 1 if arguments.smoke else 2
    timing_repeats = 1 if arguments.smoke else 20
    timing_sizes = [120] if arguments.smoke else [120, 1000, 10000, 100000]
    records: list[dict[str, object]] = []
    manifest = {
        "workflow": "SoftwareX publication reproduction",
        "mode": "smoke" if arguments.smoke else "full",
        "output_root": str(output_root),
        "python": sys.version,
        "platform": platform.platform(),
        "thread_environment": {name: environment[name] for name in THREAD_VARIABLES},
        "commands": records,
    }
    manifest_path = output_root / "publication_workflow_manifest.json"

    try:
        run_command(
            "representative_fit",
            [
                str(VALIDATION / "run_representative_fit.py"),
                "--output",
                str(output_root),
            ],
            environment=environment,
            records=records,
        )

        accuracy_command = [
            str(VALIDATION / "run_final_accuracy_comparison.py"),
            "--num-trials",
            str(accuracy_trials),
            "--output",
            str(output_root / "final_accuracy"),
        ]
        if arguments.reuse_focused_results is not None:
            accuracy_command.extend(
                ["--reuse-focused-results", str(arguments.reuse_focused_results)]
            )
        run_command(
            "final_accuracy",
            accuracy_command,
            environment=environment,
            records=records,
        )

        timing_command = [
            str(VALIDATION / "run_fair_timing_benchmark.py"),
            "--warmups",
            str(timing_warmups),
            "--repeats",
            str(timing_repeats),
            "--scale-sizes",
            *[str(size) for size in timing_sizes],
            "--output",
            str(output_root / "fair_timing_final"),
        ]
        if arguments.smoke:
            timing_command.append("--skip-accuracy-conditions")
        run_command(
            "fair_timing",
            timing_command,
            environment=environment,
            records=records,
        )

        run_command(
            "puromycin",
            [
                str(VALIDATION / "run_puromycin_case_study.py"),
                "--output",
                str(output_root / "puromycin"),
            ],
            environment=environment,
            records=records,
        )

        if not arguments.skip_nasa:
            if not arguments.nasa_data.is_file() and arguments.download_nasa:
                run_command(
                    "fetch_nasa_b0005",
                    [
                        str(VALIDATION / "fetch_nasa_b0005.py"),
                        "--output",
                        str(arguments.nasa_data),
                    ],
                    environment=environment,
                    records=records,
                )
            if not arguments.nasa_data.is_file():
                raise FileNotFoundError(
                    f"NASA B0005 source file not found: {arguments.nasa_data}. "
                    "Run validation/fetch_nasa_b0005.py, pass --download-nasa, "
                    "or provide --nasa-data PATH."
                )
            run_command(
                "nasa_b0005",
                [
                    str(VALIDATION / "run_nasa_b0005_case_study.py"),
                    "--data",
                    str(arguments.nasa_data),
                    "--output",
                    str(output_root / "nasa_b0005"),
                ],
                environment=environment,
                records=records,
            )
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved workflow manifest: {manifest_path}")


if __name__ == "__main__":
    main()
