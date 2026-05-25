#!/usr/bin/env python3
"""Board entrypoint for PyBaMM model assets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BOARD_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = BOARD_ROOT / "outputs"
BASELINE_SCRIPT = BOARD_ROOT / "closed_loop_model" / "src" / "baseline" / "run_bol_simulation.py"
REPRO_CONFIG_ROOT = BOARD_ROOT / "reproduce_li2024_model" / "configs" / "cases"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Navigate or execute model-board tasks.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("overview", help="Print model-board overview.")

    run_baseline = subparsers.add_parser("run-baseline", help="Run the baseline DFN model.")
    run_baseline.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra args passed to baseline runner.")

    return parser


def print_overview() -> None:
    print("02_pybamm_model")
    print("")
    print("Model assets:")
    print(f"- Closed-loop model configs: {BOARD_ROOT / 'closed_loop_model' / 'configs'}")
    print(f"- Closed-loop baseline code: {BOARD_ROOT / 'closed_loop_model' / 'src' / 'baseline'}")
    print(f"- Reproduce_Li2024 model core: {BOARD_ROOT / 'reproduce_li2024_model' / 'sim'}")
    print(f"- Reproduce_Li2024 case configs: {REPRO_CONFIG_ROOT}")
    print(f"- Board outputs: {OUTPUTS_ROOT}")


def run_baseline(extra_args: list[str]) -> None:
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    default_output = OUTPUTS_ROOT / "baseline"
    forwarded_args = extra_args[1:] if extra_args and extra_args[0] == "--" else extra_args
    cmd = [sys.executable, str(BASELINE_SCRIPT), "--output-dir", str(default_output), *forwarded_args]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = build_parser().parse_args()
    command = args.command or "overview"
    if command == "overview":
        print_overview()
        return
    if command == "run-baseline":
        run_baseline(args.extra_args)
        return
    raise ValueError(f"Unsupported command: {command}")


if __name__ == "__main__":
    main()
