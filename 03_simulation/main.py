#!/usr/bin/env python3
"""Board entrypoint for simulation workflows and TensorBoard export."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BOARD_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = BOARD_ROOT / "outputs"
TB_ROOT = OUTPUTS_ROOT / "tensorboard"
CLOSED_LOOP_WORKFLOW = BOARD_ROOT / "closed_loop" / "src" / "workflow" / "closed_loop_pipeline.py"
REPRO_SCRIPT = BOARD_ROOT / "reproduce_li2024" / "scripts" / "run_from_json.py"
TB_EXPORT_SCRIPT = BOARD_ROOT / "tensorboard_export.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or inspect simulation-board workflows.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("overview", help="Print simulation-board overview.")

    run_closed_loop = subparsers.add_parser("run-closed-loop", help="Run the closed-loop workflow.")
    run_closed_loop.add_argument("extra_args", nargs=argparse.REMAINDER)

    run_reproduce = subparsers.add_parser("run-reproduce", help="Run the reproduce_li2024 workflow.")
    run_reproduce.add_argument("extra_args", nargs=argparse.REMAINDER)

    export_tb = subparsers.add_parser("export-tensorboard", help="Export simulation artifacts to TensorBoard logs.")
    export_tb.add_argument("--target", choices=["closed_loop", "reproduce", "all"], default="all")
    export_tb.add_argument("--case", default=None)

    subparsers.add_parser("tensorboard", help="Print the recommended TensorBoard command.")
    return parser


def print_overview() -> None:
    print("03_simulation")
    print("")
    print("Entrypoints:")
    print(f"- Closed-loop workflow: {CLOSED_LOOP_WORKFLOW}")
    print(f"- Reproduce_Li2024 workflow: {REPRO_SCRIPT}")
    print(f"- TensorBoard exporter: {TB_EXPORT_SCRIPT}")
    print("")
    print("Board outputs:")
    print(f"- TensorBoard root: {TB_ROOT}")
    print(f"- Closed-loop results: {BOARD_ROOT / 'closed_loop' / 'results'}")
    print(f"- Reproduce outputs: {BOARD_ROOT / 'reproduce_li2024' / 'outputs'}")


def run_python(script: Path, extra_args: list[str]) -> None:
    cmd = [sys.executable, str(script), *extra_args]
    subprocess.run(cmd, check=True)


def print_tensorboard_command() -> None:
    print(f"tensorboard --logdir {TB_ROOT}")


def main() -> None:
    args = build_parser().parse_args()
    command = args.command or "overview"
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    TB_ROOT.mkdir(parents=True, exist_ok=True)

    if command == "overview":
        print_overview()
        return
    if command == "run-closed-loop":
        run_python(CLOSED_LOOP_WORKFLOW, args.extra_args)
        return
    if command == "run-reproduce":
        run_python(REPRO_SCRIPT, args.extra_args)
        return
    if command == "export-tensorboard":
        export_args = ["--target", args.target]
        if args.case:
            export_args.extend(["--case", args.case])
        run_python(TB_EXPORT_SCRIPT, export_args)
        return
    if command == "tensorboard":
        print_tensorboard_command()
        return
    raise ValueError(f"Unsupported command: {command}")


if __name__ == "__main__":
    main()
