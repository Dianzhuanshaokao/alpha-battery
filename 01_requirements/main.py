#!/usr/bin/env python3
"""Board entrypoint for project requirements and cluster usage guidance."""

from __future__ import annotations

import argparse
from pathlib import Path


BOARD_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = BOARD_ROOT / "outputs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show requirement-board navigation and cluster usage guidance.")
    parser.add_argument(
        "--write-summary",
        action="store_true",
        help="Write the printed overview to 01_requirements/outputs/requirements_summary.txt",
    )
    return parser


def build_summary() -> str:
    return "\n".join(
        [
            "AlphaBattery requirement board",
            "",
            "Key documents:",
            f"- {BOARD_ROOT / 'README.md'}",
            f"- {BOARD_ROOT / 'alpha_closed_loop.md'}",
            f"- {BOARD_ROOT / 'reproduce_li2024.md'}",
            f"- {BOARD_ROOT / 'cluster_usage.md'}",
            "",
            "Main workflow:",
            "- Read requirements and data contracts first",
            "- Use 02_pybamm_model for model assets",
            "- Use 03_simulation for executable simulation pipelines",
            "- Use 04_rl_optimization for SmartCharging training and optimization",
            "",
            "Cluster usage highlights:",
            "- Remote simulations and training are expected to run on 121.48.164.50",
            "- GPU jobs must run after `module load gpu`",
            "- GPU nodes: c1, c2, c3",
            "- Other nodes are CPUCompute",
            "",
            "TensorBoard quick start:",
            "- Simulation logs: tensorboard --logdir 03_simulation/outputs/tensorboard",
            "- RL logs: tensorboard --logdir 04_rl_optimization/outputs/training",
        ]
    )


def main() -> None:
    args = build_parser().parse_args()
    summary = build_summary()
    print(summary)

    if args.write_summary:
        OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUTS_ROOT / "requirements_summary.txt"
        output_path.write_text(summary + "\n", encoding="utf-8")
        print(f"\nSaved summary: {output_path}")


if __name__ == "__main__":
    main()
