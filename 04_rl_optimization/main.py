#!/usr/bin/env python3
"""Board entrypoint for RL training and TensorBoard usage."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BOARD_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = BOARD_ROOT / "outputs"
TRAINING_ROOT = OUTPUTS_ROOT / "training"
TRAIN_SCRIPT = BOARD_ROOT / "SmartCharging" / "Scripts" / "RLtrain" / "train_ppo.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RL-board tasks or print TensorBoard commands.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("overview", help="Print RL-board overview.")

    train = subparsers.add_parser("train", help="Run SmartCharging PPO training.")
    train.add_argument("extra_args", nargs=argparse.REMAINDER)

    subparsers.add_parser("tensorboard", help="Print the recommended TensorBoard command.")
    return parser


def print_overview() -> None:
    print("04_rl_optimization")
    print("")
    print("Entrypoints:")
    print(f"- PPO training: {TRAIN_SCRIPT}")
    print("")
    print("Board outputs:")
    print(f"- Training root: {TRAINING_ROOT}")
    print(f"- TensorBoard command: tensorboard --logdir {TRAINING_ROOT}")
    print("")
    print("Cluster notes:")
    print("- Remote training is expected on 121.48.164.50")
    print("- Load GPU with `module load gpu` before GPU jobs")
    print("- GPU nodes: c1, c2, c3")


def run_training(extra_args: list[str]) -> None:
    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(TRAIN_SCRIPT), *extra_args]
    subprocess.run(cmd, check=True)


def print_tensorboard_command() -> None:
    print(f"tensorboard --logdir {TRAINING_ROOT}")


def main() -> None:
    args = build_parser().parse_args()
    command = args.command or "overview"
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)

    if command == "overview":
        print_overview()
        return
    if command == "train":
        run_training(args.extra_args)
        return
    if command == "tensorboard":
        print_tensorboard_command()
        return
    raise ValueError(f"Unsupported command: {command}")


if __name__ == "__main__":
    main()
