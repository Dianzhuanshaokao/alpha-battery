#!/usr/bin/env python3
"""Export simulation CSV/JSON artifacts into TensorBoard scalar logs."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import time
from pathlib import Path

import pandas as pd
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.record_writer import RecordWriter


BOARD_ROOT = Path(__file__).resolve().parent
TB_ROOT = BOARD_ROOT / "outputs" / "tensorboard"


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


class ScalarEventWriter:
    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        event_path = log_dir / (
            f"events.out.tfevents.{int(time.time())}.{socket.gethostname()}.{os.getpid()}.0"
        )
        self._file = event_path.open("wb")
        self._writer = RecordWriter(self._file)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=float(value))])
        event = Event(wall_time=time.time(), step=int(step), summary=summary)
        self._writer.write(event.SerializeToString())

    def close(self) -> None:
        self._file.flush()
        self._file.close()


def _pick_step_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "checkpoint",
        "equivalent_cycle",
        "equivalent_cycle_actual",
        "cycle",
        "iteration",
        "step",
        "epoch",
    ]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def export_csv_scalars(csv_path: Path, log_dir: Path, namespace: str) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    step_column = _pick_step_column(df)
    numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != step_column]
    writer = ScalarEventWriter(log_dir)
    try:
        for row_index, row in df.iterrows():
            step = int(row[step_column]) if step_column is not None and pd.notna(row[step_column]) else row_index
            for column in numeric_columns:
                value = row[column]
                if pd.notna(value):
                    writer.add_scalar(f"{namespace}/{column}", float(value), step)
    finally:
        writer.close()


def export_json_scalars(json_path: Path, log_dir: Path, namespace: str) -> None:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    writer = ScalarEventWriter(log_dir)
    try:
        for key, value in payload.items():
            if _is_number(value):
                writer.add_scalar(f"{namespace}/{key}", float(value), 0)
    finally:
        writer.close()


def _safe_name(path: Path) -> str:
    return str(path).replace("\\", "_").replace("/", "_").replace(".", "_")


def export_tree_csvs(root: Path, namespace_prefix: str) -> list[Path]:
    written: list[Path] = []
    if not root.exists():
        return written
    for csv_path in root.rglob("*.csv"):
        relative = csv_path.relative_to(root)
        namespace = f"{namespace_prefix}/{relative.with_suffix('').as_posix()}"
        log_dir = TB_ROOT / _safe_name(Path(namespace))
        export_csv_scalars(csv_path, log_dir, namespace)
        written.append(log_dir)
    return written


def export_tree_json_scalars(root: Path, namespace_prefix: str) -> list[Path]:
    written: list[Path] = []
    if not root.exists():
        return written
    for json_path in root.rglob("*.json"):
        relative = json_path.relative_to(root)
        namespace = f"{namespace_prefix}/{relative.with_suffix('').as_posix()}"
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and any(_is_number(value) for value in payload.values()):
            log_dir = TB_ROOT / _safe_name(Path(namespace))
            export_json_scalars(json_path, log_dir, namespace)
            written.append(log_dir)
    return written


def export_closed_loop() -> list[Path]:
    results_root = BOARD_ROOT / "closed_loop" / "results"
    written = export_tree_csvs(results_root, "closed_loop")
    written.extend(export_tree_json_scalars(results_root, "closed_loop"))
    return written


def export_reproduce(case_name: str | None = None) -> list[Path]:
    output_root = BOARD_ROOT / "reproduce_li2024" / "outputs"
    if case_name is not None:
        case_dirs = [output_root / case_name]
    else:
        case_dirs = [path for path in output_root.iterdir() if path.is_dir()] if output_root.exists() else []

    written: list[Path] = []
    for case_dir in case_dirs:
        written.extend(export_tree_csvs(case_dir, f"reproduce/{case_dir.name}"))
        written.extend(export_tree_json_scalars(case_dir, f"reproduce/{case_dir.name}"))
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export simulation artifacts to TensorBoard scalar logs.")
    parser.add_argument("--target", choices=["closed_loop", "reproduce", "all"], default="all")
    parser.add_argument("--case", default=None, help="Specific reproduce_li2024 case name.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    written: list[Path] = []
    if args.target in {"closed_loop", "all"}:
        written.extend(export_closed_loop())
    if args.target in {"reproduce", "all"}:
        written.extend(export_reproduce(case_name=args.case))

    if written:
        print("Exported TensorBoard logs:")
        for log_dir in written:
            print(f"- {log_dir}")
        print(f"Suggested command: tensorboard --logdir {TB_ROOT}")
    else:
        print("No matching artifacts found to export.")


if __name__ == "__main__":
    main()
