"""End-to-end closed-loop workflow for real data ingestion, calibration and state-vector export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_ROOT = REPO_ROOT / "02_pybamm_model" / "closed_loop_model"
RESULTS_ROOT = PROJECT_ROOT / "results"
if str(PROJECT_ROOT / "src" / "calibration") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src" / "calibration"))
if str(PROJECT_ROOT / "src" / "degradation") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src" / "degradation"))

from rpt_calibration import (  # noqa: E402
    RPTCalibrator,
    load_calibration_spec,
    load_loss_weights,
    load_real_rpt_summary,
    load_real_rpt_trace,
)
from coupled_degradation_model import (  # noqa: E402
    CoupledDegradationSimulator,
    load_cycle_protocol_records,
    load_model_config,
    load_parameter_overrides,
    load_protocol,
)
from rpt_features import estimate_rul_from_summary, summarize_cycle_timeseries  # noqa: E402


def load_cycle_timeseries(path: Path) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=["segment_id", "cycle_index", "time_s", "current_a", "voltage_v", "capacity_ah", "temperature_c"]
        ).to_csv(path, index=False)
        raise FileNotFoundError(f"未找到循环时序数据: {path}。已创建模板，请填入真实数据后重试。")
    df = pd.read_csv(path)
    required = {"segment_id", "cycle_index", "time_s", "current_a", "voltage_v", "capacity_ah"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"循环时序数据缺少列: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"循环时序数据为空: {path}")
    return df.sort_values(["segment_id", "cycle_index", "time_s"]).reset_index(drop=True)


def latest_protocol_snapshot(protocol_records: pd.DataFrame | None) -> Dict[str, Any]:
    if protocol_records is None or protocol_records.empty:
        return {}
    latest_checkpoint = int(protocol_records["checkpoint"].max())
    latest = protocol_records[protocol_records["checkpoint"] == latest_checkpoint].sort_values("step_index")
    return {
        "checkpoint": latest_checkpoint,
        "repeat_count": int(latest["repeat_count"].iloc[0]),
        "instructions": latest["instruction"].tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run closed-loop battery degradation workflow with real and simulated data.")
    parser.add_argument(
        "--protocol-json",
        type=Path,
        default=MODEL_ROOT / "configs" / "degradation_protocol.json",
    )
    parser.add_argument("--model-config-json", type=Path, default=None)
    parser.add_argument("--cycle-protocol-csv", type=Path, default=PROJECT_ROOT / "data" / "cycle_protocols.csv")
    parser.add_argument(
        "--real-cycle-timeseries-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "real_cycle_timeseries.csv",
    )
    parser.add_argument(
        "--parameter-overrides-json",
        type=Path,
        default=MODEL_ROOT / "configs" / "model_parameter_overrides.json",
    )

    parser.add_argument("--real-rpt-summary-csv", type=Path, default=PROJECT_ROOT / "data" / "real_rpt_summary.csv")
    parser.add_argument(
        "--real-rpt-trace-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "real_rpt_low_rate_trace.csv",
    )
    parser.add_argument("--calibration-spec-json", type=Path, default=PROJECT_ROOT / "configs" / "calibration_map.json")
    parser.add_argument("--loss-weights-json", type=Path, default=PROJECT_ROOT / "configs" / "loss_weights.json")
    parser.add_argument("--temperature-c", type=float, default=None)
    parser.add_argument("--optimizer", choices=["bayesian", "de", "sbi"], default="bayesian")
    parser.add_argument("--n-init", type=int, default=6)
    parser.add_argument("--n-iter", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=4)
    parser.add_argument("--popsize", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--target-rel-error", type=float, default=0.03)
    parser.add_argument("--state-vector-json", type=Path, default=RESULTS_ROOT / "workflow" / "state_vector.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol_json)
    model_config = load_model_config(args.model_config_json, ambient_temperature_c=args.temperature_c)
    cycle_protocol_records = load_cycle_protocol_records(args.cycle_protocol_csv)
    cycle_timeseries = load_cycle_timeseries(args.real_cycle_timeseries_csv)
    base_parameter_overrides = load_parameter_overrides(args.parameter_overrides_json)
    real_summary = load_real_rpt_summary(args.real_rpt_summary_csv)
    real_trace = load_real_rpt_trace(args.real_rpt_trace_csv)
    calibration_spec = load_calibration_spec(args.calibration_spec_json)
    loss_weights = load_loss_weights(args.loss_weights_json)

    simulator = CoupledDegradationSimulator(model_config=model_config, protocol=protocol)
    calibrator = RPTCalibrator(
        simulator=simulator,
        cycle_protocol_records=cycle_protocol_records,
        base_parameter_overrides=base_parameter_overrides,
        real_summary=real_summary,
        real_trace=real_trace,
        calibration_spec=calibration_spec,
        loss_weights=loss_weights,
        rel_error_target=args.target_rel_error,
        model_dump_dir=RESULTS_ROOT / "calibration",
    )

    if args.optimizer == "bayesian":
        best_named_values, best_error = calibrator.calibrate_bayesian(
            n_init=args.n_init, n_iter=args.n_iter, random_seed=args.random_seed
        )
    elif args.optimizer == "sbi":
        best_named_values, best_error = calibrator.calibrate_sbi(
            n_simulations=args.n_init, random_seed=args.random_seed
        )
    else:
        best_named_values, best_error = calibrator.calibrate_de(
            maxiter=args.maxiter, popsize=args.popsize, random_seed=args.random_seed
        )

    rul_info = estimate_rul_from_summary(real_summary)
    cycle_stats = summarize_cycle_timeseries(cycle_timeseries)
    latest_real_rpt = real_summary.sort_values("checkpoint").iloc[-1].to_dict()
    latest_sim_rpt = (
        calibrator.best_sim_summary.sort_values("checkpoint").iloc[-1].to_dict()
        if calibrator.best_sim_summary is not None
        else {}
    )

    state_vector = {
        "rul": rul_info,
        "cycle_data_stats": cycle_stats,
        "latest_protocol": latest_protocol_snapshot(cycle_protocol_records),
        "best_error": best_error,
        "optimizer": args.optimizer,
        "updated_hyperparameters": best_named_values,
        "updated_parameter_map": calibrator.best_parameter_updates,
        "real_rpt_latest": latest_real_rpt,
        "sim_rpt_latest": latest_sim_rpt,
        "artifacts": {
            "best_summary_fit_csv": str((RESULTS_ROOT / "calibration" / "best_summary_fit.csv").resolve()),
            "best_curve_fit_csv": str((RESULTS_ROOT / "calibration" / "best_curve_fit.csv").resolve()),
            "best_simulated_rpt_summary_csv": str(
                (RESULTS_ROOT / "calibration" / "best_simulated_rpt_summary.csv").resolve()
            ),
            "best_simulated_rpt_traces_csv": str(
                (RESULTS_ROOT / "calibration" / "best_simulated_rpt_traces.csv").resolve()
            ),
            "best_simulated_rpt_features_csv": str(
                (RESULTS_ROOT / "calibration" / "best_simulated_rpt_features.csv").resolve()
            ),
        },
    }

    args.state_vector_json.parent.mkdir(parents=True, exist_ok=True)
    with args.state_vector_json.open("w", encoding="utf-8") as f:
        json.dump(state_vector, f, indent=2)

    history_df = pd.DataFrame(calibrator.history_rows)
    history_path = RESULTS_ROOT / "workflow" / "calibration_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(history_path, index=False)

    print(f"Saved state vector: {args.state_vector_json.resolve()}")
    print(f"Saved workflow history: {history_path.resolve()}")
    print(json.dumps(state_vector, indent=2))


if __name__ == "__main__":
    main()
