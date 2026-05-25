"""Closed-loop RPT calibration with multi-metric loss and Bayesian optimization."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_ROOT = REPO_ROOT / "02_pybamm_model" / "closed_loop_model"
RESULTS_ROOT = PROJECT_ROOT / "results"
if str(PROJECT_ROOT / "src" / "degradation") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src" / "degradation"))

from coupled_degradation_model import (  # noqa: E402
    CoupledDegradationSimulator,
    load_cycle_protocol_records,
    load_model_config,
    load_parameter_overrides,
    load_protocol,
)
from rpt_features import (  # noqa: E402
    compute_curve_loss,
    compute_dqdv_curve,
    compute_soh_pct,
    validate_summary_columns,
    validate_trace_columns,
)


DEFAULT_SUMMARY_WEIGHTS = {
    "capacity_01c_ah": 1.0,
    "dcir_ohm": 1.0,
    "hppc_power_w": 0.5,
    "soh_pct": 0.8,
    "lli_pct": 0.7,
    "lam_ne_pct": 0.6,
    "lam_pe_pct": 0.6,
}

DEFAULT_CURVE_WEIGHTS = {
    "ica_rmse": 0.3,
    "dqdv_rmse": 0.3,
    "ica_peak_voltage_error": 0.2,
    "ica_peak_height_error": 0.1,
    "ica_area_rel_error": 0.1,
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_calibration_spec(path: Path) -> List[Dict[str, Any]]:
    spec = load_json(path)
    if not isinstance(spec, list) or not spec:
        raise ValueError("Calibration spec must be a non-empty JSON list.")
    return spec


def load_loss_weights(path: Path | None) -> Dict[str, Dict[str, float]]:
    if path is None or not path.exists():
        return {
            "summary": DEFAULT_SUMMARY_WEIGHTS.copy(),
            "curve": DEFAULT_CURVE_WEIGHTS.copy(),
        }
    data = load_json(path)
    return {
        "summary": {**DEFAULT_SUMMARY_WEIGHTS, **data.get("summary", {})},
        "curve": {**DEFAULT_CURVE_WEIGHTS, **data.get("curve", {})},
    }


def create_rpt_summary_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        columns=[
            "checkpoint",
            "cycle",
            "capacity_01c_ah",
            "dcir_ohm",
            "hppc_power_w",
            "lli_pct",
            "lam_ne_pct",
            "lam_pe_pct",
            "soh_pct",
        ]
    ).to_csv(path, index=False)


def create_rpt_trace_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        columns=["checkpoint", "trace_role", "time_s", "current_a", "voltage_v", "capacity_ah"]
    ).to_csv(path, index=False)


def load_real_rpt_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        create_rpt_summary_template(path)
        raise FileNotFoundError(f"未找到真实 RPT summary: {path}。已创建模板，请填入真实数据后重试。")
    df = pd.read_csv(path)
    validate_summary_columns(df)
    if df.empty:
        raise ValueError(f"真实 RPT summary 为空: {path}")
    if "soh_pct" not in df.columns:
        df = compute_soh_pct(df)
    return df.sort_values("checkpoint").reset_index(drop=True)


def load_real_rpt_trace(path: Path) -> pd.DataFrame:
    if not path.exists():
        create_rpt_trace_template(path)
        raise FileNotFoundError(f"未找到真实 RPT trace: {path}。已创建模板，请填入真实数据后重试。")
    df = pd.read_csv(path)
    validate_trace_columns(df)
    if df.empty:
        raise ValueError(f"真实 RPT trace 为空: {path}")
    df["source"] = "real"
    return df.sort_values(["checkpoint", "trace_role", "time_s"]).reset_index(drop=True)


def decode_value(spec: Dict[str, Any], raw_value: float) -> float:
    transform = spec.get("transform", "linear")
    if transform == "log10":
        return 10 ** raw_value
    if transform == "linear":
        return raw_value
    raise ValueError(f"Unsupported transform: {transform}")


def vector_to_updates(specs: List[Dict[str, Any]], x: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
    parameter_updates: Dict[str, float] = {}
    named_values: Dict[str, float] = {}
    for spec, raw in zip(specs, x):
        value = float(decode_value(spec, float(raw)))
        named_values[spec["name"]] = value
        kind = spec.get("kind", "parameter")
        if kind == "parameter":
            for target in spec.get("targets", []):
                parameter_updates[target] = value
        else:
            raise ValueError(f"Unsupported calibration kind: {kind}")
    return parameter_updates, named_values


def relative_error(sim: pd.Series, real: pd.Series) -> pd.Series:
    return (sim - real).abs() / real.abs().clip(lower=1e-9)


def compute_summary_loss(
    real_summary: pd.DataFrame, sim_summary: pd.DataFrame, weights: Dict[str, float]
) -> Tuple[float, pd.DataFrame]:
    real = compute_soh_pct(real_summary).copy()
    sim = compute_soh_pct(sim_summary).copy()
    merged = real.merge(sim, on="checkpoint", suffixes=("_real", "_sim"), how="inner")
    if merged.empty:
        raise ValueError("No overlapping checkpoints between real and simulated RPT summaries.")

    score = 0.0
    weight_sum = 0.0
    for metric, weight in weights.items():
        real_col = f"{metric}_real"
        sim_col = f"{metric}_sim"
        if real_col in merged.columns and sim_col in merged.columns:
            merged[f"{metric}_rel_error"] = relative_error(merged[sim_col], merged[real_col])
            score += float(merged[f"{metric}_rel_error"].mean()) * weight
            weight_sum += weight
    if weight_sum == 0:
        raise ValueError("No overlapping summary metrics found for loss computation.")
    return score / weight_sum, merged


def compute_curve_loss_table(
    real_trace: pd.DataFrame, sim_trace: pd.DataFrame, weights: Dict[str, float]
) -> Tuple[float, pd.DataFrame]:
    rows: List[Dict[str, float]] = []
    score = 0.0
    weight_sum = 0.0

    keys = (
        real_trace[["checkpoint", "trace_role"]]
        .drop_duplicates()
        .merge(sim_trace[["checkpoint", "trace_role"]].drop_duplicates(), on=["checkpoint", "trace_role"])
    )
    for _, key in keys.iterrows():
        checkpoint = int(key["checkpoint"])
        trace_role = str(key["trace_role"])
        real_curve = compute_dqdv_curve(real_trace[(real_trace["checkpoint"] == checkpoint) & (real_trace["trace_role"] == trace_role)])
        sim_curve = compute_dqdv_curve(sim_trace[(sim_trace["checkpoint"] == checkpoint) & (sim_trace["trace_role"] == trace_role)])
        curve_errors = compute_curve_loss(real_curve, sim_curve)
        row = {"checkpoint": checkpoint, "trace_role": trace_role, **curve_errors}
        rows.append(row)
        for metric, weight in weights.items():
            if metric in curve_errors:
                score += curve_errors[metric] * weight
                weight_sum += weight
    curve_df = pd.DataFrame(rows)
    if weight_sum == 0:
        return 0.0, curve_df
    return score / weight_sum, curve_df


def expected_improvement(
    x_candidates: np.ndarray, model: GaussianProcessRegressor, y_best: float, xi: float = 0.01
) -> np.ndarray:
    mu, sigma = model.predict(x_candidates, return_std=True)
    sigma = np.maximum(sigma, 1e-12)
    improvement = y_best - mu - xi
    z = improvement / sigma
    normal_pdf = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    normal_cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))
    return improvement * normal_cdf + sigma * normal_pdf


class RPTCalibrator:
    def __init__(
        self,
        simulator: CoupledDegradationSimulator,
        cycle_protocol_records: pd.DataFrame | None,
        base_parameter_overrides: Dict[str, Any],
        real_summary: pd.DataFrame,
        real_trace: pd.DataFrame,
        calibration_spec: List[Dict[str, Any]],
        loss_weights: Dict[str, Dict[str, float]],
        rel_error_target: float,
        model_dump_dir: Path,
    ):
        self.simulator = simulator
        self.cycle_protocol_records = cycle_protocol_records
        self.base_parameter_overrides = base_parameter_overrides
        self.real_summary = real_summary
        self.real_trace = real_trace
        self.calibration_spec = calibration_spec
        self.loss_weights = loss_weights
        self.rel_error_target = rel_error_target
        self.model_dump_dir = model_dump_dir
        self.best_error = float("inf")
        self.best_named_values: Dict[str, float] | None = None
        self.best_parameter_updates: Dict[str, float] | None = None
        self.best_summary_fit: pd.DataFrame | None = None
        self.best_curve_fit: pd.DataFrame | None = None
        self.best_sim_summary: pd.DataFrame | None = None
        self.best_sim_trace: pd.DataFrame | None = None
        self.best_sim_feature: pd.DataFrame | None = None
        self.history_rows: List[Dict[str, float]] = []
        self.model_dump_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, x: np.ndarray, iteration: int) -> float:
        parameter_updates, named_values = vector_to_updates(self.calibration_spec, x)
        merged_parameter_updates = {**self.base_parameter_overrides, **parameter_updates}

        sim_summary, sim_trace, sim_feature = self.simulator.run(
            parameter_overrides=merged_parameter_updates,
            cycle_protocol_records=self.cycle_protocol_records,
            output_csv=None,
            output_trace_csv=None,
            output_feature_csv=None,
        )
        summary_loss, summary_fit = compute_summary_loss(
            real_summary=self.real_summary,
            sim_summary=sim_summary,
            weights=self.loss_weights["summary"],
        )
        curve_loss, curve_fit = compute_curve_loss_table(
            real_trace=self.real_trace,
            sim_trace=sim_trace,
            weights=self.loss_weights["curve"],
        )
        objective_val = 0.7 * summary_loss + 0.3 * curve_loss

        row = {
            "iteration": iteration,
            "objective": objective_val,
            "summary_loss": summary_loss,
            "curve_loss": curve_loss,
            **named_values,
        }
        self.history_rows.append(row)

        if objective_val < self.best_error:
            self.best_error = objective_val
            self.best_named_values = named_values
            self.best_parameter_updates = merged_parameter_updates
            self.best_summary_fit = summary_fit
            self.best_curve_fit = curve_fit
            self.best_sim_summary = sim_summary
            self.best_sim_trace = sim_trace
            self.best_sim_feature = sim_feature
            self._dump_best()
        return objective_val

    def _dump_best(self) -> None:
        payload = {
            "best_error": self.best_error,
            "best_named_values": self.best_named_values,
            "best_parameter_updates": self.best_parameter_updates,
        }
        with (self.model_dump_dir / "best_calibration_snapshot.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        if self.best_summary_fit is not None:
            self.best_summary_fit.to_csv(self.model_dump_dir / "best_summary_fit.csv", index=False)
        if self.best_curve_fit is not None:
            self.best_curve_fit.to_csv(self.model_dump_dir / "best_curve_fit.csv", index=False)
        if self.best_sim_summary is not None:
            self.best_sim_summary.to_csv(self.model_dump_dir / "best_simulated_rpt_summary.csv", index=False)
        if self.best_sim_trace is not None:
            self.best_sim_trace.to_csv(self.model_dump_dir / "best_simulated_rpt_traces.csv", index=False)
        if self.best_sim_feature is not None:
            self.best_sim_feature.to_csv(self.model_dump_dir / "best_simulated_rpt_features.csv", index=False)

    def extract_flat_features(self, summary_df: pd.DataFrame, trace_df: pd.DataFrame) -> np.ndarray:
        features = []
        checkpoints = sorted(summary_df["checkpoint"].unique())
        for cp in checkpoints:
            sum_row = summary_df[summary_df["checkpoint"] == cp].iloc[0]
            features.extend([sum_row.get("soh_pct", 100.0), sum_row.get("dcir_ohm", 0.015)])
            cp_trace = trace_df[(trace_df["checkpoint"] == cp) & (trace_df["trace_role"] == "discharge")]
            if not cp_trace.empty:
                try:
                    curve = compute_dqdv_curve(cp_trace)
                    features.extend([curve.peak_voltage_v, curve.peak_height, curve.area_abs])
                except Exception:
                    features.extend([3.7, 0.0, 0.0])
            else:
                features.extend([3.7, 0.0, 0.0])
        return np.array(features, dtype=float)

    def calibrate_sbi(self, n_simulations: int, random_seed: int) -> Tuple[Dict[str, float], float]:
        import torch
        from sbi.inference import SNPE_C
        from sbi.utils import BoxUniform

        bounds = self.bounds_array()
        prior = BoxUniform(
            low=torch.as_tensor(bounds[:, 0], dtype=torch.float32),
            high=torch.as_tensor(bounds[:, 1], dtype=torch.float32)
        )

        x_obs = self.extract_flat_features(self.real_summary, self.real_trace)
        x_obs_tensor = torch.as_tensor(x_obs, dtype=torch.float32)

        theta_samples = prior.sample((n_simulations,))
        x_sims = []
        valid_thetas = []

        for i, theta in enumerate(theta_samples):
            theta_np = theta.numpy()
            try:
                parameter_updates, named_values = vector_to_updates(self.calibration_spec, theta_np)
                merged_parameter_updates = {**self.base_parameter_overrides, **parameter_updates}
                sim_summary, sim_trace, _ = self.simulator.run(
                    parameter_overrides=merged_parameter_updates,
                    cycle_protocol_records=self.cycle_protocol_records,
                )
                sim_feat = self.extract_flat_features(sim_summary, sim_trace)
                x_sims.append(sim_feat)
                valid_thetas.append(theta_np)
            except Exception as e:
                print(f"[SBI Sim Warning] Simulation {i} failed: {e}")
                continue

        if not x_sims:
            raise RuntimeError("All SBI simulations failed.")

        theta_tensor = torch.as_tensor(np.array(valid_thetas), dtype=torch.float32)
        x_tensor = torch.as_tensor(np.array(x_sims), dtype=torch.float32)

        inference = SNPE_C(prior=prior, density_estimator="maf")
        inference.append_simulations(theta_tensor, x_tensor)
        density_estimator = inference.train(show_progress_bars=False)
        posterior = inference.build_posterior(density_estimator)

        samples = posterior.sample((1000,), x=x_obs_tensor, show_progress_bars=False)
        best_theta = samples.mean(dim=0).numpy()

        best_error = self.evaluate(best_theta, iteration=9999)
        return self.best_named_values or {}, best_error

    def bounds_array(self) -> np.ndarray:
        return np.asarray([spec["bounds"] for spec in self.calibration_spec], dtype=float)

    def calibrate_bayesian(self, n_init: int, n_iter: int, random_seed: int) -> Tuple[Dict[str, float], float]:
        rng = np.random.default_rng(random_seed)
        bounds = self.bounds_array()
        dim = bounds.shape[0]
        xs: List[np.ndarray] = []
        ys: List[float] = []

        for i in range(n_init):
            x = rng.uniform(bounds[:, 0], bounds[:, 1], size=dim)
            y = self.evaluate(x, iteration=i + 1)
            xs.append(x)
            ys.append(y)

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-6)
        for i in range(n_iter):
            model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=random_seed)
            x_train = np.vstack(xs)
            y_train = np.asarray(ys)
            model.fit(x_train, y_train)

            candidates = rng.uniform(bounds[:, 0], bounds[:, 1], size=(512, dim))
            ei = expected_improvement(candidates, model=model, y_best=float(y_train.min()))
            x_next = candidates[int(np.argmax(ei))]
            y_next = self.evaluate(x_next, iteration=n_init + i + 1)
            xs.append(x_next)
            ys.append(y_next)
            if self.best_error < self.rel_error_target:
                break
        return self.best_named_values or {}, self.best_error

    def calibrate_de(self, maxiter: int, popsize: int, random_seed: int) -> Tuple[Dict[str, float], float]:
        bounds = [tuple(spec["bounds"]) for spec in self.calibration_spec]
        counter = {"i": 0}

        def objective(x: np.ndarray) -> float:
            counter["i"] += 1
            return self.evaluate(x, iteration=counter["i"])

        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=maxiter,
            popsize=popsize,
            polish=True,
            seed=random_seed,
            updating="deferred",
        )
        if self.best_named_values is None:
            _, self.best_named_values = vector_to_updates(self.calibration_spec, result.x)
            self.best_error = float(result.fun)
        return self.best_named_values or {}, self.best_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate coupled degradation with real-vs-simulated RPT losses.")
    parser.add_argument(
        "--protocol-json",
        type=Path,
        default=MODEL_ROOT / "configs" / "degradation_protocol.json",
    )
    parser.add_argument("--model-config-json", type=Path, default=None)
    parser.add_argument("--cycle-protocol-csv", type=Path, default=None)
    parser.add_argument(
        "--parameter-overrides-json",
        type=Path,
        default=MODEL_ROOT / "configs" / "model_parameter_overrides.json",
    )

    parser.add_argument("--real-rpt-summary-csv", type=Path, default=PROJECT_ROOT / "data" / "real_rpt_summary.csv")
    parser.add_argument("--real-rpt-trace-csv", type=Path, default=PROJECT_ROOT / "data" / "real_rpt_low_rate_trace.csv")
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
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "configs" / "calibrated_params.json")
    parser.add_argument(
        "--output-log-csv",
        type=Path,
        default=RESULTS_ROOT / "calibration" / "calibration_history.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.optimizer == "sbi":
        raise NotImplementedError("当前环境未安装 sbi；请先使用 --optimizer bayesian 或安装 sbi 后再接入。")

    protocol = load_protocol(args.protocol_json)
    model_config = load_model_config(args.model_config_json, ambient_temperature_c=args.temperature_c)
    cycle_protocol_records = load_cycle_protocol_records(args.cycle_protocol_csv)
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

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "optimizer": args.optimizer,
                "best_named_values": best_named_values,
                "best_parameter_updates": calibrator.best_parameter_updates,
                "best_error": best_error,
            },
            f,
            indent=2,
        )

    history_df = pd.DataFrame(calibrator.history_rows)
    args.output_log_csv.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(args.output_log_csv, index=False)
    print(f"Best objective error: {best_error:.4%}")
    print(f"Saved calibrated params: {args.output_json.resolve()}")
    print(f"Saved calibration history: {args.output_log_csv.resolve()}")


if __name__ == "__main__":
    main()
