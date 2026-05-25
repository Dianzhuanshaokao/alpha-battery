"""Shared feature extraction for real and simulated RPT traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


RPT_SUMMARY_REQUIRED = {
    "checkpoint",
    "cycle",
    "capacity_01c_ah",
    "dcir_ohm",
}

RPT_TRACE_REQUIRED = {
    "checkpoint",
    "trace_role",
    "time_s",
    "current_a",
    "voltage_v",
    "capacity_ah",
}


@dataclass
class CurveFeatures:
    voltage_grid_v: np.ndarray
    dq_dv: np.ndarray
    ica: np.ndarray
    peak_voltage_v: float
    peak_height: float
    area_abs: float


def validate_summary_columns(df: pd.DataFrame, required: Iterable[str] = RPT_SUMMARY_REQUIRED) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"RPT summary missing columns: {sorted(missing)}")


def validate_trace_columns(df: pd.DataFrame, required: Iterable[str] = RPT_TRACE_REQUIRED) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"RPT trace missing columns: {sorted(missing)}")


def normalize_trace(trace_df: pd.DataFrame) -> pd.DataFrame:
    trace = trace_df.sort_values("time_s").copy()
    trace["time_s"] = trace["time_s"] - float(trace["time_s"].iloc[0])
    return trace


def compute_soh_pct(summary_df: pd.DataFrame) -> pd.DataFrame:
    validate_summary_columns(summary_df)
    out = summary_df.copy()
    base_capacity = float(out["capacity_01c_ah"].iloc[0])
    out["soh_pct"] = 100.0 * out["capacity_01c_ah"] / max(base_capacity, 1e-9)
    return out


def compute_dqdv_curve(trace_df: pd.DataFrame, voltage_points: int = 256) -> CurveFeatures:
    validate_trace_columns(trace_df)
    trace = normalize_trace(trace_df)
    # Capacity must be monotonic for incremental capacity analysis.
    q = trace["capacity_ah"].to_numpy(dtype=float)
    v = trace["voltage_v"].to_numpy(dtype=float)

    # Remove flat and duplicated voltage segments to stabilize differentiation.
    order = np.argsort(v)
    v_sorted = v[order]
    q_sorted = q[order]
    keep = np.concatenate([[True], np.diff(v_sorted) > 1e-6])
    v_unique = v_sorted[keep]
    q_unique = q_sorted[keep]
    if len(v_unique) < 2:
        raise ValueError("Trace does not contain enough unique voltage points for dQ/dV analysis.")

    effective_points = max(16, min(voltage_points, len(v_unique) * 16))
    voltage_grid = np.linspace(float(v_unique.min()), float(v_unique.max()), effective_points)
    q_interp = np.interp(voltage_grid, v_unique, q_unique)
    dq_dv = np.gradient(q_interp, voltage_grid)
    ica = dq_dv.copy()

    peak_idx = int(np.nanargmax(np.abs(ica)))
    peak_voltage = float(voltage_grid[peak_idx])
    peak_height = float(ica[peak_idx])
    area_abs = float(np.trapz(np.abs(ica), voltage_grid))
    return CurveFeatures(
        voltage_grid_v=voltage_grid,
        dq_dv=dq_dv,
        ica=ica,
        peak_voltage_v=peak_voltage,
        peak_height=peak_height,
        area_abs=area_abs,
    )


def features_to_frame(checkpoint: int, source: str, trace_role: str, features: CurveFeatures) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "checkpoint": checkpoint,
            "source": source,
            "trace_role": trace_role,
            "voltage_v": features.voltage_grid_v,
            "dq_dv": features.dq_dv,
            "ica": features.ica,
        }
    )


def compute_curve_loss(real_curve: CurveFeatures, sim_curve: CurveFeatures) -> Dict[str, float]:
    sim_ica_interp = np.interp(real_curve.voltage_grid_v, sim_curve.voltage_grid_v, sim_curve.ica)
    sim_dqdv_interp = np.interp(real_curve.voltage_grid_v, sim_curve.voltage_grid_v, sim_curve.dq_dv)
    ica_rmse = float(np.sqrt(np.mean((real_curve.ica - sim_ica_interp) ** 2)))
    dqdv_rmse = float(np.sqrt(np.mean((real_curve.dq_dv - sim_dqdv_interp) ** 2)))
    peak_voltage_error = abs(real_curve.peak_voltage_v - sim_curve.peak_voltage_v)
    peak_height_error = abs(real_curve.peak_height - sim_curve.peak_height)
    area_error = abs(real_curve.area_abs - sim_curve.area_abs) / max(real_curve.area_abs, 1e-9)
    return {
        "ica_rmse": ica_rmse,
        "dqdv_rmse": dqdv_rmse,
        "ica_peak_voltage_error": peak_voltage_error,
        "ica_peak_height_error": peak_height_error,
        "ica_area_rel_error": area_error,
    }


def estimate_rul_from_summary(summary_df: pd.DataFrame, soh_floor_pct: float = 80.0) -> Dict[str, float]:
    summary = compute_soh_pct(summary_df).sort_values("cycle").copy()
    if len(summary) < 2:
        return {"rul_cycles": None, "soh_floor_pct": soh_floor_pct, "fade_rate_pct_per_cycle": None}
    x = summary["cycle"].to_numpy(dtype=float)
    y = summary["soh_pct"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    latest_cycle = float(x[-1])
    if slope >= 0:
        rul_cycles = float("inf")
    else:
        cycle_at_floor = (soh_floor_pct - intercept) / slope
        rul_cycles = max(0.0, float(cycle_at_floor - latest_cycle))
    return {
        "rul_cycles": rul_cycles,
        "soh_floor_pct": soh_floor_pct,
        "fade_rate_pct_per_cycle": float(slope),
    }


def summarize_cycle_timeseries(cycle_df: pd.DataFrame) -> Dict[str, float]:
    required = {"segment_id", "cycle_index", "current_a", "voltage_v", "capacity_ah"}
    missing = required - set(cycle_df.columns)
    if missing:
        raise ValueError(f"Cycle timeseries missing columns: {sorted(missing)}")
    abs_current = cycle_df["current_a"].abs()
    return {
        "n_rows": int(len(cycle_df)),
        "n_cycles": int(cycle_df["cycle_index"].nunique()),
        "mean_abs_current_a": float(abs_current.mean()),
        "max_voltage_v": float(cycle_df["voltage_v"].max()),
        "min_voltage_v": float(cycle_df["voltage_v"].min()),
        "max_capacity_ah": float(cycle_df["capacity_ah"].max()),
    }
