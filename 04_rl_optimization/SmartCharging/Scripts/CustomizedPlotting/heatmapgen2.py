#!/usr/bin/env python3
"""Parse env_*.log files and visualize metrics for charge rates and times.

This script reads log files named env_*.log from a specified directory,
extracts charge rates and computes charge times for each cycle,
and generates PCA and trend plots similar to heatmapgen.py.
"""

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

FONT_SIZE_OFFSET = 10


def _increase_font_size(size_value, offset: int) -> float:
    if isinstance(size_value, str):
        size_value = FontProperties(size=size_value).get_size_in_points()
    return float(size_value) + offset

plt.rcParams.update(
    {
        "font.size": _increase_font_size(plt.rcParams["font.size"], FONT_SIZE_OFFSET),
        "axes.titlesize": _increase_font_size(plt.rcParams["axes.titlesize"], FONT_SIZE_OFFSET),
        "axes.labelsize": _increase_font_size(plt.rcParams["axes.labelsize"], FONT_SIZE_OFFSET),
        "xtick.labelsize": _increase_font_size(plt.rcParams["xtick.labelsize"], FONT_SIZE_OFFSET),
        "ytick.labelsize": _increase_font_size(plt.rcParams["ytick.labelsize"], FONT_SIZE_OFFSET),
        "legend.fontsize": _increase_font_size(plt.rcParams["legend.fontsize"], FONT_SIZE_OFFSET),
        "figure.titlesize": _increase_font_size(plt.rcParams["figure.titlesize"], FONT_SIZE_OFFSET),
    }
)

# Configuration
TEMPERATURE = 10
LOG_DIR = Path(f"overhauled_train_iter2/T{TEMPERATURE}")
OUTPUT_DIR = Path("heatmapgen_output_2")
SMOOTHING_WINDOW = 1
PCA_SMOOTHING_WINDOW = 256
PLOT_SUBSAMPLE = 16  # 绘图散点采样率，用于减小散点图中绘制点的数量
#PCA_FEATURES = ["chargerate1_c", "charge_time_s1", "chargerate2_c", "charge_time_s2"]
PCA_FEATURES = ["charge_time_s1", "charge_time_s2", "charge_time_s3", "total_time_s"]

def parse_log_files(log_dir: Path) -> Dict[str, np.ndarray]:
    """Parse all env_*.log files in the directory."""
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    # Aggregated metrics: list of values for each cycle across all environments
    # We use step/cycle as the index. In the logs, 'Cycle: X' is used.
    data = defaultdict(list)
    
    # Cycle pattern: Cycle: 1, SOH(measured): 0.9890
    cycle_pattern = re.compile(r"Cycle:\s+(?P<cycle>\d+),\s+SOH\(measured\)")
    # Rates pattern: Charge Rates: Stage1=1.75C, Stage2=2.25C | Terminal Current=0.100C
    rates_pattern = re.compile(r"Charge Rates: Stage1=(?P<r1>[\d\.]+)C,\s+Stage2=(?P<r2>[\d\.]+)C")
    # Stage Times pattern: Stage Times: S1=411.43s, S2=960.00s, S3=1242.62s
    stage_times_pattern = re.compile(r"Stage Times: S1=(?P<t1>[\d\.]+)s,\s+S2=(?P<t2>[\d\.]+)s,\s+S3=(?P<t3>[\d\.]+)s")

    log_files = sorted(log_dir.glob("env_*.log"))
    if not log_files:
        print(f"No log files found in {log_dir}")
        return {}

    for log_file in log_files:
        with open(log_file, "r") as f:
            current_rates = None
            current_cycle = None
            for line in f:
                # Try to find Cycle info line
                cycle_match = cycle_pattern.search(line)
                if cycle_match:
                    current_cycle = int(cycle_match.group("cycle"))
                    continue
                    
                # Try to find Rates line
                rates_match = rates_pattern.search(line)
                if rates_match:
                    current_rates = (float(rates_match.group("r1")), float(rates_match.group("r2")))
                    continue
                
                # Try to find Stage Times line
                stage_match = stage_times_pattern.search(line)
                if stage_match and current_rates and current_cycle is not None:
                    t1 = float(stage_match.group("t1"))
                    t2 = float(stage_match.group("t2"))
                    t3 = float(stage_match.group("t3"))
                    total_time = t1 + t2 + t3
                    
                    r1, r2 = current_rates
                    
                    data["cycle"].append(current_cycle)
                    data["chargerate1_c"].append(r1)
                    data["chargerate2_c"].append(r2)
                    data["charge_time_s1"].append(t1)
                    data["charge_time_s2"].append(t2)
                    data["charge_time_s3"].append(t3)
                    data["total_time_s"].append(total_time)
                    
                    # Reset to ensure we only pair it with the next cycle
                    current_rates = None
                    current_cycle = None

    # Do not sort parsed lists by cycle - maintain original distribution order
    return {k: np.array(v) for k, v in data.items()}

def compute_pca_projection(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if values.ndim != 2:
        raise ValueError("PCA input must be a 2D array")
    features = values.astype(np.float64, copy=False)
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    normalized = (features - mean) / std
    u, singular_values, vt = np.linalg.svd(normalized, full_matrices=False)
    explained_variance = singular_values**2
    projection = normalized @ vt[:2].T
    explained_ratio = explained_variance / explained_variance.sum()
    return projection.astype(np.float32), explained_ratio[:2].astype(np.float32)

def smooth_series(series: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(series) < 3:
        return series
    window = min(window, len(series) if len(series) % 2 == 1 else len(series) - 1)
    if window < 3:
        return series
    if window % 2 == 0:
        window -= 1
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(series.astype(np.float64), (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")

def plot_metric_trends(
    steps: np.ndarray,
    metrics: Dict[str, np.ndarray],
    output_path: Path,
    smoothing_window: int,
) -> None:
    plt.figure(figsize=(14, 8))
    for key in PCA_FEATURES + ["cycle"]:
        if key not in metrics:
            continue
        values = metrics[key]
        if smoothing_window > 1:
            values = smooth_series(values, smoothing_window)
        plt.plot(steps[: len(values)], values, label=key, linewidth=1.8, alpha=0.8)

    plt.title("Metrics Tracker")
    plt.xlabel("Sequence Index")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(alpha=0.35, linestyle="--")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()

def plot_pca_cycle_count(
    projection: np.ndarray,
    cycle_values: np.ndarray,
    explained_ratio: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.tick_params(axis="both", which="both", direction="in")
    scatter = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=cycle_values,
        cmap="viridis",
        s=28,
        alpha=0.8,
        linewidths=0,
        rasterized=True,
    )
    feature_names = "/".join(PCA_FEATURES)
    #ax.set_title(f"PCA 2D of {feature_names} colored by cycle count")
    ax.set_xlabel(f"PC1") # ({explained_ratio[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2") # ({explained_ratio[1] * 100:.1f}%)")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("cycle")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

def main():
    print(f"Using log directory: {LOG_DIR}")
    
    try:
        metrics = parse_log_files(LOG_DIR)
        if not metrics or len(metrics.get("cycle", [])) == 0:
            print("No data collected.")
            return

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Use sequential indices as the x-axis to show the wave-like progression
        steps = np.arange(len(metrics["cycle"]))
        
        feature_values = np.stack(
            [smooth_series(metrics[feature], PCA_SMOOTHING_WINDOW) for feature in PCA_FEATURES],
            axis=1,
        )
        cycle_values = metrics["cycle"]
        
        projection, explained = compute_pca_projection(feature_values)

        plot_metric_trends(steps, metrics, OUTPUT_DIR / f"metric_trends_T{TEMPERATURE}_logs.png", SMOOTHING_WINDOW)
        plot_pca_cycle_count(projection[::PLOT_SUBSAMPLE], cycle_values[::PLOT_SUBSAMPLE], explained, OUTPUT_DIR / f"pca_cycle_count_T{TEMPERATURE}_logs.png")
        
        print(f"Aggregated {len(steps)} cycles from logs")
        print(f"Plots written to: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
