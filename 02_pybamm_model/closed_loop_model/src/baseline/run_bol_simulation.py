"""Run BOL baseline simulation and export voltage-capacity and temperature curves."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

EXTERNAL_ROOT = Path(__file__).resolve().parents[4] / "external"
if str(EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ROOT))

import pybamm

from dfn_thermal_model import BOLParameters, BaselineDFNThermalModel, ThermalParameters, pick_first_variable

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "02_pybamm_model" / "outputs" / "baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BOL DFN + lumped thermal simulation.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--c-rate", type=float, default=0.5, help="Discharge C-rate for baseline curve.")
    parser.add_argument("--ambient-temp-c", type=float, default=25.0, help="Ambient temperature in Celsius.")
    parser.add_argument("--initial-soc", type=float, default=1.0, help="Initial SOC between 0 and 1.")
    parser.add_argument("--capacity-ah", type=float, default=5.0)
    parser.add_argument("--resistance-ohm", type=float, default=0.01)
    parser.add_argument("--sei-thickness-m", type=float, default=5e-9)
    parser.add_argument("--li-inventory-ratio", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bol = BOLParameters(
        initial_capacity_ah=args.capacity_ah,
        initial_internal_resistance_ohm=args.resistance_ohm,
        initial_sei_thickness_m=args.sei_thickness_m,
        initial_lithium_inventory_ratio=args.li_inventory_ratio,
    )
    thermal = ThermalParameters()
    baseline = BaselineDFNThermalModel(bol=bol, thermal=thermal)
    baseline.parameter_values.update(
        {"Ambient temperature [K]": args.ambient_temp_c + 273.15},
        check_already_exists=False,
    )

    experiment = pybamm.Experiment([f"Discharge at {args.c_rate}C until 2.5 V"], period="10 seconds")
    simulation = baseline.create_simulation(experiment=experiment)
    solution = simulation.solve(initial_soc=args.initial_soc)

    _, capacity_var = pick_first_variable(solution, ["Discharge capacity [A.h]"])
    _, voltage_var = pick_first_variable(solution, ["Terminal voltage [V]"])
    _, temperature_var = pick_first_variable(
        solution,
        ["X-averaged cell temperature [K]", "Volume-averaged cell temperature [K]", "Cell temperature [K]"],
    )

    results = pd.DataFrame(
        {
            "time_s": solution.t,
            "capacity_ah": capacity_var(solution.t),
            "voltage_v": voltage_var(solution.t),
            "temperature_c": temperature_var(solution.t) - 273.15,
        }
    )
    results.to_csv(args.output_dir / "bol_curves.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(results["capacity_ah"], results["voltage_v"], linewidth=2)
    plt.xlabel("Discharge Capacity [Ah]")
    plt.ylabel("Terminal Voltage [V]")
    plt.title("BOL Voltage-Capacity Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output_dir / "bol_voltage_capacity.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(results["time_s"] / 60.0, results["temperature_c"], linewidth=2)
    plt.xlabel("Time [min]")
    plt.ylabel("Cell Temperature [°C]")
    plt.title("BOL Temperature Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output_dir / "bol_temperature.png", dpi=150)
    plt.close()

    print(f"Saved baseline outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
