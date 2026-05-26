"""Physics-based coupled degradation runner using PyBaMM + external data interfaces."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[4]
EXTERNAL_ROOT = REPO_ROOT / "external"
if str(EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ROOT))

import pandas as pd
import pybamm

from rpt_features import compute_dqdv_curve, compute_soh_pct, features_to_frame

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "02_pybamm_model" / "closed_loop_model"
RESULTS_ROOT = PROJECT_ROOT / "results"


def load_parameter_values_helper(parameter_set: str) -> pybamm.ParameterValues:
    try:
        return pybamm.ParameterValues(parameter_set)
    except Exception:
        module_name = f"pybamm.input.parameters.lithium_ion.{parameter_set}"
        module = importlib.import_module(module_name)
        return pybamm.ParameterValues(module.get_parameter_values())


def _default_model_options() -> Dict[str, Any]:
    return {
        "SEI": "interstitial-diffusion limited",
        "SEI on cracks": "true",
        "lithium plating": "partially reversible",
        "lithium plating porosity change": "true",
        "particle mechanics": ("swelling and cracking", "swelling only"),
        "loss of active material": "stress-driven",
        "contact resistance": "true",
        "open-circuit potential": "current sigmoid",
        "SEI film resistance": "distributed",
        "SEI porosity change": "true",
        "thermal": "isothermal"
    }


def _default_var_pts() -> Dict[str, int]:
    return {"x_n": 5, "x_s": 3, "x_p": 5, "r_n": 10, "r_p": 10}



def _default_ageing_steps() -> List[str]:
    return [
        "Discharge at 1C until 2.5 V",
        "Charge at 0.3C until 4.2 V",
        "Hold at 4.2 V until C/100",
    ]


def _default_rpt_steps() -> List[str]:
    return [
        "Rest for 30 minutes",
        "Discharge at 0.1C until 2.5 V",
        "Rest for 30 minutes",
        "Charge at 0.1C until 4.2 V",
        "Hold at 4.2 V until C/100",
        "Rest for 10 minutes",
        "Discharge at 1C for 10 seconds",
    ]


def _default_extractors() -> Dict[str, Any]:
    return {
        "capacity_step_index": 1,
        "dcir_rest_step_index": 5,
        "dcir_pulse_step_index": 6,
        "hppc_pulse_step_index": 6,
        "low_rate_discharge_step_index": 1,
        "low_rate_charge_step_index": 3,
        "pulse_pairs": [
            {
                "label": "100_soc",
                "rest_step_index": 5,
                "discharge_pulse_step_index": 6,
                "charge_pulse_step_index": 8,
            },
            {
                "label": "80_soc",
                "rest_step_index": 11,
                "discharge_pulse_step_index": 12,
                "charge_pulse_step_index": 14,
            },
            {
                "label": "50_soc",
                "rest_step_index": 17,
                "discharge_pulse_step_index": 18,
                "charge_pulse_step_index": 20,
            },
            {
                "label": "20_soc",
                "rest_step_index": 23,
                "discharge_pulse_step_index": 24,
                "charge_pulse_step_index": 26,
            },
        ],
    }


@dataclass
class DegradationModelConfig:
    parameter_set: str = "OKane2023"
    ambient_temperature_c: float = 25.0
    initial_soc: float = 1.0
    solver_preference: str = "casadi"
    solver_rtol: float = 1e-6
    solver_atol: float = 1e-6
    solver_dt_max: float | None = None
    solver_max_step_decrease_count: int = 10
    return_solution_if_failed_early: bool = False
    model_options: Dict[str, Any] = field(default_factory=_default_model_options)
    var_pts: Dict[str, int] = field(default_factory=_default_var_pts)



@dataclass
class ProtocolConfig:
    period: str = "1 minute"
    cycles_per_checkpoint: int = 50
    n_checkpoints: int = 4
    ageing_cycle_steps: List[str] = field(default_factory=_default_ageing_steps)
    rpt_steps: List[str] = field(default_factory=_default_rpt_steps)
    rpt_extractors: Dict[str, Any] = field(default_factory=_default_extractors)


def load_cycle_protocol_records(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    df = pd.read_csv(path)
    required = {"segment_id", "checkpoint", "repeat_count", "step_index", "instruction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cycle protocol CSV missing columns: {sorted(missing)}")
    return df.sort_values(["checkpoint", "step_index"]).reset_index(drop=True)


def select_solver(model_config: DegradationModelConfig) -> pybamm.BaseSolver:
    preference = model_config.solver_preference
    
    if preference == "casadi_fast":
        try:
            return pybamm.CasadiSolver(
                mode="fast",
                rtol=model_config.solver_rtol,
                atol=model_config.solver_atol,
                dt_max=model_config.solver_dt_max,
                max_step_decrease_count=model_config.solver_max_step_decrease_count,
                return_solution_if_failed_early=model_config.return_solution_if_failed_early,
            )
        except Exception:
            preference = "casadi"

    if preference == "casadi":
        return pybamm.CasadiSolver(
            mode="safe",
            rtol=model_config.solver_rtol,
            atol=model_config.solver_atol,
            dt_max=model_config.solver_dt_max,
            max_step_decrease_count=model_config.solver_max_step_decrease_count,
            return_solution_if_failed_early=model_config.return_solution_if_failed_early,
        )

    # If idaklu was requested explicitly, attempt to load it, falling back to casadi safe if unavailable
    if preference == "idaklu":
        try:
            return pybamm.IDAKLUSolver(
                rtol=model_config.solver_rtol,
                atol=model_config.solver_atol,
            )
        except Exception:
            return pybamm.CasadiSolver(
                mode="safe",
                rtol=model_config.solver_rtol,
                atol=model_config.solver_atol,
                dt_max=model_config.solver_dt_max,
                max_step_decrease_count=model_config.solver_max_step_decrease_count,
                return_solution_if_failed_early=model_config.return_solution_if_failed_early,
            )

    # Default fallback
    return pybamm.CasadiSolver(
        mode="safe",
        rtol=model_config.solver_rtol,
        atol=model_config.solver_atol,
        dt_max=model_config.solver_dt_max,
        max_step_decrease_count=model_config.solver_max_step_decrease_count,
        return_solution_if_failed_early=model_config.return_solution_if_failed_early,
    )


def safe_parameter_update(parameter_values: pybamm.ParameterValues, updates: Dict[str, Any]) -> None:
    keys = set(parameter_values.keys())
    filtered = {k: v for k, v in updates.items() if k in keys}
    if filtered:
        parameter_values.update(filtered, check_already_exists=False)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_protocol(path: Path | None) -> ProtocolConfig:
    if path is None:
        return ProtocolConfig()
    return ProtocolConfig(**load_json(path))


def load_model_config(path: Path | None, ambient_temperature_c: float | None) -> DegradationModelConfig:
    config = DegradationModelConfig() if path is None else DegradationModelConfig(**load_json(path))
    if ambient_temperature_c is not None:
        config.ambient_temperature_c = ambient_temperature_c
    return config


def load_parameter_overrides(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    return load_json(path)


def validate_operating_window(temperature_c: float) -> None:
    if not (-20.0 <= temperature_c <= 60.0):
        raise ValueError("Temperature must be within -20 to 60 C.")


def get_cycle_tuple(protocol: ProtocolConfig) -> tuple[str, ...]:
    steps: List[str] = []
    for _ in range(protocol.cycles_per_checkpoint):
        steps.extend(protocol.ageing_cycle_steps)
    steps.extend(protocol.rpt_steps)
    return tuple(steps)


def get_cycle_tuple_from_records(protocol_records: pd.DataFrame, checkpoint: int, rpt_steps: List[str]) -> tuple[str, ...]:
    segment = protocol_records[protocol_records["checkpoint"] == checkpoint]
    if segment.empty:
        raise ValueError(f"No cycle protocol rows found for checkpoint={checkpoint}")
    base_steps = segment.sort_values("step_index")["instruction"].tolist()
    repeat_count = int(segment["repeat_count"].iloc[0])
    steps: List[str] = []
    for _ in range(repeat_count):
        steps.extend(base_steps)
    steps.extend(rpt_steps)
    return tuple(steps)


def step_delta(step_solution: pybamm.Solution, variable_name: str) -> float:
    values = step_solution[variable_name].entries
    return float(values[-1] - values[0])


def step_final(step_solution: pybamm.Solution, variable_name: str) -> float:
    return float(step_solution[variable_name].entries[-1])


def step_min(step_solution: pybamm.Solution, variable_name: str) -> float:
    return float(step_solution[variable_name].entries.min())


def step_max_abs_power(step_solution: pybamm.Solution) -> float:
    current = step_solution["Current [A]"].entries
    voltage = step_solution["Terminal voltage [V]"].entries
    return float((abs(current) * voltage).max())


def compute_dcir_from_steps(rest_step: pybamm.Solution, pulse_step: pybamm.Solution) -> float:
    v_before = step_final(rest_step, "Terminal voltage [V]")
    v_after = step_min(pulse_step, "Terminal voltage [V]")
    pulse_current = abs(step_final(pulse_step, "Current [A]"))
    return abs(v_before - v_after) / max(pulse_current, 1e-9)


def solution_trace_frame(step_solution: pybamm.Solution, checkpoint: int, trace_role: str, source: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "checkpoint": checkpoint,
            "source": source,
            "trace_role": trace_role,
            "time_s": step_solution["Time [s]"].entries,
            "current_a": step_solution["Current [A]"].entries,
            "voltage_v": step_solution["Terminal voltage [V]"].entries,
            "capacity_ah": step_solution["Discharge capacity [A.h]"].entries,
        }
    )


class CoupledDegradationSimulator:
    """Run PyBaMM coupled degradation in checkpointed blocks with optional dry-out data."""

    def __init__(self, model_config: DegradationModelConfig, protocol: ProtocolConfig):
        self.model_config = model_config
        self.protocol = protocol
        self.base_parameter_values = load_parameter_values_helper(model_config.parameter_set)
        self.base_porosity = {
            "Negative electrode porosity": self.base_parameter_values["Negative electrode porosity"],
            "Separator porosity": self.base_parameter_values["Separator porosity"],
            "Positive electrode porosity": self.base_parameter_values["Positive electrode porosity"],
        }
        self.model = pybamm.lithium_ion.DFN(model_config.model_options)
        self.solver = select_solver(model_config)

    def build_parameter_values(
        self,
        parameter_overrides: Dict[str, Any],
    ) -> pybamm.ParameterValues:
        parameter_values = self.base_parameter_values.copy()
        safe_parameter_update(
            parameter_values,
            {"Ambient temperature [K]": self.model_config.ambient_temperature_c + 273.15},
        )
        safe_parameter_update(parameter_values, parameter_overrides)
        return parameter_values

    def run(
        self,
        parameter_overrides: Dict[str, Any] | None = None,
        cycle_protocol_records: pd.DataFrame | None = None,
        output_csv: Path | None = None,
        output_trace_csv: Path | None = None,
        output_feature_csv: Path | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        validate_operating_window(self.model_config.ambient_temperature_c)
        parameter_overrides = parameter_overrides or {}
        rows: List[Dict[str, float]] = []
        trace_frames: List[pd.DataFrame] = []
        feature_frames: List[pd.DataFrame] = []
        starting_solution = None

        for checkpoint in range(1, self.protocol.n_checkpoints + 1):
            if cycle_protocol_records is None:
                cycle_definition = get_cycle_tuple(self.protocol)
            else:
                cycle_definition = get_cycle_tuple_from_records(
                    protocol_records=cycle_protocol_records,
                    checkpoint=checkpoint,
                    rpt_steps=self.protocol.rpt_steps,
                )
            experiment = pybamm.Experiment([cycle_definition], period=self.protocol.period)
            parameter_values = self.build_parameter_values(
                parameter_overrides=parameter_overrides,
            )
            simulation = pybamm.Simulation(
                self.model,
                parameter_values=parameter_values,
                experiment=experiment,
                solver=self.solver,
                var_pts=self.model_config.var_pts,
            )
            solve_kwargs: Dict[str, Any] = {"starting_solution": starting_solution}
            if starting_solution is None:
                solve_kwargs["initial_soc"] = self.model_config.initial_soc
            # Our workflow extracts custom RPT metrics directly from the solution.
            # Disabling eSOH summary avoids long-run failures in PyBaMM cycle summaries.
            solve_kwargs["calc_esoh"] = False
            solution = simulation.solve(**solve_kwargs)
            cycle_solution = solution.cycles[-1]
            summary_row, trace_df, feature_df = self.extract_cycle_artifacts(
                cycle_solution=cycle_solution,
                checkpoint=checkpoint,
            )
            rows.append(summary_row)
            trace_frames.append(trace_df)
            feature_frames.append(feature_df)
            starting_solution = solution

        summary = compute_soh_pct(pd.DataFrame(rows))
        traces = pd.concat(trace_frames, ignore_index=True) if trace_frames else pd.DataFrame()
        features = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
        if output_csv is not None:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(output_csv, index=False)
        if output_trace_csv is not None:
            output_trace_csv.parent.mkdir(parents=True, exist_ok=True)
            traces.to_csv(output_trace_csv, index=False)
        if output_feature_csv is not None:
            output_feature_csv.parent.mkdir(parents=True, exist_ok=True)
            features.to_csv(output_feature_csv, index=False)
        return summary, traces, features

    def extract_cycle_artifacts(
        self, cycle_solution: pybamm.Solution, checkpoint: int
    ) -> tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
        steps = cycle_solution.steps
        rpt_len = len(self.protocol.rpt_steps)
        if len(steps) < rpt_len:
            raise RuntimeError(
                f"Checkpoint {checkpoint} failed before completing the full RPT sequence. "
                f"Expected at least {rpt_len} steps, got {len(steps)}."
            )
        rpt_offset = len(steps) - rpt_len
        extractors = self.protocol.rpt_extractors

        capacity_step = steps[rpt_offset + int(extractors["capacity_step_index"])]
        dcir_rest_step = steps[rpt_offset + int(extractors["dcir_rest_step_index"])]
        dcir_pulse_step = steps[rpt_offset + int(extractors["dcir_pulse_step_index"])]
        hppc_pulse_step = steps[rpt_offset + int(extractors["hppc_pulse_step_index"])]
        low_rate_discharge_step = steps[rpt_offset + int(extractors["low_rate_discharge_step_index"])]
        low_rate_charge_step = steps[rpt_offset + int(extractors["low_rate_charge_step_index"])]
        dcir_ohm = compute_dcir_from_steps(dcir_rest_step, dcir_pulse_step)

        row = {
            "checkpoint": checkpoint,
            "cycle": checkpoint * self.protocol.cycles_per_checkpoint,
            "capacity_01c_ah": step_delta(capacity_step, "Discharge capacity [A.h]"),
            "dcir_ohm": dcir_ohm,
            "hppc_power_w": step_max_abs_power(hppc_pulse_step),
            "throughput_ah": step_final(cycle_solution, "Throughput capacity [A.h]"),
            "lli_pct": step_final(cycle_solution, "Loss of lithium inventory [%]"),
            "lam_ne_pct": step_final(cycle_solution, "Loss of active material in negative electrode [%]"),
            "lam_pe_pct": step_final(cycle_solution, "Loss of active material in positive electrode [%]"),
            "neg_porosity": step_final(cycle_solution, "X-averaged negative electrode porosity"),
            "voltage_v": step_final(cycle_solution, "Terminal voltage [V]"),
        }
        for variable in [
            "Loss of capacity to negative SEI [A.h]",
            "Loss of capacity to negative SEI on cracks [A.h]",
            "Loss of capacity to negative lithium plating [A.h]",
        ]:
            try:
                key = variable.lower().replace(" ", "_").replace("[a.h]", "ah").replace("%", "pct")
                row[key] = step_final(cycle_solution, variable)
            except KeyError:
                continue

        pulse_pairs = extractors.get("pulse_pairs", [])
        pulse_powers: List[float] = [row["hppc_power_w"]]
        for pair in pulse_pairs:
            label = str(pair["label"])
            rest_step = steps[rpt_offset + int(pair["rest_step_index"])]
            discharge_pulse_step = steps[rpt_offset + int(pair["discharge_pulse_step_index"])]
            charge_pulse_step = steps[rpt_offset + int(pair["charge_pulse_step_index"])]
            dcir_value = compute_dcir_from_steps(rest_step, discharge_pulse_step)
            discharge_power = step_max_abs_power(discharge_pulse_step)
            charge_power = step_max_abs_power(charge_pulse_step)
            pulse_powers.extend([discharge_power, charge_power])
            row[f"dcir_{label}_ohm"] = dcir_value
            row[f"hppc_discharge_power_{label}_w"] = discharge_power
            row[f"hppc_charge_power_{label}_w"] = charge_power
        if pulse_pairs:
            first_label = str(pulse_pairs[0]["label"])
            row["dcir_ohm"] = row[f"dcir_{first_label}_ohm"]
            row["hppc_power_w"] = max(pulse_powers)

        discharge_trace = solution_trace_frame(
            low_rate_discharge_step, checkpoint=checkpoint, trace_role="low_rate_discharge", source="sim"
        )
        charge_trace = solution_trace_frame(
            low_rate_charge_step, checkpoint=checkpoint, trace_role="low_rate_charge", source="sim"
        )
        trace_df = pd.concat([discharge_trace, charge_trace], ignore_index=True)

        discharge_features = compute_dqdv_curve(discharge_trace)
        charge_features = compute_dqdv_curve(charge_trace)
        row.update(
            {
                "dqdv_peak_v_discharge": discharge_features.peak_voltage_v,
                "dqdv_peak_h_discharge": discharge_features.peak_height,
                "ica_area_discharge": discharge_features.area_abs,
                "dqdv_peak_v_charge": charge_features.peak_voltage_v,
                "dqdv_peak_h_charge": charge_features.peak_height,
                "ica_area_charge": charge_features.area_abs,
            }
        )
        feature_df = pd.concat(
            [
                features_to_frame(
                    checkpoint=checkpoint, source="sim", trace_role="low_rate_discharge", features=discharge_features
                ),
                features_to_frame(
                    checkpoint=checkpoint, source="sim", trace_role="low_rate_charge", features=charge_features
                ),
            ],
            ignore_index=True,
        )
        return row, trace_df, feature_df

    @staticmethod
    def snapshot(summary: pd.DataFrame) -> Dict[str, Any]:
        final = summary.iloc[-1].to_dict()
        final["n_rows"] = int(len(summary))
        return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run physics-based coupled degradation simulation with data interfaces.")
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
    parser.add_argument("--temperature-c", type=float, default=None)
    parser.add_argument("--output-csv", type=Path, default=RESULTS_ROOT / "degradation" / "coupled_summary.csv")
    parser.add_argument(
        "--output-trace-csv",
        type=Path,
        default=RESULTS_ROOT / "degradation" / "coupled_rpt_traces.csv",
    )
    parser.add_argument(
        "--output-feature-csv",
        type=Path,
        default=RESULTS_ROOT / "degradation" / "coupled_rpt_features.csv",
    )
    parser.add_argument("--output-state-json", type=Path, default=RESULTS_ROOT / "degradation" / "final_state.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol_json)
    model_config = load_model_config(args.model_config_json, ambient_temperature_c=args.temperature_c)
    cycle_protocol_records = load_cycle_protocol_records(args.cycle_protocol_csv)
    parameter_overrides = load_parameter_overrides(args.parameter_overrides_json)
    simulator = CoupledDegradationSimulator(model_config=model_config, protocol=protocol)
    summary, traces, features = simulator.run(
        parameter_overrides=parameter_overrides,
        cycle_protocol_records=cycle_protocol_records,
        output_csv=args.output_csv,
        output_trace_csv=args.output_trace_csv,
        output_feature_csv=args.output_feature_csv,
    )

    args.output_state_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_state_json.open("w", encoding="utf-8") as f:
        json.dump(simulator.snapshot(summary), f, indent=2)

    print(f"Saved degradation summary: {args.output_csv.resolve()}")
    print(f"Saved RPT traces: {args.output_trace_csv.resolve()}")
    print(f"Saved RPT features: {args.output_feature_csv.resolve()}")
    print(f"Saved final state: {args.output_state_json.resolve()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
