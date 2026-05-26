import sys
from pathlib import Path
import importlib
import json

# Add external directory to sys.path first to use vendored pybamm
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent.parent
EXTERNAL_ROOT = REPO_ROOT / "external"
if str(EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ROOT))

import pybamm
import pandas as pd

sys.path.append(str(PROJECT_ROOT / "src" / "degradation"))
sys.path.append(str(PROJECT_ROOT / "src" / "calibration"))

from coupled_degradation_model import CoupledDegradationSimulator, load_model_config, load_protocol, load_cycle_protocol_records
from rpt_calibration import load_parameter_overrides

def load_parameter_values_helper(parameter_set: str):
    try:
        return pybamm.ParameterValues(parameter_set)
    except Exception:
        module_name = f"pybamm.input.parameters.lithium_ion.{parameter_set}"
        module = importlib.import_module(module_name)
        return pybamm.ParameterValues(module.get_parameter_values())

def _flatten_degradation_updates(value):
    flattened = {}
    for key, item in value.items():
        if isinstance(item, dict):
            flattened.update(_flatten_degradation_updates(item))
        else:
            flattened[key] = item
    return flattened

def main():
    print("Testing OKane2023 in 'safe' mode WITHOUT a Rest step...")
    
    # 1. Load config file
    config_path = REPO_ROOT / "02_pybamm_model" / "reproduce_li2024_model" / "configs" / "cases" / "okane2023_full_cycle.json"
    with config_path.open("r") as f:
        cfg = json.load(f)
        
    # 2. Build options
    options = cfg["model"]["options"]
    options = {k: (tuple(v) if isinstance(v, list) else v) for k, v in options.items()}
    
    model = pybamm.lithium_ion.DFN(options)
    parameter_values = load_parameter_values_helper(cfg["model"]["parameter_set"])
    
    # 3. Merge ONLY config updates
    updates = dict(cfg.get("parameter_updates", {}))
    updates.update(_flatten_degradation_updates(cfg.get("degradation_parameter_updates", {})))
    
    parameter_values.update(updates, check_already_exists=False)
    
    # 4. Build mesh points (var_pts)
    var = pybamm.standard_spatial_vars
    mesh_cfg = cfg["mesh_points"]
    var_pts = {
        var.x_n: int(mesh_cfg.get("x_n", 5)),
        var.x_s: int(mesh_cfg.get("x_s", 3)),
        var.x_p: int(mesh_cfg.get("x_p", 5)),
        var.r_n: int(mesh_cfg.get("r_n", 10)),
        var.r_p: int(mesh_cfg.get("r_p", 10)),
    }
    
    # 5. Build solver in "safe" mode
    solver = pybamm.CasadiSolver(
        mode="safe",
        dt_max=30.0,
        rtol=1e-6,
        atol=1e-6,
    )
    
    steps = ["Discharge at 1C until 2.5 V"]
    
    solution = None
    
    for idx, step in enumerate(steps):
        print(f"\n--- Simulating Step {idx}: '{step}' ---")
        try:
            experiment = pybamm.Experiment([step], period="1 minute")
            simulation = pybamm.Simulation(
                model,
                parameter_values=parameter_values,
                experiment=experiment,
                solver=solver,
                var_pts=var_pts,
            )
            solve_kwargs = {"calc_esoh": False}
            if solution is not None:
                solve_kwargs["starting_solution"] = solution
                
            solution = simulation.solve(**solve_kwargs)
            print(f"  SUCCESS! Terminal voltage range: {solution['Terminal voltage [V]'].entries[0]:.4f} V to {solution['Terminal voltage [V]'].entries[-1]:.4f} V")
        except Exception as e:
            print(f"  FAILED on step {idx}: '{step}'")
            import traceback
            traceback.print_exc()
            break

if __name__ == "__main__":
    main()
