import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src" / "calibration") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src" / "calibration"))
if str(PROJECT_ROOT / "src" / "degradation") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src" / "degradation"))

from rpt_calibration import RPTCalibrator, vector_to_updates
from coupled_degradation_model import CoupledDegradationSimulator, ProtocolConfig, DegradationModelConfig

def test_vector_to_updates():
    specs = [
        {
            "name": "sei_solvent_diffusivity",
            "kind": "parameter",
            "targets": ["Outer SEI solvent diffusivity [m2.s-1]"],
            "transform": "log10",
            "bounds": [-24.0, -18.0]
        },
        {
            "name": "dryout_scale",
            "kind": "parameter",
            "targets": ["Negative electrode porosity"],
            "transform": "linear",
            "bounds": [0.7, 1.0]
        }
    ]
    x = np.array([-20.0, 0.85])
    param_updates, named_values = vector_to_updates(specs, x)
    assert np.isclose(param_updates["Outer SEI solvent diffusivity [m2.s-1]"], 1e-20)
    assert np.isclose(param_updates["Negative electrode porosity"], 0.85)
    assert named_values["sei_solvent_diffusivity"] == -20.0
    assert named_values["dryout_scale"] == 0.85

def test_extract_flat_features():
    # Construct a mock summary DataFrame
    summary_df = pd.DataFrame([
        {"checkpoint": 0, "soh_pct": 100.0, "dcir_ohm": 0.015},
        {"checkpoint": 1, "soh_pct": 95.0, "dcir_ohm": 0.018}
    ])
    
    # Construct a mock trace DataFrame
    trace_df = pd.DataFrame([
        {"checkpoint": 0, "trace_role": "discharge", "time_s": 0.0, "current_a": -0.3, "voltage_v": 4.2, "capacity_ah": 0.0},
        {"checkpoint": 0, "trace_role": "discharge", "time_s": 10.0, "current_a": -0.3, "voltage_v": 3.5, "capacity_ah": 3.0},
        {"checkpoint": 1, "trace_role": "discharge", "time_s": 0.0, "current_a": -0.3, "voltage_v": 4.1, "capacity_ah": 0.0},
        {"checkpoint": 1, "trace_role": "discharge", "time_s": 10.0, "current_a": -0.3, "voltage_v": 3.4, "capacity_ah": 2.8}
    ])
    
    # We initialize RPTCalibrator with mock components
    class DummySimulator:
        pass

    calibrator = RPTCalibrator(
        simulator=DummySimulator(),
        cycle_protocol_records=None,
        base_parameter_overrides={},
        real_summary=summary_df,
        real_trace=trace_df,
        calibration_spec=[],
        loss_weights={"summary": {}, "curve": {}},
        rel_error_target=0.03,
        model_dump_dir=Path("./tmp_test_dump")
    )
    
    features = calibrator.extract_flat_features(summary_df, trace_df)
    # 2 checkpoints, each has: SOH, DCIR, Peak V, Peak H, Area = 5 features. Total = 10 features.
    assert len(features) == 10
    assert np.isclose(features[0], 100.0)
    assert np.isclose(features[1], 0.015)
    assert np.isclose(features[5], 95.0)
    assert np.isclose(features[6], 0.018)

    # Clean up dump dir
    if Path("./tmp_test_dump").exists():
        import shutil
        shutil.rmtree("./tmp_test_dump")
