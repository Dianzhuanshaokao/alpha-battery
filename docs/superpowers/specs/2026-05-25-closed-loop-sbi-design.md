# Design Spec: Integrating Simulation-Based Inference (SBI) for Closed-Loop Battery Calibration

This document specifies the design for integrating Simulation-Based Inference (SBI) as an optimization and parameter estimation backend in the `03_simulation/closed_loop` module of AlphaBattery.

## 1. Objectives & Context
Currently, `03_simulation/closed_loop/src/calibration/rpt_calibration.py` supports two optimization modes:
- **`bayesian`**: Gaussian Process Bayesian Optimization with Expected Improvement.
- **`de`**: Differential Evolution (Scipy).

We will implement the third option:
- **`sbi`**: Simulation-Based Inference using Single-Round Neural Posterior Estimation (SNPE-C) to obtain full posterior distributions of calibrated parameters under the given observed features.

Additionally:
- We will completely remove the difficult-to-obtain `electrolyte_dryout_profile.csv` file and its corresponding interfaces/parameters in the code.
- We will rely entirely on PyBaMM's built-in dynamic porosity change model (`SEI porosity change` and `lithium plating porosity change`).
- We will generate mock templates for all remaining input CSV files in `03_simulation/closed_loop/data` to ensure the codebase executes correctly out-of-the-box.

---

## 2. Dynamic Porosity & Code Cleanup
To eliminate the hard-to-measure dependency on `electrolyte_dryout_profile.csv`:
1. **PyBaMM Internal Porosity Change**: Ensure PyBaMM's `SEI porosity change` and `lithium plating porosity change` are enabled in model options.
2. **Remove File Dependency**: Delete the `electrolyte_dryout_profile.csv` file from `03_simulation/closed_loop/data`.
3. **Refactor Code Interfaces**: 
   - Remove `--dryout-profile-csv` from `closed_loop_pipeline.py` and `rpt_calibration.py` command line arguments.
   - Remove `dryout_profile` and `dryout_scale` parameters from `CoupledDegradationSimulator.run()`, `build_parameter_values()`, and `RPTCalibrator`.
   - Remove the `dryout_scale` parameter update and logging from `rpt_calibration.py` and `closed_loop_pipeline.py`.
   - Remove the `"dryout_scale"` parameter from `calibration_map.json` and adjust the calibration bounds to only include physical parameters.

---

## 3. SBI Pipeline (Single-Round SNPE) Design

### 3.1. Workflow Architecture
```
Prior Uniform Distribution (Bounds from calibration_map.json)
                   │
                   ▼
       Sample N sets of Parameters (θ)
                   │
                   ▼
     Run PyBaMM Simulations in Parallel
                   │
                   ▼
     Extract Concatenated Feature Vectors (X_sim)
                   │
                   ▼
        Train SNPE Estimator (MAF)
                   │
                   ▼
   Infer Posterior p(θ | X_obs) for Real Data (X_obs)
                   │
                   ▼
      Calculate Mean/MAP Parameters
```

### 3.2. Feature Representation ($X$)
To handle sequential timeseries curves and summaries in a fixed-size vector for neural network training:
For each checkpoint $k \in \{1, \dots, K\}$:
1. Extract summary features: `capacity_01c_ah`, `dcir_ohm`.
2. Extract ICA curve features: `peak_voltage_v`, `peak_height`, `area_abs`.
3. Concatenate these features into a single flat 1D vector:
   $$X = [SOH_1, DCIR_1, V_{peak,1}, H_{peak,1}, A_{area,1}, \dots, SOH_K, DCIR_K, V_{peak,K}, H_{peak,K}, A_{area,K}]$$

### 3.3. Density Estimator Configuration
- **Package**: `sbi` (Simulation-Based Inference in PyTorch).
- **Algorithm**: `sbi.inference.SNPE_C` (Sequential/Single-Round Neural Posterior Estimation).
- **Neural Density Estimator**: Masked Autoregressive Flow (MAF) with 5 flow layers and 50 hidden units per layer.
- **Parameters**: Log-transformed values mapped to the physical parameters via `calibration_map.json`.

---

## 4. Input File Templates & Mock Data (in `closed_loop/data/`)

We will generate the following mock data templates in the workspace:

### 4.1. `real_rpt_summary.csv`
```csv
checkpoint,cycle,capacity_01c_ah,dcir_ohm,dcir_100_soc_ohm,dcir_80_soc_ohm,dcir_50_soc_ohm,dcir_20_soc_ohm,hppc_power_w,lli_pct,lam_ne_pct,lam_pe_pct,soh_pct
0,0,3.0,0.015,0.015,0.015,0.015,0.015,30.0,0.0,0.0,0.0,100.0
1,50,2.85,0.018,0.018,0.018,0.018,0.018,28.0,3.0,2.0,1.0,95.0
```

### 4.2. `real_rpt_low_rate_trace.csv`
Provide charge/discharge voltage and capacity traces for the initial and aged checkpoints.

### 4.3. `real_cycle_timeseries.csv`
Provide mock current, voltage, capacity, and temperature timeseries representing regular aging cycles.

### 4.4. `cycle_protocols.csv`
```csv
segment_id,checkpoint,repeat_count,step_index,instruction
1,1,50,1,Discharge at 1C until 2.5 V
1,1,50,2,Charge at 0.3C until 4.2 V
1,1,50,3,Hold at 4.2 V until C/100
```

---

## 5. Dependencies
Add the following to the project's `requirements.txt`:
```txt
sbi>=0.22.0
torch>=2.0.0
```
