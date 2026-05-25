# Section 03: Simulation Run Report - Reproducing Li2024 Model

We have successfully configured and launched the reproduction simulation for the Li2024 model (Section 03) on the GPU cluster. Below is a detailed summary of the issues encountered, the solutions implemented, and the current run status.

## 1. Solver Compatibility & KLU Dependency Bypass
The default configuration file `okane2023_full_cycle.json` specifies `IDAKLUSolver` as the solver. However:
- The compiled KLU library (`_idaklu.so`) is not built or available in either the standard conda environment `pp` or the custom `external/pybamm` source.
- Building the KLU solver from source on the cluster requires external system dependencies (SuiteSparse, sundials) and compilation tools.

**Solution**: 
We changed the solver to `CasadiSolver` in the case configuration file `okane2023_full_cycle.json`. `CasadiSolver` runs out-of-the-box using the built-in CasADi solver in PyBaMM without any binary compilation requirements, while solving the identical mathematical degradation equations.

## 2. Parameter Set Module Resolving Fallback
When executing the simulation using a standard PyBaMM package inside the conda environment, standard parameter loading failed to find the custom parameter set `OKane2023` because it only exists in the local source directory under `external/pybamm/input/parameters/lithium_ion/OKane2023.py`.
Furthermore, newer versions of PyBaMM raise `ValueError` rather than `FileNotFoundError` when a parameter set name is unknown.

**Solution**:
We updated `load_parameter_values` in `02_pybamm_model/reproduce_li2024_model/sim/config.py` to:
1. Catch both `FileNotFoundError` and `ValueError` to handle different PyBaMM API versions.
2. Dynamically locate the custom parameter file `OKane2023.py` in the local `external/pybamm` submodule directory using the project root path.
3. Import the file dynamically using `importlib.util` and load the parameter dictionary via its `get_parameter_values()` function.

## 3. Remote Git Sync
All changes have been successfully committed to the remote repository on the server:
- `02_pybamm_model/reproduce_li2024_model/sim/config.py` (Fallback loading mechanism)
- `02_pybamm_model/reproduce_li2024_model/configs/cases/okane2023_full_cycle.json` (Solver switched to `CasadiSolver`)
- `03_simulation/reproduce_li2024/slurm/run_reproduce_li2024.sbatch` and `run_reproduce_li2024_gpu.sbatch` (Updated project paths and activated `pp` conda environment)

## 4. Current Job Status
A Slurm job has been submitted to the `GPUcompute` partition on the cluster:
- **Job ID**: `30244`
- **Partition**: `GPUcompute`
- **Node**: `c1`
- **Status**: Running (`R`)
- **Output Log**: `/home/zxsun/AlphaBattery/logs/alpha-repro-gpu-30244.out`
- **Error Log**: `/home/zxsun/AlphaBattery/logs/alpha-repro-gpu-30244.err`

The job is running smoothly without any compatibility errors or initial failures. We will continue monitoring the job until completion.
