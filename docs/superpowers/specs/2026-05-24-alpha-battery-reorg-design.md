# AlphaBattery Repository Reorganization Design

**Date:** 2026-05-24

## Goal

Reorganize the `AlphaBattery` repository around four top-level functional boards so the project is easier to navigate:

1. `01_requirements`: project requirements, research scope, data contracts, and workflow descriptions
2. `02_pybamm_model`: PyBaMM-facing model code, model configs, and model-specific utilities
3. `03_simulation`: executable simulation pipelines, notebooks, data inputs, and simulation results
4. `04_rl_optimization`: reinforcement-learning and optimization demos, scripts, and outputs

The vendored `pybamm/` source should be treated as an external upstream dependency rather than as the main project structure.

## Current Problems

- The repository currently mixes three different workstreams at the root:
  - closed-loop real-data + PyBaMM calibration under `src/`, `configs/`, `data/`, `models/`
  - pure PyBaMM reproduction work under `Reproduce_Li2024/`
  - RL charging demo under `SmartCharging/`
- The vendored `pybamm/` source sits beside project code, making it harder to distinguish upstream dependency code from local project code.
- Generated artifacts and executable code are interleaved.
- Several scripts assume their current relative location, so naive file moves would break entrypoints.

## Approved Structure

```text
AlphaBattery/
├── README.md
├── docs/
├── external/
│   └── pybamm/
├── 01_requirements/
├── 02_pybamm_model/
├── 03_simulation/
├── 04_rl_optimization/
├── requirements.txt
└── .gitignore
```

## Mapping Rules

### 01 Requirements

Contains:

- top-level project scope docs
- data format and workflow descriptions
- Reproduce_Li2024 research scope summary

This board is documentation-first. It should not become a second code location.

### 02 PyBaMM Model

Contains model-oriented code and configs:

- closed-loop baseline model code from `src/baseline/`
- model/protocol/override JSON files from the root `configs/`
- Reproduce_Li2024 model-core code from `Reproduce_Li2024/sim/`
- Reproduce_Li2024 case configs from `Reproduce_Li2024/configs/`
- PyBaMM helper script `scripts/build_local_idaklu.py`

### 03 Simulation

Contains executable simulation workflows and artifacts:

- closed-loop simulation/calibration/workflow code from `src/degradation/`, `src/calibration/`, `src/workflow/`
- root `data/`, `notebooks/`, `slurm/`, and generated `models/` outputs
- Reproduce_Li2024 runnable scripts, notebooks, legacy material, and results

### 04 RL Optimization

Contains the SmartCharging demo:

- RL training scripts
- Bayesian optimization scripts
- plotting notebooks/scripts
- all SmartCharging datasets and result folders

The Bayesian optimization entrypoint currently depends on a missing `config.yaml`. This reorganization will preserve the code and document that the BO entrypoint is not yet fully wired up.

### External Dependency

`pybamm/` moves to `external/pybamm/` and is documented as an upstream dependency mirror.

## Script Compatibility Strategy

### Closed-loop simulation

Move the closed-loop project under `03_simulation/closed_loop/` and update default paths to point at:

- `03_simulation/closed_loop/configs/`
- `03_simulation/closed_loop/data/`
- `03_simulation/closed_loop/results/`

The `PROJECT_ROOT = Path(__file__).resolve().parents[2]` pattern remains valid after the move because the internal `src/` layout is preserved.

### Reproduce_Li2024

Split the model core and executable assets:

- `02_pybamm_model/reproduce_li2024_model/`
- `03_simulation/reproduce_li2024/`

Update `run_from_json.py` so it can locate the moved `sim/` package and configs in the model board while keeping outputs and notebooks in the simulation board.

### SmartCharging

Move the package to `04_rl_optimization/SmartCharging/`.
Patch package-style imports so the RL entrypoints can still resolve neighboring modules after the move.
Do not claim the BO entrypoint is fixed.

## Documentation Deliverables

- new top-level `README.md` as a navigation page
- one `README.md` per board
- requirements-board docs that summarize the two PyBaMM tracks and the RL demo

## Constraints

- preserve existing results inside the repo
- avoid deleting historical material; keep it under `legacy/` where applicable
- do not silently represent the BO entrypoint as working
- keep path changes minimal where possible

## Success Criteria

- the repository root is organized by the four approved boards
- `pybamm/` is isolated under `external/`
- the main closed-loop simulation entrypoints use the new directory layout
- the Reproduce_Li2024 runnable entrypoint uses the new split layout
- the RL training entrypoint resolves local imports after the move
- README files explain where to look for each kind of work
