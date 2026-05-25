# AlphaBattery Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the AlphaBattery repository into four top-level functional boards without losing runnable simulation entrypoints.

**Architecture:** Keep executable subprojects internally coherent while moving them under functional top-level boards. Preserve generated artifacts, isolate vendored `pybamm/` under `external/`, and patch path-sensitive scripts so the new layout stays understandable and usable.

**Tech Stack:** Python, PowerShell filesystem moves, Markdown documentation, PyBaMM-based scripts, Stable-Baselines3 RL scripts

---

### Task 1: Create Documentation Backbone

**Files:**
- Create: `Z:/AlphaBattery/docs/superpowers/specs/2026-05-24-alpha-battery-reorg-design.md`
- Create: `Z:/AlphaBattery/docs/superpowers/plans/2026-05-24-alpha-battery-reorg.md`
- Modify: `Z:/AlphaBattery/README.md`
- Create: `Z:/AlphaBattery/01_requirements/README.md`
- Create: `Z:/AlphaBattery/02_pybamm_model/README.md`
- Create: `Z:/AlphaBattery/03_simulation/README.md`
- Create: `Z:/AlphaBattery/04_rl_optimization/README.md`

- [ ] **Step 1: Write the failing test**

There is no automated test for repository navigation. Use a structural check instead:

```powershell
Get-ChildItem 'Z:/AlphaBattery'
```

Expected: the new board directories are absent before implementation.

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
Get-ChildItem 'Z:/AlphaBattery/01_requirements'
```

Expected: path not found

- [ ] **Step 3: Write minimal implementation**

Create the board directories and add README files that explain the role of each board.

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
Get-ChildItem 'Z:/AlphaBattery/01_requirements','Z:/AlphaBattery/02_pybamm_model','Z:/AlphaBattery/03_simulation','Z:/AlphaBattery/04_rl_optimization'
```

Expected: all four directories exist

- [ ] **Step 5: Commit**

```bash
git add README.md docs/superpowers 01_requirements 02_pybamm_model 03_simulation 04_rl_optimization
git commit -m "docs: add repository reorganization docs"
```

### Task 2: Move Vendored and Closed-Loop Assets

**Files:**
- Move: `Z:/AlphaBattery/pybamm` -> `Z:/AlphaBattery/external/pybamm`
- Move: `Z:/AlphaBattery/src/baseline` -> `Z:/AlphaBattery/02_pybamm_model/closed_loop_model/src/baseline`
- Move: `Z:/AlphaBattery/configs` -> split across `Z:/AlphaBattery/02_pybamm_model/closed_loop_model/configs` and `Z:/AlphaBattery/03_simulation/closed_loop/configs`
- Move: `Z:/AlphaBattery/data` -> `Z:/AlphaBattery/03_simulation/closed_loop/data`
- Move: `Z:/AlphaBattery/notebooks` -> `Z:/AlphaBattery/03_simulation/closed_loop/notebooks`
- Move: `Z:/AlphaBattery/models` -> `Z:/AlphaBattery/03_simulation/closed_loop/results`
- Move: `Z:/AlphaBattery/slurm` -> `Z:/AlphaBattery/03_simulation/closed_loop/slurm`
- Move: `Z:/AlphaBattery/src/{degradation,calibration,workflow,__init__.py}` -> `Z:/AlphaBattery/03_simulation/closed_loop/src/...`

- [ ] **Step 1: Write the failing test**

```powershell
Get-ChildItem 'Z:/AlphaBattery/03_simulation/closed_loop/src'
```

Expected: path not found before moves

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
Get-ChildItem 'Z:/AlphaBattery/03_simulation/closed_loop/src/workflow/closed_loop_pipeline.py'
```

Expected: path not found

- [ ] **Step 3: Write minimal implementation**

Create `03_simulation/closed_loop/`, move the closed-loop executable folders under it, and isolate vendored `pybamm` under `external/`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
Get-ChildItem 'Z:/AlphaBattery/external/pybamm','Z:/AlphaBattery/03_simulation/closed_loop/src/workflow/closed_loop_pipeline.py'
```

Expected: both paths exist

- [ ] **Step 5: Commit**

```bash
git add external 02_pybamm_model 03_simulation
git commit -m "refactor: move closed-loop and vendored pybamm assets"
```

### Task 3: Move Reproduce_Li2024 and Patch Its Entrypoint

**Files:**
- Move: `Z:/AlphaBattery/Reproduce_Li2024/sim` -> `Z:/AlphaBattery/02_pybamm_model/reproduce_li2024_model/sim`
- Move: `Z:/AlphaBattery/Reproduce_Li2024/configs` -> `Z:/AlphaBattery/02_pybamm_model/reproduce_li2024_model/configs`
- Move: `Z:/AlphaBattery/Reproduce_Li2024/scripts` -> `Z:/AlphaBattery/03_simulation/reproduce_li2024/scripts`
- Move: `Z:/AlphaBattery/Reproduce_Li2024/notebooks` -> `Z:/AlphaBattery/03_simulation/reproduce_li2024/notebooks`
- Move: `Z:/AlphaBattery/Reproduce_Li2024/results` -> `Z:/AlphaBattery/03_simulation/reproduce_li2024/results`
- Move: `Z:/AlphaBattery/Reproduce_Li2024/legacy` -> `Z:/AlphaBattery/03_simulation/reproduce_li2024/legacy`
- Move: `Z:/AlphaBattery/Reproduce_Li2024/Fun_NC.py` -> `Z:/AlphaBattery/03_simulation/reproduce_li2024/legacy/Fun_NC.py`
- Modify: `Z:/AlphaBattery/03_simulation/reproduce_li2024/scripts/run_from_json.py`

- [ ] **Step 1: Write the failing test**

```powershell
python 'Z:/AlphaBattery/Reproduce_Li2024/scripts/run_from_json.py' --help
```

Expected after the move, the old path should no longer be the maintained entrypoint.

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python 'Z:/AlphaBattery/03_simulation/reproduce_li2024/scripts/run_from_json.py' --help
```

Expected before patching: import/path failure because `sim/` moved

- [ ] **Step 3: Write minimal implementation**

Patch `run_from_json.py` so it can locate the model board and import `sim.runner` from `02_pybamm_model/reproduce_li2024_model/`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python 'Z:/AlphaBattery/03_simulation/reproduce_li2024/scripts/run_from_json.py' --help
```

Expected: argparse help output and exit code 0

- [ ] **Step 5: Commit**

```bash
git add 02_pybamm_model/reproduce_li2024_model 03_simulation/reproduce_li2024
git commit -m "refactor: split reproduce li2024 model and simulation assets"
```

### Task 4: Move SmartCharging and Patch RL Imports

**Files:**
- Move: `Z:/AlphaBattery/SmartCharging/SmartCharging` -> `Z:/AlphaBattery/04_rl_optimization/SmartCharging`
- Modify: `Z:/AlphaBattery/04_rl_optimization/SmartCharging/Scripts/RLtrain/train_ppo.py`
- Modify: `Z:/AlphaBattery/04_rl_optimization/SmartCharging/Scripts/RLtrain/BatteryEnv.py`
- Modify: `Z:/AlphaBattery/04_rl_optimization/SmartCharging/Scripts/BatesianOptimization/main_bo.py`

- [ ] **Step 1: Write the failing test**

```powershell
python 'Z:/AlphaBattery/04_rl_optimization/SmartCharging/Scripts/RLtrain/train_ppo.py' --help
```

Expected before import patching: module import failure from package-style `SmartCharging...`

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python 'Z:/AlphaBattery/04_rl_optimization/SmartCharging/Scripts/RLtrain/train_ppo.py'
```

Expected: import failure before patching or runtime failure before training starts

- [ ] **Step 3: Write minimal implementation**

Patch RL imports so neighboring modules can be imported from the moved script location. Add BO README notes and conservative import fallbacks, but do not claim BO is fully wired because `config.yaml` is still missing.

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -c "from pathlib import Path; import sys; sys.path.insert(0, str(Path(r'Z:/AlphaBattery/04_rl_optimization').resolve())); import SmartCharging.Scripts.RLtrain.BatteryEnv"
```

Expected: import succeeds

- [ ] **Step 5: Commit**

```bash
git add 04_rl_optimization
git commit -m "refactor: relocate smartcharging assets"
```

### Task 5: Patch Closed-Loop Defaults and Verify New Layout

**Files:**
- Modify: `Z:/AlphaBattery/03_simulation/closed_loop/src/degradation/coupled_degradation_model.py`
- Modify: `Z:/AlphaBattery/03_simulation/closed_loop/src/calibration/rpt_calibration.py`
- Modify: `Z:/AlphaBattery/03_simulation/closed_loop/src/workflow/closed_loop_pipeline.py`
- Modify: `Z:/AlphaBattery/README.md`
- Modify: board README files

- [ ] **Step 1: Write the failing test**

```powershell
python 'Z:/AlphaBattery/03_simulation/closed_loop/src/degradation/coupled_degradation_model.py' --help
```

Expected before patching: help may show stale default output paths such as `models/...`

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python 'Z:/AlphaBattery/03_simulation/closed_loop/src/workflow/closed_loop_pipeline.py' --help
```

Expected: stale path defaults referencing pre-reorg layout

- [ ] **Step 3: Write minimal implementation**

Patch default paths from `models/...` to `results/...` and update docs to match the new tree.

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python 'Z:/AlphaBattery/03_simulation/closed_loop/src/degradation/coupled_degradation_model.py' --help
python 'Z:/AlphaBattery/03_simulation/reproduce_li2024/scripts/run_from_json.py' --help
python -c "from pathlib import Path; import sys; sys.path.insert(0, str(Path(r'Z:/AlphaBattery/04_rl_optimization').resolve())); import SmartCharging.Scripts.RLtrain.BatteryEnv"
```

Expected: all commands exit 0

- [ ] **Step 5: Commit**

```bash
git add README.md 03_simulation 04_rl_optimization
git commit -m "refactor: finish repository reorganization"
```
