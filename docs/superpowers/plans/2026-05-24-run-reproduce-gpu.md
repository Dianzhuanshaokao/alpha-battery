# Run Reproduce Li2024 on Cluster GPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a custom GPU-enabled sbatch script to run the reproduce_li2024 simulation on the cluster and verify successful execution.

**Architecture:** We use a python validator to implement TDD for the sbatch script headers and paths. Once validated, we submit the sbatch script to the `GPUcompute` partition on the remote server via SSH.

**Tech Stack:** Bash, Python, Slurm, SSH

---

### Task 1: Add Sbatch Validation Test

**Files:**
- Create: `03_simulation/reproduce_li2024/slurm/validate_sbatch.py`

- [ ] **Step 1: Write the validation test**
  Create the validation script to check sbatch settings.
  
  Code for `03_simulation/reproduce_li2024/slurm/validate_sbatch.py`:
  ```python
  import sys
  from pathlib import Path

  def main():
      sbatch_path = Path(__file__).resolve().parent / "run_reproduce_li2024_gpu.sbatch"
      if not sbatch_path.is_file():
          print(f"Error: {sbatch_path} does not exist.")
          sys.exit(1)
      
      content = sbatch_path.read_text(encoding="utf-8")
      
      # 验证关键配置
      checks = {
          "#SBATCH --partition=GPUcompute": False,
          "#SBATCH --gres=gpu:1": False,
          "PROJECT_ROOT=\"/home/zxsun/AlphaBattery\"": False,
          "ENV_NAME=\"pp\"": False,
          "module load gpu": False,
      }
      
      for key in checks:
          if key in content:
              checks[key] = True
      
      # 确保不含有旧的 yanli 路径
      has_old_path = "/home/yanli" in content
      
      failed = False
      for key, passed in checks.items():
          if not passed:
              print(f"Failed check: [{key}] not found in sbatch script")
              failed = True
      
      if has_old_path:
          print("Failed check: Found old path /home/yanli in sbatch script")
          failed = True
          
      if failed:
          sys.exit(1)
          
      print("All sbatch validation checks passed successfully.")
      sys.exit(0)

  if __name__ == "__main__":
      main()
  ```

- [ ] **Step 2: Run validation to verify it fails**
  Run: `python 03_simulation/reproduce_li2024/slurm/validate_sbatch.py`
  Expected: Prints "Error: ...run_reproduce_li2024_gpu.sbatch does not exist." and exits with code 1.

- [ ] **Step 3: Commit the validation script**
  Run:
  `git add 03_simulation/reproduce_li2024/slurm/validate_sbatch.py`
  `git commit -m "test: add GPU sbatch validation script"`


### Task 2: Implement GPU Sbatch Script

**Files:**
- Create: `03_simulation/reproduce_li2024/slurm/run_reproduce_li2024_gpu.sbatch`

- [ ] **Step 1: Create the sbatch file**
  Write the following content to `03_simulation/reproduce_li2024/slurm/run_reproduce_li2024_gpu.sbatch`:
  ```bash
  #!/bin/bash
  #SBATCH --job-name=alpha-repro-gpu
  #SBATCH --partition=GPUcompute
  #SBATCH --gres=gpu:1
  #SBATCH --output=/home/zxsun/AlphaBattery/logs/%x-%j.out
  #SBATCH --error=/home/zxsun/AlphaBattery/logs/%x-%j.err
  #SBATCH --nodes=1
  #SBATCH --ntasks=1
  #SBATCH --cpus-per-task=4
  #SBATCH --mem=32G
  #SBATCH --time=48:00:00

  set -euo pipefail

  PROJECT_ROOT="/home/zxsun/AlphaBattery"
  REPRO_ROOT="${PROJECT_ROOT}/03_simulation/reproduce_li2024"
  MODEL_ROOT="${PROJECT_ROOT}/02_pybamm_model/reproduce_li2024_model"
  ENV_NAME="pp"
  CONFIG_PATH="${MODEL_ROOT}/configs/cases/okane2023_full_cycle.json"
  OUTPUT_ROOT="${REPRO_ROOT}/outputs"
  MPLCONFIGDIR="${PROJECT_ROOT}/logs/matplotlib"

  mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_ROOT}" "${MPLCONFIGDIR}"
  export PYBAMM_ROOT="${PROJECT_ROOT}/external/pybamm"
  export MPLCONFIGDIR

  module load gpu

  if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
      CONDA_BASE="${HOME}/miniconda3"
  elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
      CONDA_BASE="${HOME}/anaconda3"
  else
      echo "Cannot find conda initialization script." >&2
      exit 1
  fi

  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"

  cd "${REPRO_ROOT}"

  echo "Running on host: $(hostname)"
  echo "Using python: $(which python)"
  echo "Config: ${CONFIG_PATH}"
  echo "Output root: ${OUTPUT_ROOT}"

  python -u scripts/run_from_json.py \
      --config "${CONFIG_PATH}" \
      --output-root "${OUTPUT_ROOT}" \
      --pybamm-root "${PROJECT_ROOT}/external/pybamm"
  ```

- [ ] **Step 2: Run validation to verify it passes**
  Run: `python 03_simulation/reproduce_li2024/slurm/validate_sbatch.py`
  Expected: Prints "All sbatch validation checks passed successfully." and exits with code 0.

- [ ] **Step 3: Commit the sbatch file**
  Run:
  `git add 03_simulation/reproduce_li2024/slurm/run_reproduce_li2024_gpu.sbatch`
  `git commit -m "feat: add GPU sbatch script for reproduce_li2024"`


### Task 3: Remote Submission & Verification

**Files:**
- None (remote invocation)

- [ ] **Step 1: Run remote validation test via SSH**
  Run: `ssh EMI_zxsun "python /home/zxsun/AlphaBattery/03_simulation/reproduce_li2024/slurm/validate_sbatch.py"`
  Expected: Prints "All sbatch validation checks passed successfully." and exits with code 0.

- [ ] **Step 2: Submit the job to Slurm queue**
  Run: `ssh EMI_zxsun "sbatch /home/zxsun/AlphaBattery/03_simulation/reproduce_li2024/slurm/run_reproduce_li2024_gpu.sbatch"`
  Expected: Returns "Submitted batch job <job_id>" (e.g. 1234).

- [ ] **Step 3: Check job status in queue**
  Run: `ssh EMI_zxsun "squeue -u zxsun"`
  Expected: Shows job `alpha-repro-gpu` in partition `GPUcompute` with state `R` (Running) or `PD` (Pending).

- [ ] **Step 4: Verify outputs and log logs after completion**
  Monitor the stdout log file until the job completes.
  Run (adjusting job ID accordingly):
  `ssh EMI_zxsun "cat /home/zxsun/AlphaBattery/logs/alpha-repro-gpu-*.out"`
  Expected: Final log output shows `[DONE]` messages, final time, SOH %, and list of saved variables.
