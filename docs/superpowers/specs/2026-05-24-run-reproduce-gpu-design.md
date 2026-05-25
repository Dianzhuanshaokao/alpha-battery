# Spec: Run Reproduce Li2024 on Cluster GPU

- **Date**: 2026-05-24
- **Author**: Antigravity
- **Target environment**: Remote GPU partition (`GPUcompute`) on server `121.48.164.50` (user `zxsun`)

---

## 1. Intent & Context
We need to run the `reproduce_li2024` simulation workflow on the remote cluster's GPU resources. The current codebase contains a CPU-focused `run_reproduce_li2024.sbatch` script that is configured for a different user's home directory (`yanli/zhongxiansun`) and conda env.
We will create a separate, dedicated GPU sbatch file to target `GPUcompute` partition, correct the file paths to `/home/zxsun/AlphaBattery`, load the `gpu` module, activate the local `pp` conda environment, and execute the run.

---

## 2. Design Specification

### 2.1 Slurm Job Requirements
- **Job Name**: `alpha-repro-gpu`
- **Partition**: `GPUcompute`
- **Gres**: `gpu:1` (Request 1 GPU)
- **CPUs per Task**: `4`
- **Memory**: `32G`
- **Time limit**: `48:00:00`
- **Stdout Log File**: `/home/zxsun/AlphaBattery/logs/alpha-repro-gpu-%j.out`
- **Stderr Log File**: `/home/zxsun/AlphaBattery/logs/alpha-repro-gpu-%j.err`

### 2.2 Sbatch Script Template
The script will be created at: `03_simulation/reproduce_li2024/slurm/run_reproduce_li2024_gpu.sbatch`

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

# Prepare directories
mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_ROOT}" "${MPLCONFIGDIR}"
export PYBAMM_ROOT="${PROJECT_ROOT}/external/pybamm"
export MPLCONFIGDIR

# Load GPU module
module load gpu

# Activate Conda env pp
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

# Run simulation
python -u scripts/run_from_json.py \
    --config "${CONFIG_PATH}" \
    --output-root "${OUTPUT_ROOT}" \
    --pybamm-root "${PROJECT_ROOT}/external/pybamm"
```

---

## 3. Verification Plan
1. Write the `.sbatch` script to `z:\AlphaBattery\03_simulation\reproduce_li2024\slurm\run_reproduce_li2024_gpu.sbatch`.
2. Confirm the file has synchronized via RaiDrive to `/home/zxsun/AlphaBattery/03_simulation/reproduce_li2024/slurm/run_reproduce_li2024_gpu.sbatch`.
3. Submit the sbatch job via SSH:
   `ssh EMI_zxsun "sbatch /home/zxsun/AlphaBattery/03_simulation/reproduce_li2024/slurm/run_reproduce_li2024_gpu.sbatch"`
4. Check the job status with `squeue` to confirm it is running on a GPUcompute node.
5. Wait for execution and verify logs output.
