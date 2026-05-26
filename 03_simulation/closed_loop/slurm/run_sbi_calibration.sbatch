#!/bin/bash
#SBATCH --job-name=alpha-closed-loop-sbi
#SBATCH --partition=CPUcompute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/zxsun/AlphaBattery/03_simulation/closed_loop/results/slurm/%x-%j.out
#SBATCH --error=/home/zxsun/AlphaBattery/03_simulation/closed_loop/results/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="/home/zxsun/AlphaBattery"
CLOSED_LOOP_ROOT="${PROJECT_ROOT}/03_simulation/closed_loop"
SLURM_LOG_DIR="${CLOSED_LOOP_ROOT}/results/slurm"
ENV_NAME="pp"

# Ensure output directories exist
mkdir -p "${SLURM_LOG_DIR}"

# Locate and initialize conda
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="${HOME}/miniconda3"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="${HOME}/anaconda3"
else
    echo "Cannot find conda initialization script." >&2
    exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Navigate to closed-loop directory
cd "${CLOSED_LOOP_ROOT}"

echo "=== System Information ==="
echo "Running on host: $(hostname)"
echo "Using python: $(which python)"
echo "Environment: ${ENV_NAME}"
echo "Current directory: $(pwd)"
echo "=========================="

# Run the closed-loop pipeline using SBI optimizer
# n-init is used as the number of simulations for training the posterior density estimator
python -u src/workflow/closed_loop_pipeline.py \
    --optimizer sbi \
    --n-init 200 \
    --random-seed 42 \
    --target-rel-error 0.03

echo "Job finished at: $(date)"
