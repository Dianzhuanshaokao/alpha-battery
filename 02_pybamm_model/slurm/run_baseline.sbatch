#!/bin/bash
#SBATCH --job-name=alpha-02-baseline
#SBATCH --partition=CPUcompute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/zxsun/AlphaBattery/02_pybamm_model/outputs/slurm/%x-%j.out
#SBATCH --error=/home/zxsun/AlphaBattery/02_pybamm_model/outputs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="/home/zxsun/AlphaBattery"
OUTPUT_DIR="${PROJECT_ROOT}/02_pybamm_model/outputs/baseline"
SLURM_LOG_DIR="${PROJECT_ROOT}/02_pybamm_model/outputs/slurm"

mkdir -p "${OUTPUT_DIR}" "${SLURM_LOG_DIR}"

CONDA_BASE="${HOME}/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate pp

cd "${PROJECT_ROOT}"

echo "Host: $(hostname)"
echo "User: $(whoami)"
echo "Workdir: $(pwd)"
echo "Python: $(which python)"
python --version

python 02_pybamm_model/main.py run-baseline
