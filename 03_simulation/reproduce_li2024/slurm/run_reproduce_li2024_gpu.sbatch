#!/bin/bash
#SBATCH --job-name=alpha-repro-gpu
#SBATCH --partition=GPUcompute
#SBATCH --gres=gpu:1
#SBATCH --output=/home/zxsun/AlphaBattery/logs/%x-%j.out
#SBATCH --error=/home/zxsun/AlphaBattery/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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
    --pybamm-root "${PYBAMM_ROOT}"
