#!/usr/bin/env bash
# ALMA Wheat Festival — Launch Fine-Tuning
ALMA_TRAIN_VERSION="0.0.3"; echo "ALMA_Train_Wheat_Festival.sh v${ALMA_TRAIN_VERSION}"
#
# Fine-tunes OlmoEarth-Base on the Meilha wheat field segmentation task.
# Run from $HOME/dev/olmoearth_projects/ with the venv activated.
#
# Usage:
#   bash ALMA_Train_Wheat_Festival.sh                    # use default /data/alma_wheat_festival
#   bash ALMA_Train_Wheat_Festival.sh /path/to/scratch   # custom scratch dir
#
# After training, the best checkpoint is saved to:
#   ${ALMA_SCRATCH}/trainer_checkpoints/best-*.ckpt
# Use this checkpoint path in ALMA_use_wheat_Festival inference calls.

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Paths and environment
# ---------------------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_PATH="${REPO_DIR}/olmoearth_run_data/wheat_festival"
ALMA_SCRATCH="${1:-${HOME}/RSDATA/alma_wheat_festival}"
TRAINER_CHECKPOINTS="${ALMA_SCRATCH}/trainer_checkpoints"
PREDICTION_OUTPUT_LAYER="output"

echo "=== ALMA Wheat Festival — Fine-Tuning ==="
echo "  Project:      ${PROJECT_PATH}"
echo "  Scratch:      ${ALMA_SCRATCH}"
echo "  Checkpoints:  ${TRAINER_CHECKPOINTS}"

# ---------------------------------------------------------------------------
# 1. Pre-flight checks
# ---------------------------------------------------------------------------
if [ ! -d "${ALMA_SCRATCH}/dataset" ]; then
    echo "ERROR: Dataset not found at ${ALMA_SCRATCH}/dataset"
    echo "       Run ALMA_Import_Data.sh first."
    exit 1
fi

if ! python -c "import torch; assert torch.cuda.is_available(), 'No GPU'" 2>/dev/null; then
    echo "WARNING: No CUDA GPU detected. Training will be very slow on CPU."
    read -p "Continue anyway? [y/N] " yn
    [[ "${yn}" == [Yy]* ]] || exit 1
fi

if ! python -c "import rslearn; import olmoearth_projects" 2>/dev/null; then
    echo "ERROR: rslearn not importable. Activate the venv first."
    echo "       source ${REPO_DIR}/.venv/bin/activate"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Environment variables required by model.yaml
# ---------------------------------------------------------------------------
export DATASET_PATH="${ALMA_SCRATCH}/dataset"
export TRAINER_DATA_PATH="${TRAINER_CHECKPOINTS}"
export PREDICTION_OUTPUT_LAYER="${PREDICTION_OUTPUT_LAYER}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

# Logging: MLflow → SQLite at ~/RSDATA/alma_wheat_festival/alma_training.db
export WANDB_MODE=disabled   # WandB replaced by MLflow local SQLite

mkdir -p "${TRAINER_CHECKPOINTS}"

# ---------------------------------------------------------------------------
# 3. Report dataset split sizes
# ---------------------------------------------------------------------------
echo ""
echo "--- Dataset split counts ---"
GROUP="random_split"
if [ -d "${ALMA_SCRATCH}/dataset/windows/${GROUP}" ]; then
    find "${ALMA_SCRATCH}/dataset/windows/${GROUP}" \
        -maxdepth 2 -name "metadata.json" \
        -exec grep -o '"split":"[^"]*"' {} \; 2>/dev/null \
        | sort | uniq -c \
        || echo "  (Could not read split metadata)"
else
    echo "  WARNING: group '${GROUP}' not found. Check ALMA_Import_Data.sh ran successfully."
fi
echo ""

# ---------------------------------------------------------------------------
# 4. Launch fine-tuning
# ---------------------------------------------------------------------------
echo "--- Starting fine-tuning ---"
echo "  Encoder frozen for first 15 epochs, then unfrozen (end-to-end)."
echo "  Best checkpoint monitored on: val_wheat_seg/wheat_f1"
echo ""

python -m olmoearth_projects.main olmoearth_run finetune \
    --project_path "${PROJECT_PATH}" \
    --scratch_path "${ALMA_SCRATCH}"

# ---------------------------------------------------------------------------
# 5. Report best checkpoint
# ---------------------------------------------------------------------------
echo ""
echo "--- Training complete ---"
BEST_CKPT=$(find "${TRAINER_CHECKPOINTS}" -name "best-*.ckpt" 2>/dev/null \
    | sort -t= -k2 -rn | head -1 || true)
LAST_CKPT="${TRAINER_CHECKPOINTS}/last.ckpt"

if [ -n "${BEST_CKPT}" ]; then
    echo "Best checkpoint : ${BEST_CKPT}"
elif [ -f "${LAST_CKPT}" ]; then
    echo "Best checkpoint : not found — using last.ckpt"
    BEST_CKPT="${LAST_CKPT}"
else
    echo "WARNING: No checkpoint found in ${TRAINER_CHECKPOINTS}"
fi

echo ""
echo "To run inference, set:"
echo "  export ALMA_CHECKPOINT_PATH='${BEST_CKPT:-${TRAINER_CHECKPOINTS}/last.ckpt}'"
echo "  export ALMA_SCRATCH='${ALMA_SCRATCH}'"
echo "Then follow ALMA_use_wheat_Festival.md — Section 2."
