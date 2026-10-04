#!/usr/bin/env bash
# SPDX-License-Identifier: Unlicense
#
# ALMA Wheat Festival: launch the Mleiha wheat fine-tuning on OpenCL.
ALMA_TRAIN_VERSION="0.1.0"
echo "ALMA_Train_Wheat_Festival_opencl.sh v${ALMA_TRAIN_VERSION}"
#
# OpenCL counterpart of ~/dev/olmoearth_projects/ALMA_Train_Wheat_Festival.sh
# (v0.0.3): same dataset, same recipe (olmoearth_run_data/wheat_festival/
# model.yaml: frozen OlmoEarth-v1-Tiny encoder, wheat_seg head, 80 epochs,
# top 3 by val_wheat_seg/wheat_f1 + last), same output directory, with
# olmo_cl doing the training on an OpenCL GPU instead of PyTorch on CUDA.
# The checkpoints are then written as Lightning .ckpt files that the
# unchanged inference pipeline loads.
#
# Usage:
#   bash ALMA_Train_Wheat_Festival_opencl.sh                    # ~/RSDATA/alma_wheat_festival
#   bash ALMA_Train_Wheat_Festival_opencl.sh /path/to/scratch   # custom scratch dir
#
# Environment overrides (defaults in brackets):
#   DEVICE [gpu]  PLATFORM [rusticl]  EPOCHS [80]  SEED [42]
#   WEIGHTS [~/models/olmoearth/OlmoEarth-v1-Tiny/weights.pth]
#   OLMOEARTH_PROJECTS [~/dev/olmoearth_projects]  (model.yaml and the venv)
#   VALIDATE [0]   1: re-validate every .ckpt with the PyTorch model (CPU)
#   OVERWRITE [0]  1: replace an earlier run in trainer_checkpoints/
#   OLMO_CL_EXTRA  extra olmo_cl train arguments (e.g. "max_train=2 max_val=1")
#
# After training, the checkpoints are in:
#   ${ALMA_SCRATCH}/trainer_checkpoints/epoch=E-step=S.ckpt   (3 best by val F1)
#   ${ALMA_SCRATCH}/trainer_checkpoints/last.ckpt
# with run_log.jsonl (per-epoch metrics) and run_config.json next to them.

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Paths and environment
# ---------------------------------------------------------------------------
OLMO_CL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLMOEARTH_PROJECTS="${OLMOEARTH_PROJECTS:-${HOME}/dev/olmoearth_projects}"
PROJECT_PATH="${OLMOEARTH_PROJECTS}/olmoearth_run_data/wheat_festival"
MODEL_YAML="${PROJECT_PATH}/model.yaml"
PYTHON="${OLMOEARTH_PROJECTS}/.venv/bin/python"
ALMA_SCRATCH="${1:-${HOME}/RSDATA/alma_wheat_festival}"
DATASET_PATH="${ALMA_SCRATCH}/dataset"
TRAINER_CHECKPOINTS="${ALMA_SCRATCH}/trainer_checkpoints"
WEIGHTS="${WEIGHTS:-${HOME}/models/olmoearth/OlmoEarth-v1-Tiny/weights.pth}"
WEIGHTS_URL="https://huggingface.co/allenai/OlmoEarth-v1-Tiny/resolve/main"
WEIGHTS_BYTES=57268875
DEVICE="${DEVICE:-gpu}"
PLATFORM="${PLATFORM-rusticl}"
EPOCHS="${EPOCHS:-80}"
SEED="${SEED:-42}"
VALIDATE="${VALIDATE:-0}"
OVERWRITE="${OVERWRITE:-0}"
OLMO_CL="${OLMO_CL_DIR}/olmo_cl"

# Mesa rusticl only exposes the AMD GPU when asked to.
export RUSTICL_ENABLE="${RUSTICL_ENABLE:-radeonsi}"

echo "=== ALMA Wheat Festival: Fine-Tuning (OpenCL) ==="
echo "  Project:      ${PROJECT_PATH}"
echo "  Scratch:      ${ALMA_SCRATCH}"
echo "  Checkpoints:  ${TRAINER_CHECKPOINTS}"
echo "  Weights:      ${WEIGHTS}"
echo "  Device:       ${DEVICE} (platform ${PLATFORM:-any}), ${EPOCHS} epochs, seed ${SEED}"

# ---------------------------------------------------------------------------
# 1. Pre-flight checks
# ---------------------------------------------------------------------------
if [ ! -d "${DATASET_PATH}" ]; then
    echo "ERROR: Dataset not found at ${DATASET_PATH}"
    echo "       Run ALMA_Import_Data.sh (olmoearth_projects) first."
    exit 1
fi

if [ ! -f "${MODEL_YAML}" ]; then
    echo "ERROR: ${MODEL_YAML} not found (set OLMOEARTH_PROJECTS)."
    exit 1
fi

if ! "${PYTHON}" -c "import rslearn, olmoearth_pretrain, lightning" 2>/dev/null; then
    echo "ERROR: ${PYTHON} cannot import rslearn/olmoearth_pretrain/lightning."
    echo "       They are needed to write the Lightning checkpoints."
    exit 1
fi

# The encoder weights, as olmoearth_run would fetch them from Hugging Face.
if [ ! -f "${WEIGHTS}" ]; then
    echo "Downloading OlmoEarth-v1-Tiny weights to $(dirname "${WEIGHTS}")..."
    mkdir -p "$(dirname "${WEIGHTS}")"
    curl -fL -o "${WEIGHTS}.part" "${WEIGHTS_URL}/weights.pth"
    mv "${WEIGHTS}.part" "${WEIGHTS}"
    curl -fsL -o "$(dirname "${WEIGHTS}")/config.json" "${WEIGHTS_URL}/config.json"
fi
if [ "$(stat -c %s "${WEIGHTS}")" != "${WEIGHTS_BYTES}" ]; then
    echo "ERROR: ${WEIGHTS} is $(stat -c %s "${WEIGHTS}") bytes, expected ${WEIGHTS_BYTES}."
    exit 1
fi

if [ ! -x "${OLMO_CL}" ] || [ -n "$(find "${OLMO_CL_DIR}/src" "${OLMO_CL_DIR}/kernels" \
        -newer "${OLMO_CL}" -type f 2>/dev/null)" ]; then
    echo "Building olmo_cl..."
    make -C "${OLMO_CL_DIR}" >/dev/null
fi

# The counterpart of the old CUDA check: the kernels must build and pass
# on the requested device.
echo ""
echo "--- OpenCL device check ---"
if ! "${OLMO_CL}" selftest device="${DEVICE}" platform="${PLATFORM}" tokens=256; then
    echo "WARNING: OpenCL self-test failed on device=${DEVICE} platform=${PLATFORM:-any}."
    echo "         (On the AMD server: RUSTICL_ENABLE=radeonsi, PLATFORM=rusticl.)"
    echo "         Training on a CPU OpenCL device takes about a minute per crop."
    read -r -p "Continue anyway with DEVICE=auto, PLATFORM=any? [y/N] " yn
    [[ "${yn}" == [Yy]* ]] || exit 1
    DEVICE=auto
    PLATFORM=
fi

if [ -f "${TRAINER_CHECKPOINTS}/run_log.jsonl" ] && [ "${OVERWRITE}" != 1 ]; then
    echo "ERROR: ${TRAINER_CHECKPOINTS} already holds a run (run_log.jsonl)."
    echo "       Move it away, or set OVERWRITE=1 to replace it."
    exit 1
fi
if [ "${OVERWRITE}" = 1 ]; then
    rm -f "${TRAINER_CHECKPOINTS}"/*.head "${TRAINER_CHECKPOINTS}"/*.ckpt
fi
mkdir -p "${TRAINER_CHECKPOINTS}"

# ---------------------------------------------------------------------------
# 2. Report dataset split sizes and timestep order
# ---------------------------------------------------------------------------
echo ""
echo "--- Dataset ---"
"${OLMO_CL}" data dataset="${DATASET_PATH}"
echo ""

# ---------------------------------------------------------------------------
# 3. Launch fine-tuning
# ---------------------------------------------------------------------------
echo "--- Starting fine-tuning ---"
echo "  Encoder frozen for all epochs (unfreeze_at_epoch: 9999); head trained."
echo "  Best checkpoints monitored on: val wheat F1 (top 3 + last)."
echo "  Per-epoch metrics: ${TRAINER_CHECKPOINTS}/run_log.jsonl"
echo ""

# shellcheck disable=SC2086  # OLMO_CL_EXTRA is a list of key=value words.
"${OLMO_CL}" train \
    dataset="${DATASET_PATH}" \
    weights="${WEIGHTS}" \
    norm="${OLMO_CL_DIR}/data/s2_norm.tsv" \
    crops="${OLMO_CL_DIR}/data/val_test_crops.tsv" \
    out="${TRAINER_CHECKPOINTS}" \
    epochs="${EPOCHS}" seed="${SEED}" \
    device="${DEVICE}" platform="${PLATFORM}" \
    overwrite="${OVERWRITE}" ${OLMO_CL_EXTRA:-} \
    2>&1 | tee "${TRAINER_CHECKPOINTS}/train.log"

# ---------------------------------------------------------------------------
# 4. Write the Lightning checkpoints
# ---------------------------------------------------------------------------
echo ""
echo "--- Writing Lightning checkpoints ---"
for head in "${TRAINER_CHECKPOINTS}"/*.head; do
    [ -e "${head}" ] || { echo "ERROR: training wrote no .head file"; exit 1; }
    args=()
    [ "${VALIDATE}" = 1 ] && args+=(--validate)
    [ "${OVERWRITE}" = 1 ] && args+=(--overwrite)
    # last.head has no epoch/step in its name: take them from the last
    # epoch of run_log.jsonl, as Lightning's last.ckpt records them.
    if [ "$(basename "${head}")" = last.head ]; then
        last=$(tail -1 "${TRAINER_CHECKPOINTS}/run_log.jsonl")
        args+=(--epoch "$(sed 's/.*"epoch": \([0-9]*\).*/\1/' <<<"${last}")"
               --step "$(sed 's/.*"step": \([0-9]*\).*/\1/' <<<"${last}")")
    fi
    "${PYTHON}" "${OLMO_CL_DIR}/tools/export_ckpt.py" "${MODEL_YAML}" \
        "${DATASET_PATH}" "${head}" "${head%.head}.ckpt" "${args[@]}" \
        2>&1 | grep -E "^wrote|^val_|Error|error" || true
    [ -f "${head%.head}.ckpt" ] || { echo "ERROR: export of ${head} failed"; exit 1; }
done

# ---------------------------------------------------------------------------
# 5. Report best checkpoint
# ---------------------------------------------------------------------------
echo ""
echo "--- Training complete ---"
# olmo_cl ends with "Kept <file>.head (val F1 <score>)" for each kept epoch.
BEST_HEAD=$(sed -n 's/^Kept \(.*\.head\) (val F1 \([0-9.]*\))$/\2 \1/p' \
    "${TRAINER_CHECKPOINTS}/train.log" | sort -k1,1gr | head -1 | cut -d' ' -f2- || true)
BEST_CKPT="${BEST_HEAD:+${BEST_HEAD%.head}.ckpt}"
LAST_CKPT="${TRAINER_CHECKPOINTS}/last.ckpt"

if [ -n "${BEST_CKPT}" ] && [ -f "${BEST_CKPT}" ]; then
    echo "Best checkpoint : ${BEST_CKPT}"
    grep -h "^Kept" "${TRAINER_CHECKPOINTS}/train.log" | sed 's/^/  /'
elif [ -f "${LAST_CKPT}" ]; then
    echo "Best checkpoint : not found, using last.ckpt"
    BEST_CKPT="${LAST_CKPT}"
else
    echo "WARNING: No checkpoint found in ${TRAINER_CHECKPOINTS}"
fi

echo ""
echo "Timestep order used (must match at inference time):"
grep '"timestep_order"' "${TRAINER_CHECKPOINTS}/run_config.json" | sed 's/^ */  /'
echo ""
echo "To run inference, set:"
echo "  export ALMA_CHECKPOINT_PATH='${BEST_CKPT:-${LAST_CKPT}}'"
echo "  export ALMA_SCRATCH='${ALMA_SCRATCH}'"
echo "and point CHECKPOINT= (line 40) of ALMA_Inference_Wheat_Festival.sh at it."
