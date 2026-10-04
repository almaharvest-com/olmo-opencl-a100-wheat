#!/usr/bin/env bash
# g13_setup.sh - from-scratch OLMO_opencl + olmoearth_projects setup, training and inference
# on a shared GPU server where you have NO sudo (written for the Maahr g13 node).
#
# Target machine (what this was built and tested on):
#   RHEL-family Linux, gcc 8.5, Python 3.6 (system), no sudo, no git, no tmux, no bzip2,
#   4x NVIDIA A100 80GB, NVIDIA OpenCL driver (/etc/OpenCL/vendors/nvidia.icd, /lib64/libOpenCL.so),
#   Slurm client present but the controller unreachable (so we run jobs directly with nohup),
#   outbound internet to GitHub, conda-forge, Hugging Face and Microsoft Planetary Computer.
#
# Inputs expected in the home folder on the server (sent from your laptop, see the guide, step 3):
#   ~/olmo_inputs.tgz                    OLMO_opencl source + the ALMA_*.sh scripts
#   ~/Sharjah_Wheat_fields_4326.geojson  the wheat field polygons
#
# Usage:
#   ./g13_setup.sh <stage>          run a stage in the foreground
#   ./g13_setup.sh bg <stage>       run a stage with nohup; log goes to ~/g13_<stage>.log
#
# Stages, in the order you run them:
#   tools    folders, unpack inputs, line-ending + permission fixes, uv, OpenCL headers, repo download
#   gdal     private GDAL (conda-forge via micromamba) + an OpenCL.pc file for the Makefile
#   build    compile olmo_cl and run its self-tests on the GPU
#   venv     Python environment for olmoearth_projects (uv sync; long download)      -> use bg
#   weights  download the OlmoEarth-v1-Tiny encoder weights from Hugging Face
#   import   patch the import script for the Tiny model, then download Sentinel-2    -> use bg
#   resume   re-run only the imagery download step (safe to repeat after a failure)  -> use bg
#   prep     tag train/val/test windows + make the label band name consistent
#   golden   PyTorch reference dumps (make golden)                                   -> use bg
#   test     compare the OpenCL head against the PyTorch dumps (make test)
#   smoke    5-epoch, 3-crop training check on one GPU
#   train    full training (80 epochs) + Lightning checkpoint export                 -> use bg
#   infer    wheat maps + field_stats.csv from the best checkpoint                   -> use bg
#   stop     stop any import/inference job that is still running
#
# Environment knobs:
#   GPU=<index>         which GPU to use (sets CUDA_VISIBLE_DEVICES). Default 0 (infer: 1).
#   OCL_DEVICE=<value>  value for olmo_cl's device= option. Default gpu.
#   CKPT=<file>         checkpoint for the infer stage. Default: best one from the training log.
#   START_DATE / END_DATE  Sentinel-2 window for the infer stage. Default 2024-11-01 / 2025-04-30
#                       (the same six-month window the training windows use).
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=disabled
INBOX="$HOME/inbox"
DEVDIR="$HOME/dev"
CL_INC="$HOME/opt/opencl"
REPO_SHA="d16c279"
REPO="$DEVDIR/olmoearth_projects"
OLMO="$DEVDIR/OLMO_opencl"
SCRATCH="$HOME/RSDATA/alma_wheat_festival"
DATASET="$SCRATCH/dataset"
PROJECT="$REPO/olmoearth_run_data/wheat_festival"
WDIR="$HOME/models/olmoearth/OlmoEarth-v1-Tiny"
GDALENV="$HOME/opt/gdalenv"
PCDIR="$HOME/opt/pc"

export C_INCLUDE_PATH="$CL_INC${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"

# ---------------------------------------------------------------------------
stage_tools() {
    mkdir -p "$DEVDIR" "$INBOX" "$HOME/opt" \
             "$HOME/GISDATA/Sharjah_WheatFestival" "$SCRATCH" "$WDIR"

    # Unpack the inputs sent from the laptop
    if [ ! -d "$INBOX/OLMO_opencl" ] && [ -f "$HOME/olmo_inputs.tgz" ]; then
        tar xzf "$HOME/olmo_inputs.tgz" -C "$INBOX"
    fi
    [ -d "$INBOX/OLMO_opencl" ] || { echo "ERROR: ~/olmo_inputs.tgz not found or incomplete."; exit 1; }
    if [ -f "$HOME/Sharjah_Wheat_fields_4326.geojson" ]; then
        cp -n "$HOME/Sharjah_Wheat_fields_4326.geojson" "$HOME/GISDATA/Sharjah_WheatFestival/"
    fi

    # Our own working copy of OLMO_opencl
    if [ ! -d "$OLMO" ]; then
        rsync -a "$INBOX/OLMO_opencl/" "$OLMO/"
        echo "copied OLMO_opencl"
    else
        echo "$OLMO already exists, not overwriting"
    fi

    # OpenCL C headers (the server has the driver but not the header files)
    if [ ! -f "$CL_INC/CL/cl.h" ]; then
        mkdir -p "$CL_INC" && cd "$CL_INC"
        curl -fsSL -o headers.tgz https://github.com/KhronosGroup/OpenCL-Headers/archive/refs/tags/v2024.05.08.tar.gz
        tar xzf headers.tgz && mv OpenCL-Headers-*/CL . && rm -rf OpenCL-Headers-* headers.tgz
    fi
    echo "CL headers: $(ls "$CL_INC/CL" | wc -l) files"

    # uv (Python package/environment manager), per-user, without editing ~/.bashrc
    if ! command -v uv >/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | UV_NO_MODIFY_PATH=1 sh
    fi
    uv --version

    # The server has neither git nor bzip2, so the repo comes as a GitHub tarball, pinned to
    # the commit we used.
    if [ ! -d "$REPO" ]; then
        if command -v git >/dev/null; then
            git clone https://github.com/allenai/olmoearth_projects.git "$REPO"
            git -C "$REPO" checkout -q "$REPO_SHA"
        else
            tmp=$(mktemp -d)
            curl -fsSL "https://github.com/allenai/olmoearth_projects/archive/$REPO_SHA.tar.gz" | tar xz -C "$tmp"
            mv "$tmp"/olmoearth_projects-* "$REPO"
        fi
    fi
    for f in ALMA_Import_Data.sh ALMA_Inference_Wheat_Festival.sh; do
        cp -n "$INBOX/Olmoearth-retraining-for-wheat-field/$f" "$REPO/"
    done

    # Files made on Windows: CRLF -> LF line endings, sane permissions
    { grep -rlI $'\r' "$OLMO" "$REPO"/ALMA_*.sh "$HOME/GISDATA" || true; } | xargs -r sed -i 's/\r$//'
    find "$OLMO" -type d -exec chmod 755 {} +
    find "$OLMO" -type f -exec chmod 644 {} +
    find "$OLMO" -name '*.sh' -exec chmod 755 {} +
    chmod 755 "$REPO"/ALMA_*.sh
    echo "remaining CRLF files:"; { grep -rlI $'\r' "$OLMO" "$REPO"/ALMA_*.sh || echo none; }
    echo "tools stage done."
}

# The Makefile finds GDAL and OpenCL through pkg-config. The server has neither dev package and
# we have no sudo, so GDAL comes from conda-forge (private env) and OpenCL.pc points at the
# system NVIDIA loader (/lib64/libOpenCL.so) plus the Khronos headers.
gdal_env() {
    export PKG_CONFIG_PATH="$PCDIR:$GDALENV/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    export LD_RUN_PATH="$GDALENV/lib"
    export LD_LIBRARY_PATH="$GDALENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export GDAL_DATA="$GDALENV/share/gdal" PROJ_DATA="$GDALENV/share/proj"
}

stage_gdal() {
    mkdir -p "$HOME/opt/mm" "$PCDIR"
    if [ ! -x "$HOME/opt/mm/micromamba" ]; then
        curl -fsSL -o "$HOME/opt/mm/micromamba" \
            https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64
        chmod +x "$HOME/opt/mm/micromamba"
    fi
    "$HOME/opt/mm/micromamba" --version
    if [ ! -f "$GDALENV/include/gdal.h" ]; then
        MAMBA_ROOT_PREFIX="$HOME/opt/mm-root" "$HOME/opt/mm/micromamba" create -y \
            -p "$GDALENV" -c conda-forge --override-channels libgdal-core
    fi
    ls -la "$GDALENV/include/gdal.h" "$GDALENV/lib/pkgconfig/gdal.pc"
    cat > "$PCDIR/OpenCL.pc" <<EOF
libdir=/lib64
includedir=$CL_INC
Name: OpenCL
Description: system OpenCL ICD loader with Khronos headers
Version: 3.0
Libs: -L\${libdir} -lOpenCL $GDALENV/lib/libgcc_s.so.1
Cflags: -I\${includedir}
EOF
    # ^ conda's libgdal needs newer (GCC 12+) symbols from libgcc_s than the server's GCC 8 copy
    #   has. Naming conda's copy here makes the linker load it before -lgdal.
    gdal_env
    echo "pkg-config says: $(pkg-config --cflags --libs gdal OpenCL)"
}

stage_build() {
    gdal_env
    cd "$OLMO"
    echo "pkg-config says: $(pkg-config --cflags --libs gdal OpenCL)"
    make clean >/dev/null 2>&1 || true
    make -j8 2>&1 | tail -25
    ls -la olmo_cl
    ldd ./olmo_cl | grep -E "gdal|OpenCL|not found" || true
    echo "--- make selftest"
    make selftest 2>&1 | tail -30
    echo "--- GPU selftest at full size"
    CUDA_VISIBLE_DEVICES="${GPU:-0}" ./olmo_cl selftest device="${OCL_DEVICE:-gpu}" tokens=4608 2>&1 | tail -30
}

stage_venv() {
    cd "$REPO"
    uv sync --locked --all-extras --python 3.12 2>&1 | tee "$HOME/uvsync.log"
    .venv/bin/python -c "import rslearn, olmoearth_projects, lightning, torch; print('imports OK, torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
}

stage_weights() {
    local url=https://huggingface.co/allenai/OlmoEarth-v1-Tiny/resolve/main
    mkdir -p "$WDIR"
    if [ ! -f "$WDIR/weights.pth" ]; then
        curl -fsL -o "$WDIR/weights.pth.part" "$url/weights.pth" && mv "$WDIR/weights.pth.part" "$WDIR/weights.pth"
        curl -fsL -o "$WDIR/config.json" "$url/config.json"
    fi
    local size; size=$(stat -c %s "$WDIR/weights.pth")
    echo "weights.pth: $size bytes (expected 57268875)"
    [ "$size" = "57268875" ] || { echo "ERROR: unexpected weights size"; exit 1; }
}

stage_import() {
    cd "$REPO"
    local F="$REPO/ALMA_Import_Data.sh"
    [ -f "$F.orig" ] || cp "$F" "$F.orig"
    cp "$F.orig" "$F"

    # Small ("Tiny") encoder, frozen, batch size 1, 64-pixel crops
    sed -i \
      -e 's/model_id: OLMOEARTH_V1_BASE .*/model_id: OLMOEARTH_V1_TINY   # Tiny encoder/' \
      -e 's/in_channels: 768/in_channels: 192/' \
      -e 's/batch_size: 4/batch_size: 1/' \
      -e 's/unfreeze_at_epoch: 15 .*/unfreeze_at_epoch: 9999   # encoder frozen for the whole run/' \
      "$F"
    "$REPO/.venv/bin/python" - "$F" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
head, sep, rest = s.partition("    predict_config:")
head = head.replace("patch_size: 128", "patch_size: 64")
head = re.sub(r"\n      load_all_patches: true", "", head)
open(p, "w").write(head + sep + rest)
PY
    echo "--- diff vs original"; diff "$F.orig" "$F" || true

    # Planetary Computer request timeout 10 s -> 120 s (lives inside the venv)
    local PC=".venv/lib/python3.12/site-packages/rslearn/data_sources/planetary_computer.py"
    sed -i 's/timeout: timedelta = timedelta(seconds=10)/timeout: timedelta = timedelta(seconds=120)/' "$PC"
    grep -n "timedelta(seconds=" "$PC" || true

    source .venv/bin/activate
    local start; start=$(date +%s)
    bash ALMA_Import_Data.sh 2>&1 | tee "$HOME/import.log" || echo "IMPORT FAILED" | tee -a "$HOME/import.log"
    echo "import wall time: $(( $(date +%s) - start )) s" | tee -a "$HOME/import.log"
}

# Re-run only the imagery download. It skips every layer that already has a "completed" marker,
# so it is safe to repeat. Use it if the import was interrupted.
stage_resume() {
    cd "$REPO"
    source .venv/bin/activate
    python -m olmoearth_projects.main olmoearth_run build_dataset_from_windows \
        --project_path "$PROJECT" --scratch_path "$SCRATCH" 2>&1 | tee -a "$HOME/import.log"
    echo "completed sentinel2 layers (expect 6 per window):"
    find "$DATASET/windows" -path '*layers/sentinel2*' -name completed | wc -l
}

# 1) Make every window carry the right train/val/test tag.
# 2) Make the label band name "label" everywhere (the generated files disagree: some say
#    "category").
stage_prep() {
    local tsv="$OLMO/data/val_test_crops.tsv"
    local base="$DATASET/windows/random_split"
    [ -d "$base" ] || { echo "ERROR: no windows in $base - run the import stage first."; exit 1; }

    echo "--- before:"; "$REPO/.venv/bin/python" - "$base" <<'PY'
import collections, glob, json, sys
c = collections.Counter(json.load(open(f))["options"].get("split") for f in glob.glob(sys.argv[1] + "/*/metadata.json"))
print(dict(c))
PY

    # The list of val/test windows ships with OLMO_opencl (data/val_test_crops.tsv). Every other
    # window is a training window. The importer's own random splitter does not always apply it,
    # so we set the tags explicitly.
    "$REPO/.venv/bin/python" - "$tsv" "$base" <<'PY'
import glob, json, os, sys
tsv, base = sys.argv[1], sys.argv[2]
want = {}
for line in open(tsv):
    if line.startswith("#") or not line.strip():
        continue
    w, split = line.rstrip("\n").split("\t")[:2]
    want[w] = split
missing = [w for w in want if not os.path.exists(f"{base}/{w}/metadata.json")]
if missing:
    print("ERROR: windows in the crops file that are not in the dataset:", missing)
    sys.exit(1)
for f in glob.glob(base + "/*/metadata.json"):
    d = json.load(open(f))
    name = os.path.basename(os.path.dirname(f))
    d["options"]["split"] = want.get(name, "train")
    json.dump(d, open(f, "w"))
PY
    echo "--- after:"; "$REPO/.venv/bin/python" - "$base" <<'PY'
import collections, glob, json, sys
c = collections.Counter(json.load(open(f))["options"].get("split") for f in glob.glob(sys.argv[1] + "/*/metadata.json"))
print(dict(c), "(expect train 58, val 8, test 8)")
PY

    # Label band name
    for f in "$DATASET/config.json" "$PROJECT/dataset.json"; do
        sed -i 's/"bands": \["category"\]/"bands": ["label"]/' "$f"
        grep -n '"bands": \["label"\]' "$f"
    done
    sed -i 's/bands: \["category"\]/bands: ["label"]/' "$PROJECT/model.yaml"
    grep -n 'bands: \["label"\]' "$PROJECT/model.yaml"
    echo "prep stage done."
}

stage_golden() {
    # make golden may re-link olmo_cl: pkg-config + rpath only (no LD_LIBRARY_PATH, which could
    # upset the torch environment).
    export PKG_CONFIG_PATH="$PCDIR:$GDALENV/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    export LD_RUN_PATH="$GDALENV/lib"
    cd "$OLMO"
    make golden 2>&1 | grep -vE "Warning|warn\(|loglevel=INFO|^\s*$" | tail -25
    echo "--- golden dumps"; ls golden | head -30
    echo "--- crops file"; cat data/val_test_crops.tsv
    du -sh golden
}

stage_test() {
    gdal_env
    cd "$OLMO"
    export CUDA_VISIBLE_DEVICES="${GPU:-0}"
    # one dump per fixed val/test crop that "make golden" wrote (val0 val1 test0 test1)
    local list; list=$(ls -d golden/val[0-9]* golden/test[0-9]* 2>/dev/null | paste -sd, -)
    echo "HEAD_GOLDEN=$list"
    HEAD_GOLDEN="$list" make test 2>&1 | tail -40
}

stage_smoke() {
    gdal_env
    cd "$OLMO"
    rm -rf runs/smoke; mkdir -p runs
    export CUDA_VISIBLE_DEVICES="${GPU:-0}"
    ./olmo_cl train \
        dataset="$DATASET" \
        weights="$WDIR/weights.pth" \
        out=runs/smoke epochs=5 max_train=3 max_val=2 augment=0 device="${OCL_DEVICE:-gpu}" \
        2>&1 | tee "$HOME/smoke.log"
}

stage_train() {
    gdal_env
    cd "$OLMO"
    # PLATFORM= (empty): the script's default is "rusticl" (AMD GPUs); on NVIDIA we want any platform.
    CUDA_VISIBLE_DEVICES="${GPU:-0}" PLATFORM= DEVICE="${OCL_DEVICE:-gpu}" \
        bash ALMA_Train_Wheat_Festival_opencl.sh "$SCRATCH" < /dev/null
}

stage_infer() {
    cd "$REPO"
    local TC="$SCRATCH/trainer_checkpoints"
    local ckpt="${CKPT:-}"
    if [ -z "$ckpt" ]; then
        # best checkpoint = highest val F1 in the "Kept ..." lines of the training log
        ckpt=$(sed -n 's/^Kept \(.*\.head\) (val F1 \([0-9.]*\))$/\2 \1/p' "$TC/train.log" \
               | sort -k1,1gr | head -1 | cut -d' ' -f2-)
        ckpt="${ckpt%.head}.ckpt"
    fi
    [ -f "$ckpt" ] || { echo "ERROR: checkpoint not found: $ckpt"; exit 1; }
    echo "checkpoint: $ckpt"

    # the inference script needs mlflow (not in the locked environment)
    if ! .venv/bin/python -c "import mlflow" 2>/dev/null; then
        uv pip install --python .venv/bin/python mlflow
    fi
    .venv/bin/python -c "import rasterio, geopandas, matplotlib, scipy, mlflow, torch; print('deps ok; torch', torch.__version__, torch.cuda.is_available())"

    # point the script at our checkpoint (it has a hard-coded CHECKPOINT= line)
    [ -f ALMA_Inference_Wheat_Festival.sh.orig ] || cp ALMA_Inference_Wheat_Festival.sh ALMA_Inference_Wheat_Festival.sh.orig
    sed -i "s#^CHECKPOINT=.*#CHECKPOINT=\"$ckpt\"#" ALMA_Inference_Wheat_Festival.sh
    grep -n '^CHECKPOINT=' ALMA_Inference_Wheat_Festival.sh

    # leftovers of an earlier inference attempt would be reused: move them aside
    if [ -d "$SCRATCH/dataset_0" ]; then
        mv "$SCRATCH/dataset_0" "$SCRATCH/dataset_0.old_$(date +%H%M%S)"
    fi

    CUDA_VISIBLE_DEVICES="${GPU:-1}" bash ALMA_Inference_Wheat_Festival.sh \
        "${START_DATE:-2024-11-01}" "${END_DATE:-2025-04-30}" "$SCRATCH" < /dev/null
    echo "outputs: $SCRATCH/inference_outputs/  and  $SCRATCH/results/results_raster/"
}

stage_stop() {
    pkill -u "$USER" -f "olmoearth_projects.main|multiprocessing|ALMA_Inference|ALMA_Import" || true
    sleep 3
    pgrep -u "$USER" -af "olmoearth_projects.main|multiprocessing|ALMA_Inference|ALMA_Import" || echo "all stopped"
}

stage="${1:-}"
case "$stage" in
    tools)   stage_tools ;;
    gdal)    stage_gdal ;;
    build)   stage_build ;;
    venv)    stage_venv ;;
    weights) stage_weights ;;
    import)  stage_import ;;
    resume)  stage_resume ;;
    prep)    stage_prep ;;
    golden)  stage_golden ;;
    test)    stage_test ;;
    smoke)   stage_smoke ;;
    train)   stage_train ;;
    infer)   stage_infer ;;
    stop)    stage_stop ;;
    bg)
        sub="${2:?usage: $0 bg <stage>}"
        self="$(readlink -f "$0")"
        nohup "$self" "$sub" > "$HOME/g13_$sub.log" 2>&1 < /dev/null &
        disown
        echo "started '$sub' in the background; follow it with: tail -f ~/g13_$sub.log"
        ;;
    *)
        sed -n '2,45p' "$0"
        exit 1
        ;;
esac
