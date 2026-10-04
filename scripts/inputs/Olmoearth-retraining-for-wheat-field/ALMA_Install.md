# ALMA Wheat Festival — Server Installation Guide

Fresh-server setup for the `olmoearth_projects` fine-tuning stack.
Tested on Debian 12 / Ubuntu 22.04 LTS with a Quadro P1000 4 GB GPU.

---

## 1. System packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    git curl wget build-essential \
    libgdal-dev gdal-bin python3-gdal \
    libgeos-dev libproj-dev proj-bin \
    libspatialindex-dev \
    pkg-config libhdf5-dev \
    sqlite3 \
    nvidia-cuda-toolkit          # skip if CUDA already installed via NVIDIA runfile
```

### 1a. NVIDIA driver + CUDA (if not already present)

```bash
# Check driver
nvidia-smi
# If missing, install CUDA 12.x via official NVIDIA runfile:
# https://developer.nvidia.com/cuda-downloads
# Minimum: CUDA 12.1, driver >= 525
```

---

## 2. Python 3.12 via deadsnakes PPA

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev python3.12-distutils
```

---

## 3. Install uv (package + environment manager)

`uv` is **not** in apt — `sudo apt install uv` will fail. Install via the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The installer registers `uv` automatically on current versions (0.11+).
If `uv` is not found after installation:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Verify:

```bash
uv --version
```

---

## 4. Clone and install

```bash
mkdir -p $HOME/dev && cd $HOME/dev

git clone https://github.com/allenai/olmoearth_projects.git
cd olmoearth_projects

# Install all dependencies (creates .venv automatically)
uv sync --locked --all-extras --python 3.12

# Install MLflow for local SQLite experiment tracking
uv add mlflow
```

---

## 5. Activate the virtual environment

```bash
source $HOME/dev/olmoearth_projects/.venv/bin/activate
python -c "import rslearn; import olmoearth_projects; import mlflow; print('OK')"
```

Add alias to `~/.bashrc` for convenience:

```bash
echo 'alias alma-env="source $HOME/dev/olmoearth_projects/.venv/bin/activate"' >> ~/.bashrc
```

---

## 6. Disable WandB (replaced by local MLflow SQLite)

The training framework has a hardcoded `wandb.Api()` call after training.
Disable it permanently so it never prompts for an API key:

```bash
source $HOME/dev/olmoearth_projects/.venv/bin/activate
wandb disabled
# Verify: wandb status should show "mode: disabled"
wandb status
```

This writes to `~/.config/wandb/settings` and persists across all runs.
The `WANDB_MODE=disabled` env var alone is **not** sufficient.

---

## 7. Patch Planetary Computer timeout

The default rslearn Sentinel-2 read timeout is 10 s, which causes failures on
slow or rate-limited connections. Patch it to 120 s:

```bash
sed -i 's/timeout: timedelta = timedelta(seconds=10)/timeout: timedelta = timedelta(seconds=120)/' \
    $HOME/dev/olmoearth_projects/.venv/lib/python3.12/site-packages/rslearn/data_sources/planetary_computer.py

# Verify
grep "timedelta(seconds=" \
    $HOME/dev/olmoearth_projects/.venv/lib/python3.12/site-packages/rslearn/data_sources/planetary_computer.py
```

Expected output: `timeout: timedelta = timedelta(seconds=120),`

---

## 8. Planetary Computer access (Sentinel-2 data)

No account is required for public Sentinel-2 data, but a subscription key
improves rate limits significantly:

```bash
# Optional: set your Planetary Computer token
export PC_SDK_SUBSCRIPTION_KEY="your_token_here"
echo 'export PC_SDK_SUBSCRIPTION_KEY="your_token_here"' >> ~/.bashrc
```

---

## 9. Transfer GIS data and project config from the preparation machine

If the dataset was prepared on a separate machine, rsync it over:

```bash
# Dataset (250 MB — Sentinel-2 tiles + label rasters)
rsync -avz --progress \
    ~/RSDATA/alma_wheat_festival/ \
    user@server:~/RSDATA/alma_wheat_festival/

# Project config files (model.yaml, dataset.json, annotations, etc.)
rsync -avz --progress \
    --exclude='.venv/' --exclude='.git/' --exclude='__pycache__/' \
    ~/dev/olmoearth_projects/ \
    user@server:~/dev/olmoearth_projects/
```

After rsync, run `uv sync --locked --all-extras --python 3.12` again on the
server to ensure the venv matches the lockfile (step 4).

---

## 10. Verify GPU

```bash
source $HOME/dev/olmoearth_projects/.venv/bin/activate
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')
"
```

---

## 11. Environment variables

Put in `~/.bashrc` or a `.env` file in the project root:

```bash
# Planetary Computer (optional but recommended)
export PC_SDK_SUBSCRIPTION_KEY=""

# Parallelism — set to nproc/2
export NUM_WORKERS=8

# Memory fragmentation fix for Pascal/Turing GPUs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

WandB vars are **not** needed — WandB is disabled (step 6).
Experiment metrics are logged to:
`~/RSDATA/alma_wheat_festival/alma_training.db` (MLflow SQLite)

---

## 12. Directory layout

```
$HOME/
├── dev/
│   └── olmoearth_projects/
│       ├── ALMA_Import_Data.sh
│       ├── ALMA_Train_Wheat_Festival.sh
│       ├── olmoearth_run_data/
│       │   └── wheat_festival/          # project config (model.yaml etc.)
│       └── .venv/
├── GISDATA/
│   └── Sharjah_WheatFestival/           # source GIS data
│       ├── Sharjah_Wheat_fields_4326.geojson
│       └── ...
└── RSDATA/
    └── alma_wheat_festival/             # scratch — datasets + checkpoints
        ├── dataset/                     # Sentinel-2 tiles + label rasters
        ├── trainer_checkpoints/         # model checkpoints written here
        │   ├── epoch=12-step=754.ckpt   # best checkpoint (val F1 peak)
        │   └── last.ckpt
        └── alma_training.db             # MLflow SQLite metrics log
```

---

## 13. Known constraints for Quadro P1000 (4 GB VRAM)

The following settings in `model.yaml` are required to fit within 4 GB:

| Parameter | Value | Reason |
|---|---|---|
| `model_id` | `OLMOEARTH_V1_TINY` | Base (89 M params) OOMs at 4 GB |
| `in_channels` | `192` | Tiny hidden dim (not 384 or 768) |
| `patch_size` (data) | `64` | 128 px → 6144 tokens → OOM; 64 px → 1536 tokens → fits |
| `batch_size` | `1` | Encoder backward needs full VRAM |
| `unfreeze_at_epoch` | `9999` | Encoder stays frozen; end-to-end backprop OOMs at 4 GB |
| `precision` | `16-mixed` | FP16 halves activation memory |

With these settings, VRAM usage stays under ~2 GB throughout training.

---

## 14. Pre-commit hooks (development only)

```bash
uv tool install pre-commit --with pre-commit-uv --force-reinstall
pre-commit install
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `uv: command not found` | `export PATH="$HOME/.local/bin:$PATH"` |
| `source $HOME/.local/bin/env: No such file` | Newer uv doesn't create `env` file — just export PATH above |
| `GDAL_DATA not found` | `export GDAL_DATA=$(gdal-config --datadir)` |
| `CUDA out of memory` | Check section 13; reduce `patch_size` or `batch_size` further |
| `Read timed out` during data import | Step 7 timeout patch not applied, or re-run `ALMA_Import_Data.sh` |
| `No such file: layers/label/category/geotiff.tif` | Change `bands: ["category"]` → `bands: ["label"]` in `model.yaml` |
| `Shape mismatch 529 != 552` | Remove `load_all_patches: true` from `val_config` in `model.yaml` |
| `wandb: Paste an API key` | Run `wandb disabled` inside the venv (step 6) |
| `ModuleNotFoundError: mlflow` | Run `uv add mlflow` then `uv sync` (step 4) |
| `ModuleNotFoundError: rslearn` | Activate venv: `source .venv/bin/activate` |
