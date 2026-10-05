# Wheat segmentation on a GPU server: setup, training and inference, step by step

Written 2026-10-04, updated 2026-10-05 (inference script 0.0.3 adds a GeoJSON output, see step 15). This guide repeats, on any similar GPU server, what we did on the Maahr **g13** node: build the OLMO_opencl trainer, import the Sharjah wheat dataset, train the wheat-segmentation head on a frozen OlmoEarth-v1-Tiny encoder, and produce wheat maps.

All the scripts and input files are in the `scripts` folder next to this document. The guide refers to them by path (`scripts/...`) instead of pasting them, so the steps stay short.

---

## 0. What you get, what you need, how long it takes

**The result:** a trained checkpoint (`.ckpt`) and six PNG maps plus a CSV table with one row per wheat field (active area, credit score, insurable area). Since inference script 0.0.3 the same table is also written as a GeoJSON with the field polygons, so GIS tools can draw every theme without a join. On g13 the validation F1 score came out at **0.77**.

**What you need**
- A Linux GPU server you can log in to with SSH. No `sudo` is needed. The scripts install everything inside your home folder.
- An NVIDIA (or other OpenCL-capable) GPU with a working OpenCL driver. The check is `ls /etc/OpenCL/vendors/` and `nvidia-smi`.
- Outbound internet from the server to GitHub, conda-forge, Hugging Face and Microsoft Planetary Computer (where the Sentinel-2 imagery comes from).
- About 15 GB of free disk in your home folder (most of it is the Python environment).
- Your laptop with WSL (Ubuntu), if the server is behind a VPN (step 1).

**Rough times on g13** (4x A100, 48 cores): GDAL install a few minutes, Python environment 10 to 20 minutes, imagery download 20 to 60 minutes, training about **6 minutes**, inference 5 to 15 minutes.

**What is in this folder**

| Path | What it is |
|---|---|
| `G13_GPU_Setup_Guide.md` | this guide |
| `Maahr_VPN_in_WSL.md` | the full VPN guide (connect, use, disconnect, troubleshooting) |
| `scripts/g13_setup.sh` | the one script that runs every server-side stage |
| `scripts/maahr-vpn.sh` | connects your WSL to the VPN |
| `scripts/maahr.conf.example` | the VPN settings file (no password in it) |
| `scripts/inputs/olmo_inputs.tgz` | the bundle you send to the server (OLMO_opencl source + the ALMA scripts) |
| `scripts/inputs/Sharjah_Wheat_fields_4326.geojson` | the 37 wheat field polygons |
| `scripts/inputs/OLMO_opencl/` | readable copy of the trainer source (same content as the bundle) |
| `scripts/inputs/Olmoearth-retraining-for-wheat-field/` | readable copy of the import, training and inference scripts from Dr. Yann |
| `scripts/inputs/make_inputs_bundle.sh` | rebuilds `olmo_inputs.tgz` from the two folders above |
| `scripts/reference_run/` | small text-only copy of three log files of our g13 run (`run_config.json`, `run_log.jsonl`, `train.log`); the same files are also inside `g13_run/` |
| `g13_run/trainer_checkpoints/` | **the results of our training run on g13**: the checkpoints `epoch=12-step=754`, `epoch=13-step=812` (best), `epoch=14-step=870` and `last` (each as `.ckpt`, the Lightning file the inference script loads, and `.head`, the trainer's own small file), plus `run_config.json`, `run_log.jsonl` and `train.log` |
| `g13_inference/` | **the results of our first inference run on g13** (2026-10-04, script 0.0.2): the six PNG maps `01_...` to `06_...`, `field_stats.csv` and the QGIS raster `result_epsg32640_0.tif` (no GeoJSON yet) |
| `g13_inference_20261005/` | **the results of our second inference run** (2026-10-05, script 0.0.3): the same eight files, byte for byte, plus the new `field_stats.geojson` |
| `scripts/patches/` | Dr. Yann's patch that adds the GeoJSON (`ALMA_inference_geojson_2026-10-04.diff`) and his changelog for it |

---

## 1. Connect to the server (only if it sits behind a VPN)

**What this does:** the Maahr cluster is only reachable through a VPN. We run the VPN inside WSL (not with FortiClient on Windows), so Windows keeps its normal network and only WSL talks to the cluster.

Do this once in WSL: install the client, create the settings file from `scripts/maahr.conf.example`, and install the connect script `scripts/maahr-vpn.sh`. The full instructions, with the exact commands, are in `Maahr_VPN_in_WSL.md` (sections 1 and 2).

To connect every time, open a WSL terminal and run:

```bash
~/maahr-vpn.sh
```

It asks for your WSL (sudo) password, then the VPN password. Wait for `Tunnel is up and running`, then **leave that terminal open**. Do all other work in a second terminal.

The script reads the gateway certificate fresh at every connection and asks you to confirm when it has changed. The gateway renews its certificate often, so a pinned certificate stops working.

If your server is not behind a VPN, skip this step.

---

## 2. Look at the server before you start

**What this does:** confirms the server has what the scripts expect, so you find problems before the long steps.

Log in (on g13 the address is `10.172.0.233`, and the admin asked us to work on that node directly):

```bash
ssh almauser1@10.172.0.233
```

Then run these checks:

```bash
nvidia-smi                               # GPUs and driver; note which GPUs are idle
ls /etc/OpenCL/vendors/ ; ls /lib64/libOpenCL.so   # the OpenCL driver the trainer will use
gcc --version | head -1                  # a C compiler is needed (gcc 8 was fine)
python3 --version                        # the system Python can be old; it is not used for the pipeline
nproc; free -g | head -2; df -h ~ | tail -1
curl -sI https://planetarycomputer.microsoft.com | head -1   # internet for imagery
curl -sI https://huggingface.co | head -1
```

You should see the GPUs listed with little memory in use, an `nvidia.icd` file, a compiler, and `HTTP/2 200` (or similar) from both sites. Without GPU OpenCL files or without internet to those sites the later steps cannot work, so stop and ask the administrators.

Useful to know about g13: it has no `sudo`, no `git`, no `tmux`, no `bzip2`, and the job scheduler (Slurm) does not answer. That is why the script downloads the code as archives, installs the tools inside the home folder, and starts long jobs with `nohup` so they keep running after you disconnect.

---

## 3. Send the inputs and the setup script to the server

**What this does:** copies three things to your home folder on the server: the bundle with the trainer source and the ALMA scripts, the field polygons, and the setup script.

Run this in WSL on your laptop (with the VPN up). The folder is wherever you keep this guide:

```bash
G=/mnt/d/work/alma/olmo-opencl-a100-wheat/scripts
scp $G/inputs/olmo_inputs.tgz $G/inputs/Sharjah_Wheat_fields_4326.geojson $G/g13_setup.sh almauser1@10.172.0.233:~/
```

Then on the server:

```bash
chmod +x ~/g13_setup.sh
~/g13_setup.sh            # prints the list of stages
```

From here on, every command is typed **on the server**, in the home folder, and runs one stage of `scripts/g13_setup.sh`. Long stages run with the `bg` word in front, which starts them in the background and writes a log file `~/g13_<stage>.log`. Follow a log with `tail -f ~/g13_<stage>.log` (Ctrl+C stops the viewing, not the job).

---

## 4. Stage `tools`: folders, unpacked inputs, uv, headers, code

```bash
~/g13_setup.sh tools
```

**What this does, in plain English:**
- creates the working folders (`~/dev`, `~/RSDATA`, `~/GISDATA`, `~/models`, `~/opt`);
- unpacks the bundle and puts the trainer into `~/dev/OLMO_opencl`;
- downloads the OpenCL header files (the server has the driver but not the headers) into `~/opt/opencl`;
- installs `uv`, a tool that builds Python environments, into `~/.local/bin` without editing your shell files;
- downloads the `olmoearth_projects` code (pinned to the commit `d16c279`) as an archive and copies in Dr. Yann's `ALMA_Import_Data.sh` and `ALMA_Inference_Wheat_Festival.sh`;
- fixes Windows line endings and file permissions in the copied files (files made on Windows break bash scripts otherwise).

**You should see:** `CL headers: ... files`, a `uv` version, `remaining CRLF files: none`, and `tools stage done.`

---

## 5. Stage `gdal`: a private GDAL and the OpenCL link file

```bash
~/g13_setup.sh gdal
```

**What this does:** the trainer reads the satellite files with the GDAL library, and compiling it needs GDAL's development files. Without `sudo` we cannot install them system-wide, so the script installs GDAL from conda-forge into a private folder (`~/opt/gdalenv`) using `micromamba`, a single downloadable program. It also writes `~/opt/pc/OpenCL.pc`, a small file that tells the build where the GPU driver library and headers are, and that names conda's own `libgcc_s.so.1`. (That last detail matters: GDAL from conda needs a newer compiler runtime than the server's old gcc 8 provides.)

**You should see:** the path of `gdal.h`, and a line starting `pkg-config says:` that lists `-lgdal` and `-lOpenCL` flags.

---

## 6. Stage `build`: compile the trainer and test it on the GPU

```bash
~/g13_setup.sh build
```

**What this does:** compiles `olmo_cl` (the C + OpenCL trainer) and runs its self-tests. The tests compare each GPU kernel against a plain C loop, first through `make selftest`, then again at full size on the GPU (`GPU=<index>` chooses the card; the default is 0).

**You should see:** an `olmo_cl` file, `ldd` lines showing `libgdal` and `libOpenCL`, no `not found`, and `All self-test checks passed`, with errors around 1e-6. The line `OpenCL device: NVIDIA A100-SXM4-80GB` appears in the GPU test.

---

## 7. Stage `venv`: the Python environment (long download)

```bash
~/g13_setup.sh bg venv
tail -f ~/g13_venv.log
```

**What this does:** builds the Python 3.12 environment for `olmoearth_projects` with `uv sync --locked`. It installs PyTorch with CUDA support, `rslearn`, Lightning and the OlmoEarth libraries, several gigabytes in total. "Locked" means every package version is fixed by the project's lock file, so the result matches ours.

**You should see**, at the end: `imports OK, torch 2.7.1+cu126 cuda available: True`. If CUDA says `False`, do not continue; the driver or the PyTorch build does not match.

---

## 8. Stage `weights`: the pretrained encoder

```bash
~/g13_setup.sh weights
```

**What this does:** downloads `weights.pth` (the OlmoEarth-v1-Tiny encoder) and `config.json` from Hugging Face into `~/models/olmoearth/OlmoEarth-v1-Tiny/`. The encoder stays frozen during training; only a small head on top of it is trained.

**You should see:** `weights.pth: 57268875 bytes (expected 57268875)`.

---

## 9. Stage `import`: build the dataset from satellite imagery

```bash
~/g13_setup.sh bg import
tail -f ~/g13_import.log
```

**What this does, in plain English:**
1. Patches `ALMA_Import_Data.sh` so that it uses the small Tiny encoder (192 features instead of 768), batch size 1, 64-pixel crops, and keeps the encoder frozen. The original is kept as `ALMA_Import_Data.sh.orig`.
2. Raises the Planetary Computer request timeout from 10 to 120 seconds (inside the Python environment), because slow answers otherwise abort the download.
3. Runs `ALMA_Import_Data.sh`, which turns the 37 wheat-field polygons into **74 windows** (each field in three winters, November to April), draws a label raster for each, and downloads six Sentinel-2 snapshots per window from Planetary Computer.

The dataset ends up in `~/RSDATA/alma_wheat_festival/dataset`.

**Check the result** (each window should have six finished imagery layers, so 74 x 6 = 444):

```bash
ls ~/RSDATA/alma_wheat_festival/dataset/windows/random_split | wc -l                     # expect 74
find ~/RSDATA/alma_wheat_festival/dataset/windows -path '*layers/sentinel2*' -name completed | wc -l   # expect 444
```

**If the download stops or some layers are missing** (network hiccup, timeouts), re-run only the download step. It skips everything already finished:

```bash
~/g13_setup.sh stop           # first make sure no earlier job is still running
~/g13_setup.sh bg resume
```

Wait until the count above reaches 444 and the job has ended. Never run two import jobs at the same time on the same folder.

---

## 10. Stage `prep`: set train/val/test tags and one label name

```bash
~/g13_setup.sh prep
```

**What this does:** two small repairs, both needed before training.
- **Split tags.** Each window carries a tag saying whether it belongs to training, validation or test. The importer can leave all windows as `train`. The list of the 16 fixed validation and test windows ships with the trainer (`scripts/inputs/OLMO_opencl/data/val_test_crops.tsv`), so the script tags exactly those windows and every other window as `train`.
- **Label band name.** The generated files disagree on what the label raster's band is called (`category` or `label`). The script makes it `label` in the dataset config, the project's `dataset.json`, and `model.yaml`.

**You should see:** the counts before, then `after: {'train': 58, 'val': 8, 'test': 8}`, then lines showing `bands: ["label"]`. If it prints `ERROR: windows in the crops file that are not in the dataset`, your dataset has different window names than ours; stop and check that the import used the same polygon file (`scripts/inputs/Sharjah_Wheat_fields_4326.geojson`).

---

## 11. Stage `golden`: reference numbers from PyTorch

```bash
~/g13_setup.sh bg golden
tail -f ~/g13_golden.log
```

**What this does:** runs the original PyTorch model on a few fixed crops (two validation, two test, plus one training-mode sample) and saves its intermediate results into `~/dev/OLMO_opencl/golden`. These are the answer key against which the OpenCL code is checked in the next step. It also refreshes `data/val_test_crops.tsv`.

**You should see:** `crops.tsv: 16 fixed val/test crops`, five lines such as `val0: task_... (val) offset (24,26) loss 0.79`, and the list `crops.tsv test0 test1 train_mode val0 val1`.

---

## 12. Stage `test`: check the OpenCL head against PyTorch

```bash
~/g13_setup.sh test
```

**What this does:** compares the OpenCL training head with the PyTorch reference (`make test`): predicted probabilities, loss, and gradients on every golden dump, then the optimizer (AdamW) and the learning-rate schedule against PyTorch's.

**You should see:** every line ending in `ok`, with relative errors around 1e-7 against a tolerance of 1e-4, then `AdamW max |param diff| after 300 steps: ~3e-08`. If any line says anything other than `ok`, do not train.

---

## 13. Stage `smoke`: a one-minute training check

```bash
~/g13_setup.sh smoke
```

**What this does:** trains for 5 epochs on 3 windows and validates on 2, just to prove the whole pipeline (dataset reading, GPU training, checkpoint writing) runs on this GPU.

**You should see:** the training and validation losses decrease slightly from epoch to epoch (we saw 0.488 to 0.462 and 0.720 to 0.703), and lines `Kept runs/smoke/epoch=...head`. The F1 stays at 0.0000, which is normal after 15 steps.

---

## 14. Stage `train`: the full training

```bash
GPU=0 ~/g13_setup.sh bg train
tail -f ~/g13_train.log
```

**What this does:** runs `ALMA_Train_Wheat_Festival_opencl.sh` (in the trainer folder) on the chosen GPU. It checks the GPU again, prints the dataset summary, then trains the head for 80 epochs on the 58 training windows, scoring the 8 validation windows after every epoch. It keeps the three best epochs by validation F1 plus the last one, and converts them to Lightning `.ckpt` files that the inference pipeline can load.

On the NVIDIA server the script is run with an empty platform setting. (Its default is `rusticl`, which is for AMD cards.) The `train` stage already does this.

**You should see:** the dataset summary `windows=74 train=58 val=8 test=8 other=0 timesteps=6`, then one line per epoch, about **4 seconds each**, with the validation F1 climbing (0.52 after the first epoch, about 0.77 at the best). The whole run takes about 6 minutes. At the end:

```
--- Training complete ---
Best checkpoint : .../trainer_checkpoints/epoch=13-step=812.ckpt
```

Everything is written to `~/RSDATA/alma_wheat_festival/trainer_checkpoints/`: the checkpoints, `run_log.jsonl` (one line per epoch), `run_config.json` (including the **timestep order**, which must match at inference time) and `train.log`. Our own results from this step are in the folder `g13_run/trainer_checkpoints/` of this guide: the four checkpoints and the three text files, so you can compare your run with ours (a text-only copy of the three log files is also in `scripts/reference_run/`). Your numbers will differ a little, because the timestep order follows the server's directory order.

A second run on the same folder is refused unless you set `OVERWRITE=1` in front of the command.

---

## 15. Stage `infer`: wheat maps and the field table

### 15a. First apply Dr. Yann's 0.0.3 patch (adds `field_stats.geojson`)

The inference script in `olmo_inputs.tgz` is version 0.0.2 and writes no GeoJSON. Dr. Yann's patch (`scripts/patches/ALMA_inference_geojson_2026-10-04.diff`, his notes are in `scripts/patches/CHANGELOG_ALMA_Inference_0.0.3_from_Dr_Yann.md`) changes the script to 0.0.3: it adds a block that writes `field_stats.geojson` next to the CSV. The CSV, the six PNGs and the inference itself do not change.

**g13 has no `patch` command**, so patch the script in WSL on your laptop (VPN up) and copy it back. Run this after the `tools` stage, and before `infer`:

```bash
mkdir -p ~/g13patch && cd ~/g13patch
scp <user>@<server>:~/dev/olmoearth_projects/ALMA_Inference_Wheat_Festival.sh .
cp ALMA_Inference_Wheat_Festival.sh before.sh

# keep only the .sh part of the diff (its first 39 lines; the .md part starts at line 40)
head -39 <path-to-this-repo>/scripts/patches/ALMA_inference_geojson_2026-10-04.diff > sh.diff

patch --dry-run ALMA_Inference_Wheat_Festival.sh < sh.diff     # all hunks must say "succeeded"
patch ALMA_Inference_Wheat_Festival.sh < sh.diff
bash -n ALMA_Inference_Wheat_Festival.sh && echo "syntax ok"
grep -n 'ALMA_INFER_VERSION=\|field_stats.geojson' ALMA_Inference_Wheat_Festival.sh
diff before.sh ALMA_Inference_Wheat_Festival.sh

scp ALMA_Inference_Wheat_Festival.sh <user>@<server>:~/dev/olmoearth_projects/
ssh <user>@<server> 'chmod 755 ~/dev/olmoearth_projects/ALMA_Inference_Wheat_Festival.sh'
```

You should see version `0.0.3` and an added block of about eight lines in the diff. The `.md` part of Dr. Yann's diff is documentation for a file that is not in the repository on the server, so it is skipped on purpose. The patch does not touch the hard-coded `CHECKPOINT=` line, so it also applies after the `infer` stage has set the checkpoint. `~/g13_setup.sh infer` copies the script only if it is missing (`cp -n`), so it will not overwrite your patched version.

### 15b. Run the stage

```bash
GPU=1 ~/g13_setup.sh bg infer
tail -f ~/g13_infer.log
```

**What this does:**
- picks the best checkpoint from the training log (or use `CKPT=/path/file.ckpt`);
- installs `mlflow` into the environment, because the inference script needs it and the locked environment does not include it;
- points `ALMA_Inference_Wheat_Festival.sh` at that checkpoint (the path is hard-coded in the script; the original is kept as `.orig`);
- runs it for the six-month window **2024-11-01 to 2025-04-30**. That is the same window the training windows use. The script's own default (January to March 2025) is too short: it finds only two usable Sentinel-2 scenes per area, and the model needs six timesteps. Change the dates with `START_DATE=... END_DATE=...`.

The script downloads the imagery for the whole Meilha area, runs the model and draws the outputs. It takes about 5 to 15 minutes.

**You should see**, in the log, `Materializing 6 item groups` for each prediction window and, at the end, a "Next steps" block listing the output files.

**Outputs**

| File | Location | What it shows |
|---|---|---|
| `01_wheat_probability.png` | `~/RSDATA/alma_wheat_festival/inference_outputs/<date>/` | wheat probability for every pixel |
| `02_wheat_mask.png` | same | wheat / not wheat with field boundaries |
| `03_field_activation.png` | same | each of the 37 fields as active, partial or inactive |
| `04_uncertainty.png` | same | pixels where the model is unsure (probability 0.30 to 0.70) |
| `05_credit_score.png` | same | micro-credit score per field, 0 to 100 |
| `06_insurance_exposure.png` | same | insurable active area per field, in hectares |
| `field_stats.csv` | same | one row per field, for Excel or a dashboard |
| `field_stats.geojson` | same | the same table as properties of the 37 field polygons (script 0.0.3 or later), see 15c |
| `result_epsg32640_0.tif` | `~/RSDATA/alma_wheat_festival/results/results_raster/` | the prediction raster for QGIS (UTM zone 40N) |

The files from our own g13 runs are in the folders `g13_inference/` (first run, eight files) and `g13_inference_20261005/` (second run with script 0.0.3, nine files including the GeoJSON). Open them to see what a good run looks like before you start your own. The second run reproduced the first one exactly: the six PNGs, the CSV and the raster have identical checksums, so the patch changed only the new GeoJSON.

On 2026-10-05 the end of the log read: season 2024-11-01 to 2025-04-30, total active wheat **1179.4 ha**, **34 of 37** fields active, mean credit score **84 / 100**, no fields flagged.

### 15c. What is in `field_stats.geojson`

One feature per field (37), polygons in EPSG:4326 (stored as longitude/latitude, `CRS84`, as MultiPolygons), with the same figures as the CSV as properties. We compared the file with the CSV: all nine properties match exactly. Statuses in the 2026-10-05 run: 34 `active`, 2 `partial`, 1 `inactive`.

| Property | Theme / map |
|---|---|
| `field_id` | 1-based row order of `Sharjah_Wheat_fields_4326.geojson` |
| `diam_km` | field diameter, copied from the boundary file |
| `status` | 03 field activation (`active` / `partial` / `inactive`) |
| `active_frac`, `mean_prob` | 03 field activation, 05 credit score |
| `uncertainty_frac` | 04 uncertainty (share of pixels at 0.30 to 0.70) |
| `area_ha`, `active_area_ha` | 06 insurance exposure |
| `credit_score` | 05 credit score (0 to 100) |

So one GeoJSON file carries the data of four maps (03 to 06); you get each map by colouring the polygons by a different property. Maps 01 (wheat probability) and 02 (wheat mask) are drawn from the probability raster `result_epsg32640_0.tif`, not from the GeoJSON.

**Caution:** `field_id` is the position of a field in `Sharjah_Wheat_fields_4326.geojson`, not an ID stored in that file. If the boundary file is reordered or edited, the IDs of a new run no longer match earlier runs. The GeoJSON carries its own polygons, so it is not affected; joins of the CSV to the boundary file are.

### 15d. If `g13_setup.sh` only prints its help text instead of running `infer`

The copy of `g13_setup.sh` on the server may be older than `scripts/g13_setup.sh` in this repository and have no `infer` stage; it then prints its usage text and stops, and `~/g13_infer.log` contains only that text. Either send the current `scripts/g13_setup.sh` again, or run the inference script directly (this is what the stage does; the checkpoint path is already set in the script):

```bash
cd ~/dev/olmoearth_projects
export PATH="$HOME/.local/bin:$PATH" WANDB_MODE=disabled
SCRATCH=$HOME/RSDATA/alma_wheat_festival
ls -lh "$SCRATCH/trainer_checkpoints/"*.ckpt                      # the CHECKPOINT= line in the script must point at one of these
.venv/bin/python -c "import mlflow" 2>/dev/null || uv pip install --python .venv/bin/python mlflow
[ -d "$SCRATCH/dataset_0" ] && mv "$SCRATCH/dataset_0" "$SCRATCH/dataset_0.old_$(date +%H%M%S)"
CUDA_VISIBLE_DEVICES=1 nohup bash ALMA_Inference_Wheat_Festival.sh 2024-11-01 2025-04-30 "$SCRATCH" \
  < /dev/null > ~/g13_infer.log 2>&1 &
tail -f ~/g13_infer.log
```

On g13 this ran in a few minutes, because the imagery was already cached (`computed 0 ingest jobs`). The log starts with `ALMA_Inference_Wheat_Festival.sh v0.0.3` when the patch is in place, and near the end shows `Wrote field_stats.csv` followed by `Wrote field_stats.geojson`.

---

## 16. Copy the results to your laptop

Run this in WSL on the laptop (VPN up). It puts your run's files in `Downloads\g13_inference` and `Downloads\g13_run`. (Our run's copies of these two folders already sit inside this guide folder, as `g13_inference/` and `g13_run/`; use different names or folders for your own run so you do not mix them up.)

```bash
D=/mnt/c/Users/Yousuf/Downloads/g13_inference
mkdir -p $D
scp "almauser1@10.172.0.233:RSDATA/alma_wheat_festival/inference_outputs/*/*" $D/
scp almauser1@10.172.0.233:RSDATA/alma_wheat_festival/results/results_raster/result_epsg32640_0.tif $D/
mkdir -p /mnt/c/Users/Yousuf/Downloads/g13_run
scp -r almauser1@10.172.0.233:RSDATA/alma_wheat_festival/trainer_checkpoints /mnt/c/Users/Yousuf/Downloads/g13_run/
```

Open `result_epsg32640_0.tif` in QGIS and put `scripts/inputs/Sharjah_Wheat_fields_4326.geojson` on top to compare the field boundaries with the prediction. Open `field_stats.csv` in Excel. When you are done, disconnect the VPN with Ctrl+C in the VPN terminal.

Keep the checkpoints and `run_config.json` together with the results. The `.ckpt` file loads with the unchanged `ALMA_Inference_Wheat_Festival.sh`.

### 16b. Show the six maps in QGIS

1. Drag `field_stats.geojson` into QGIS (or Layer > Add Layer > Add Vector Layer). Check the attribute table: 37 rows and nine properties.
2. Add a basemap (XYZ Tiles) and, for maps 01 and 02, the raster `result_epsg32640_0.tif` (UTM 40N; QGIS reprojects it on the fly). Keep the field layer above the raster.
3. Colour the field layer under Layer Properties > Symbology, one copy of the layer per map (right-click > Duplicate Layer):

| Map | Property | Style |
|---|---|---|
| 03 Field activation | `status` | Categorized: active green, partial amber, inactive red |
| 04 Uncertainty | `uncertainty_frac` | Graduated, sequential ramp (yellow to dark red) |
| 05 Credit score | `credit_score` | Graduated, fixed range 0 to 100, five bands, red to green |
| 06 Insurance exposure | `active_area_ha` | Graduated, sequential ramp, hectares |

4. Maps 01 and 02 come from the raster (one band, values 0.0 to 1.0, the wheat probability). Style 01 as Singleband pseudocolor from 0 to 1 in the tan-to-green wheat colours, and 02 as two classes split at a probability cutoff. We used 0.5 for the cutoff, which is an assumption: compare with `02_wheat_mask.png` and the inference script before relying on it.
5. Untick the field layers above a raster layer when you want to see it (the polygons are mostly opaque), and tick them again for maps 03 to 06.
6. For a printable map use Project > New Print Layout and export an image or PDF. The finished PNGs from the script are already in the results folders.

Dr. Yann's own map viewer (sharjah.almamaps.ai) can use the same GeoJSON for the fields of Mleiha.

---

## 17. How good is the result, and what to watch

- **Validation F1 about 0.77** with the best epoch at 12 to 14. The score stays flat after that, so more epochs with the same setup do not help.
- **Over-prediction of wheat.** Recall is high (about 0.95) but precision is lower (about 0.65): the model marks too many pixels as wheat. Treat the maps as a first result.
- **Small validation set.** Eight validation windows make the score noisy. The test split (8 windows) was not scored by a separate command here; it is only used as fixed crops for the golden comparison.
- **The inference run is reproducible.** Running the inference twice (2026-10-04 with script 0.0.2, 2026-10-05 with 0.0.3) gave identical PNGs, CSV and raster. Only the GeoJSON is new.
- **The training run is fast and cheap.** Because the encoder is frozen, its features are computed once and the head trains in seconds per epoch. Experiments (other seeds, learning rates, `augment=1`) are practical; use `OLMO_CL_EXTRA="..."` on the training script or edit the `train` stage.

---

## 18. Quick fixes for problems we know about

| What you see | What to do |
|---|---|
| Import stops with timeouts to the Planetary Computer, or the layer count stays below 444 | `~/g13_setup.sh stop`, then `~/g13_setup.sh bg resume`, repeat until 444. |
| The import seems to hang and `pkill -f ALMA_Import` finds nothing | The worker processes carry other names. `~/g13_setup.sh stop` covers them all. |
| `Window rejected: found 2 matches (required: 3)` during inference | The date window is too short. Use the six-month window `2024-11-01 2025-04-30` (the default of the `infer` stage). |
| `make golden` fails with `could not get all the needed bands ... layer label` | Run the `prep` stage again; the label band name is not `label` everywhere. |
| `make golden` fails with `KeyError: 'val0'` or `0 fixed val/test crops` | The windows have no val/test tags. Run the `prep` stage. |
| `olmo_cl train` says `Cannot create runs/smoke` | It creates only the last folder. `mkdir -p ~/dev/OLMO_opencl/runs` first (the `smoke` stage does this). |
| `olmo_cl` says `Unable to open checkpoint ...weights.bin` | The file is called `weights.pth`. |
| Link error mentioning `GCC_12.0.0` symbols | The `gdal` stage was not run, or `OpenCL.pc` lacks the `libgcc_s.so.1` entry. Re-run `~/g13_setup.sh gdal` and `build`. |
| The training script waits at "Continue anyway with DEVICE=auto" | Its OpenCL self-test failed on the chosen device. Fix the device first (run `~/g13_setup.sh build` and read the error). |
| `ModuleNotFoundError: mlflow` in inference | The `infer` stage installs it; run it through the script. |
| `patch: command not found` on the server | The server has no `patch`. Patch the script in WSL and copy it back (step 15a). |
| `~/g13_infer.log` contains only the script's help text | The `g13_setup.sh` on the server has no `infer` stage. Send the current one, or run the inference script directly (step 15d). |
| Only eight output files, no `field_stats.geojson` | The inference script is still version 0.0.2. Apply the patch (step 15a); the log's first line must show `v0.0.3`. |

---

## 19. Moving this to a different server

The stages are written for a server **without sudo**. If your new server is different:

- **With sudo / system packages:** you can replace stages `gdal` (and parts of `tools`) with the system packages `libgdal-dev`, `ocl-icd-opencl-dev`, `opencl-headers`, `pkg-config`, then set the `build` stage to use them. See `WSL_OLMO_Setup_and_Cleanup.md` in the project for the apt-based route.
- **A different GPU vendor:** the OpenCL platform name changes. For AMD with Mesa, set `RUSTICL_ENABLE=radeonsi` and `PLATFORM=rusticl` (the training script's own defaults). For NVIDIA leave the platform empty, as the `train` stage does.
- **A working Slurm:** run the `train` and `infer` stages inside an `sbatch` job (`--ntasks=1 --gres=gpu:1`) instead of the `bg` wrapper.
- **Other GPU numbers:** pick idle cards with `nvidia-smi` and set `GPU=<index>` (the script's defaults are 0 for training and 1 for inference).
- **Other paths:** the folder layout under `~` is set at the top of `scripts/g13_setup.sh` (`DEVDIR`, `SCRATCH`, and so on).
