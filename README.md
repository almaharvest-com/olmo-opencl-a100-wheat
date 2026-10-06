# olmo-opencl-a100-wheat: wheat segmentation with OlmoEarth on an NVIDIA A100 server

A tested, step-by-step recipe for training and running a **wheat-field segmentation model** on a shared GPU server where you have no `sudo`. It was built and run on the Maahr **g13** node (4x NVIDIA A100) and can be repeated on a similar server.

The model is a small trainable head on top of a frozen pretrained **OlmoEarth-v1-Tiny** encoder (Ai2's open Earth-observation foundation model). It is trained on Sentinel-2 imagery of 37 wheat fields in Sharjah (Mleiha) and produces wheat maps and a per-field table for crop insurance, traceability and micro-credit use.

## Result of the reference run on g13

| | |
|---|---|
| Dataset | 74 windows (58 train / 8 val / 8 test), 6 Sentinel-2 snapshots each, Nov to Apr |
| Training | 80 epochs, about 4 s per epoch, about 6 minutes in total on one A100 |
| Best checkpoint | `epoch=13-step=812.ckpt`, validation F1 **0.7748** |
| Behaviour | high recall (about 0.95), lower precision (about 0.65): the model over-predicts wheat |
| Outputs | six PNG maps, `field_stats.csv`, `field_stats.geojson` (0.0.3), wheat-probability GeoTIFF, `threshold_sweep.csv` and run manifest (0.0.4) |

Treat the numbers as a first result: the validation set is small, and the test split was not scored separately.

### Update 2026-10-05: GeoJSON output (inference script 0.0.3)

Dr. Yann's patch (`scripts/patches/`) makes `ALMA_Inference_Wheat_Festival.sh` also write `field_stats.geojson`: one feature per field (37), with the field polygon in EPSG:4326 and the same nine per-field figures as the CSV as properties. GIS tools and map dashboards can then draw maps 03 to 06 directly (`status`, `uncertainty_frac`, `credit_score`, `active_area_ha`) without joining the CSV to the boundary file; maps 01 and 02 still come from the probability raster. We applied the patch on g13 and re-ran the inference: 1179.4 ha active wheat, 34 of 37 fields active, mean credit score 84. The PNGs, the CSV and the raster are identical to the first run; only the GeoJSON is new. The results are in `g13_inference_20261005/`. See steps 15a to 15d and 16b of the guide (the server has no `patch` command, so the script is patched in WSL and copied back).

### Update 2026-10-06: real probabilities (inference script 0.0.4)

Up to 0.0.3 the raster held 0/1 class labels, so `mean_prob` equalled `active_frac` and `uncertainty_frac` was always 0. The fix (`scripts/patches/2026-10-06_probability_output/`) needs no retraining: a small task class, `alma_tasks.py`, makes the model write the wheat probability (the g13 rslearn is too old for the built-in `output_probs` option), and script 0.0.4 adds a cut-off parameter (`THRESH`), per-field probability histograms for a threshold slider, `threshold_sweep.csv` and a run manifest with checksums and the imagery used. At cut-off 0.50 the result equals the earlier runs (1179.4 ha, 34 of 37 active), but the probabilities are squeezed around 0.5 (maximum 0.682), so all 37 fields fall in the 0.30–0.70 uncertainty band and the mean credit score is 56. The cut-off and the uncertainty band must be chosen, or the model calibrated, before these figures are published. Results in `g13_inference_20261006/`; details and commands in section 15e of the guide.

## Start here

Read **[G13_GPU_Setup_Guide.md](G13_GPU_Setup_Guide.md)**. It explains every step in plain English, with the commands, what you should see, and fixes for the problems we met.

The short version (all stages run on the server through one script):

```bash
# on your laptop (WSL), with the VPN up: send the inputs and the script
G=scripts
scp $G/inputs/olmo_inputs.tgz $G/inputs/Sharjah_Wheat_fields_4326.geojson $G/g13_setup.sh <user>@<server>:~/

# on the server
chmod +x ~/g13_setup.sh
~/g13_setup.sh tools && ~/g13_setup.sh gdal && ~/g13_setup.sh build
~/g13_setup.sh bg venv        # long download
~/g13_setup.sh weights
~/g13_setup.sh bg import      # Sentinel-2 download
~/g13_setup.sh prep           # train/val/test tags + label band name
~/g13_setup.sh bg golden && ~/g13_setup.sh test && ~/g13_setup.sh smoke
GPU=0 ~/g13_setup.sh bg train
GPU=1 ~/g13_setup.sh bg infer
```

`~/g13_setup.sh` with no argument lists all stages.

## What is in this repository

| Path | Contents |
|---|---|
| `G13_GPU_Setup_Guide.md` | the full step-by-step guide |
| `Maahr_VPN_in_WSL.md` | how to reach the Maahr cluster through a VPN from WSL |
| `scripts/g13_setup.sh` | the one script that runs every server-side stage |
| `scripts/maahr-vpn.sh`, `scripts/maahr.conf.example` | VPN connect script and settings template (no password stored) |
| `scripts/inputs/` | `olmo_inputs.tgz` (the bundle sent to the server), readable copies of `OLMO_opencl/` and `Olmoearth-retraining-for-wheat-field/`, the field polygons `Sharjah_Wheat_fields_4326.geojson`, and `make_inputs_bundle.sh` |
| `scripts/reference_run/` | the three small log files of the reference run |
| `g13_run/trainer_checkpoints/` | results of the reference training run: checkpoints (`.ckpt`, `.head`), `run_config.json`, `run_log.jsonl`, `train.log` |
| `g13_inference/` | results of the first inference run (script 0.0.2): six PNG maps, `field_stats.csv`, `result_epsg32640_0.tif` |
| `g13_inference_20261005/` | results of the second inference run (script 0.0.3): the same files plus `field_stats.geojson` |
| `g13_inference_20261006/` | results of the third inference run (script 0.0.4): wheat probabilities, `threshold_sweep.csv`, probability GeoTIFF, `run_manifest.json`, `provenance/` |
| `scripts/patches/` | Dr. Yann's patch adding the GeoJSON output (`.diff`) and his changelog; `2026-10-06_probability_output/` with the probability fix (script 0.0.4, `alma_tasks.py`, `RUN_ON_G13.md`) |

## Requirements

- A Linux GPU server with SSH access, an OpenCL-capable GPU driver, and outbound internet to GitHub, conda-forge, Hugging Face and Microsoft Planetary Computer. No `sudo` needed.
- About 15 GB of free disk in your home folder.
- Your laptop with WSL (Ubuntu) if the server is behind a VPN.

## Credits and licences

- **OlmoEarth** and the `olmoearth_projects` / `rslearn` tooling are from Ai2 (Allen Institute for AI). The scripts download them at setup time; they are not stored here.
- **OLMO_opencl** (the C + OpenCL trainer) and the `ALMA_*.sh` import, training and inference scripts are by Dr. Yann (ALMA). They are copied under `scripts/inputs/` unchanged (the 0.0.3 inference patch is kept separately in `scripts/patches/`; the bundle still holds version 0.0.2). `OLMO_opencl` carries its own `LICENSE` file.
- The setup script, the VPN workflow and the guide were written for the ALMA EARTH Sharjah wheat project.
- No licence has been chosen for the new material in this repository (the setup script, the guide and the VPN files). Add a `LICENSE` file before making the repository public.

## Before you publish this repository

- **Keep it private** unless the points below are resolved. The VPN files and the guide contain the cluster user name and an internal server address (`10.172.0.233`).
- Check that you may share Dr. Yann's files under `scripts/inputs/` and the field polygons.
- The checkpoints in `g13_run/` are about 96 MB in total (24 MB each), which is within GitHub's per-file limit.
- No passwords or keys are stored in this folder.
