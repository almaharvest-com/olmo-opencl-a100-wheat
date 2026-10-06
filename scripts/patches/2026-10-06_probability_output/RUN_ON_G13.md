# Probability output for MleihaEarth (task A + B, 6 Oct 2026)

**Why:** the model wrote `argmax` class labels (0 or 1) into the raster, so every
field's `mean_prob` equalled `active_frac` and no pixel was ever "uncertain".
The fix is on the model side; the checkpoint is unchanged, **no retraining**.

The rslearn on g13 is too old for the built-in `output_probs` option, so the fix
is a small task class, `alma_tasks.WheatProbSegmentationTask`, that returns the
wheat probability instead of the class id. It works with any rslearn version
(tested with 0.0.5 and checked against 0.1.16).

| File | Goes to (g13) | What it does |
|---|---|---|
| `alma_tasks.py` | `~/dev/olmoearth_projects/` | the probability task |
| `patch_model_yaml_prob_task.py` | `~/dev/olmoearth_projects/incoming_v004/` | points `model.yaml` at that task (and removes `output_probs` lines if present) |
| `ALMA_Inference_Wheat_Festival.sh` | `~/dev/olmoearth_projects/` | v0.0.4 (= Yann's 0.0.3 + the changes below) |
| `ALMA_inference_probability_0.0.4.diff` | — | changes from 0.0.3 to 0.0.4 |

What 0.0.4 changes (for Yann's points):
- cut-off is a parameter: `THRESH` (default 0.50), uncertain band `UNC_LO`/`UNC_HI` (0.30/0.70);
- stops with an error if the raster still holds only 0/1;
- ignores nodata pixels (255) in all field figures;
- `uncertainty_frac` is computed on the field's own valid pixels;
- every field gets `prob_hist` (20 bins of 0.05, share of its pixels) and `threshold`
  → a slider can recompute area and status at any cut-off without the raster;
- new `threshold_sweep.csv` (active ha and field counts at cut-offs 0.10–0.90);
- copies the probability GeoTIFF into the output folder, so all outputs of one run sit together;
- `run_manifest.json` + `provenance/` (checkpoint sha256, model.yaml, alma_tasks.py,
  run_config.json, boundary file sha256, Sentinel-2 items used, sha256 of every output)
  for the long-term consistency insurers need;
- puts its own folder on `PYTHONPATH` so `alma_tasks` is found;
- fixes a latent crash in map 04 (`marker="!"` is not a valid matplotlib marker).

**Checkpoint:** the 5 Oct run used `epoch=13-step=812.ckpt`. Keep that one so the
new figures are comparable with the old ones.

---

## 1. Copy the files to g13 (WSL on the laptop, VPN up)

```bash
P=/mnt/d/work/alma/olmo-opencl-a100-wheat/scripts/patches/2026-10-06_probability_output
ssh almauser1@10.172.0.233 mkdir -p dev/olmoearth_projects/incoming_v004
scp $P/ALMA_Inference_Wheat_Festival.sh $P/alma_tasks.py $P/patch_model_yaml_prob_task.py \
    almauser1@10.172.0.233:dev/olmoearth_projects/incoming_v004/
```

## 2. On g13: install

```bash
cd ~/dev/olmoearth_projects
[ -f ALMA_Inference_Wheat_Festival.sh.v003 ] || cp ALMA_Inference_Wheat_Festival.sh ALMA_Inference_Wheat_Festival.sh.v003
cp incoming_v004/ALMA_Inference_Wheat_Festival.sh incoming_v004/alma_tasks.py .
sed -i "s#^CHECKPOINT=.*#CHECKPOINT=\"$HOME/RSDATA/alma_wheat_festival/trainer_checkpoints/epoch=13-step=812.ckpt\"#" ALMA_Inference_Wheat_Festival.sh
grep -n '^CHECKPOINT=' ALMA_Inference_Wheat_Festival.sh            # epoch=13-step=812
.venv/bin/python incoming_v004/patch_model_yaml_prob_task.py olmoearth_run_data/wheat_festival/model.yaml   # must say "1 time(s)"
PYTHONPATH=$PWD .venv/bin/python -c "import alma_tasks; print('alma_tasks ok')"
```

## 3. On g13: run

```bash
S=~/RSDATA/alma_wheat_festival
# an old 0/1 prediction in dataset_0 would be reused; imagery alone is fine to keep
find $S/dataset_0 -path '*layers/output*' -name '*.tif' 2>/dev/null | head -3   # must print nothing
#   if it prints files:  mv $S/dataset_0 $S/dataset_0.old_$(date +%H%M%S)
[ -d $S/results ] && mv $S/results $S/results.old_$(date +%H%M%S)
cd ~/dev/olmoearth_projects
CUDA_VISIBLE_DEVICES=1 THRESH=0.50 nohup bash ALMA_Inference_Wheat_Festival.sh \
    2024-11-01 2025-04-30 $S < /dev/null > ~/g13_infer_v004.log 2>&1 &
tail -f ~/g13_infer_v004.log          # Ctrl+C stops watching, not the run
```
`Retrying after catching error ... timed out` lines are Sentinel-2 download retries; let them run.
In the log look for `Raster: ... px strictly between 0 and 1` with a large number.

If it fails in post-processing with a message about `classification_fields` / allowed values,
in `olmoearth_run_data/wheat_festival/olmoearth_run.yaml` set
`classification_fields: null` and `regression_fields: [{property_name: wheat_prob, band_index: 1}]`
under `inference_results_config:`, then repeat step 3.

## 4. On g13: check task A's "done when"

```bash
O=$(ls -d ~/RSDATA/alma_wheat_festival/inference_outputs/* | tail -1)
python3 - "$O/field_stats.csv" <<'PY'
import csv, sys
r = list(csv.DictReader(open(sys.argv[1])))
d = sum(1 for x in r if x["active_frac"] != x["mean_prob"])
u = sum(1 for x in r if float(x["uncertainty_frac"]) > 0)
print(f"rows: {len(r)}  active_frac != mean_prob: {d}  uncertainty_frac > 0: {u}")
PY
ls -1 "$O"; cat "$O/threshold_sweep.csv"; tail -12 ~/g13_infer_v004.log
```
Expect most of 37 rows differing and `uncertainty_frac > 0` somewhere; 12 files plus `provenance/`.

## 5. Copy to the laptop and push (WSL)

```bash
D=/mnt/d/work/alma/olmo-opencl-a100-wheat/g13_inference_20261006
mkdir -p $D
scp -r "almauser1@10.172.0.233:RSDATA/alma_wheat_festival/inference_outputs/20261006/*" $D/   # folder name from step 4
cd /mnt/d/work/alma/olmo-opencl-a100-wheat && git add g13_inference_20261006 scripts/patches/2026-10-06_probability_output
git commit -m "Inference 0.0.4: wheat probability output, threshold parameter, per-field histograms, run manifest"
git push origin main
```
