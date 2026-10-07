# Otsu uncertainty band (inference script 0.0.5, 7 Oct 2026)

Dr. Yann's reply to the 0.0.4 results (7 Oct): the 0.50 cut-off is right (it is
what the earlier binary maps used), and the uncertainty should come from
**Otsu's method** instead of the fixed 0.30–0.70 band.

What 0.0.5 changes (on top of 0.0.4, see `ALMA_inference_otsu_0.0.5.diff`):
- The wheat / not-wheat cut-off stays `THRESH=0.50`: areas and statuses do not change.
- **Uncertain band from 3-class (multi-level) Otsu** on the probability histogram
  of the whole area: two thresholds; below = confident background, above =
  confident wheat, between = uncertain. Used for `uncertainty_frac`, map 04, the ⚠
  flags and the credit score. `UNC_METHOD=fixed UNC_LO=0.30 UNC_HI=0.70` gives the old band.
- The 2-class Otsu threshold is computed and recorded for reference (not used).
- All thresholds go to `run_params.json`, `run_manifest.json` and the GeoJSON header (`alma_run`).
- Pixels that are exactly 0.0 (tile seams, no prediction written) are now nodata.
  This removes about 7 ha of field area in 8 fields; active area is unchanged.
- Otsu is implemented in numpy (no new package) and matches scikit-image exactly.

Expected on the 6 Oct raster (tested here on the real file): Otsu 0.458;
3-class Otsu 0.430 / 0.492; still 1179.4 ha and 34 / 2 / 1 fields; **5 fields
flagged** (10, 27, 29, 32, 34) instead of 37; mean credit score **73** (was 84 on
0/1 output, 56 with the fixed band).

The model, `alma_tasks.py` and `model.yaml` stay as set up for 0.0.4.

## 1. Copy the script to g13 (WSL, VPN up)

```bash
P=/mnt/d/work/alma/olmo-opencl-a100-wheat/scripts/patches/2026-10-07_otsu_uncertainty
scp $P/ALMA_Inference_Wheat_Festival.sh almauser1@10.172.0.233:dev/olmoearth_projects/incoming_v005.sh
```

## 2. On g13: install and run

```bash
cd ~/dev/olmoearth_projects
cp ALMA_Inference_Wheat_Festival.sh ALMA_Inference_Wheat_Festival.sh.v004
cp incoming_v005.sh ALMA_Inference_Wheat_Festival.sh
sed -i "s#^CHECKPOINT=.*#CHECKPOINT=\"$HOME/RSDATA/alma_wheat_festival/trainer_checkpoints/epoch=13-step=812.ckpt\"#" ALMA_Inference_Wheat_Festival.sh
head -3 ALMA_Inference_Wheat_Festival.sh | tail -1        # 0.0.5
grep -n '^CHECKPOINT=' ALMA_Inference_Wheat_Festival.sh     # epoch=13-step=812
grep -n 'alma_tasks' olmoearth_run_data/wheat_festival/model.yaml   # still there

S=~/RSDATA/alma_wheat_festival
mv $S/inference_outputs/20261007 $S/inference_outputs/20261007.old_$(date +%H%M%S) 2>/dev/null
CUDA_VISIBLE_DEVICES=1 nohup bash ALMA_Inference_Wheat_Festival.sh \
    2024-11-01 2025-04-30 $S < /dev/null > ~/g13_infer_v005.log 2>&1 &
tail -f ~/g13_infer_v005.log
```

Do **not** move `dataset_0` this time: it holds the 6 Oct imagery and probability
prediction, so the model step reuses them and the raster stays the same as on 6 Oct.
In the log look for the `Otsu threshold` line and, in the summary box, `Fields flagged ⚠ : 5`.

## 3. Check, copy, push

```bash
O=$(ls -d ~/RSDATA/alma_wheat_festival/inference_outputs/2026* | grep -v old | tail -1)
cat $O/run_params.json; tail -12 ~/g13_infer_v005.log
sha256sum $O/wheat_probability_epsg32640.tif   # 4ef0a67b… = same raster as 6 Oct
```

```bash
# WSL
D=/mnt/d/work/alma/olmo-opencl-a100-wheat/g13_inference_20261007
mkdir -p $D
scp -r "almauser1@10.172.0.233:RSDATA/alma_wheat_festival/inference_outputs/20261007/*" $D/
```
