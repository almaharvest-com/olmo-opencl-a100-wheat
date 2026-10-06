#!/usr/bin/env bash
# ALMA Wheat Festival — Inference + PNG Output Generation
ALMA_INFER_VERSION="0.0.4"; echo "ALMA_Inference_Wheat_Festival.sh v${ALMA_INFER_VERSION}"
#
# Runs the trained OlmoEarth-Tiny model over the Meilha AOI and produces
# six PNG outputs for crop insurance, traceability, and micro-credit allocation.
#
# Usage:
#   bash ALMA_Inference_Wheat_Festival.sh [start_date] [end_date] [scratch_dir]
#
#   start_date : ISO date for Sentinel-2 window start (default: 2025-01-01)
#   end_date   : ISO date for Sentinel-2 window end   (default: 2025-03-31)
#   scratch_dir: RSDATA root                           (default: ~/RSDATA/alma_wheat_festival)
#
# Outputs — written to ${SCRATCH}/inference_outputs/YYYYMMDD/ :
#   01_wheat_probability.png  — continuous per-pixel wheat probability
#   02_wheat_mask.png         — binary mask + field boundaries
#   03_field_activation.png   — 37 fields coloured active/partial/inactive
#   04_uncertainty.png        — pixels where model is unsure (0.30–0.70)
#   05_credit_score.png       — per-field micro-credit score (0–100)
#   06_insurance_exposure.png — per-field insurable active area (ha)
#   field_stats.csv           — machine-readable per-field table
#   field_stats.geojson       — same table with the field polygons (EPSG:4326)
#   wheat_probability_epsg32640.tif — the per-pixel wheat probability (0–1)
#   threshold_sweep.csv       — total active ha / field counts at cut-offs 0.10–0.90
#   run_manifest.json         — what produced this run (checkpoint, config,
#                               imagery, checksums), for long-term reproducibility
#
# 0.0.4: the model now writes wheat PROBABILITY, not argmax 0/1: model.yaml
# names  class_path: alma_tasks.WheatProbSegmentationTask  for wheat_seg
# (alma_tasks.py sits next to this script; works with any rslearn version). The cut-off is a parameter:
#   THRESH=0.50 UNC_LO=0.30 UNC_HI=0.70 bash ALMA_Inference_Wheat_Festival.sh ...
# Each field also carries prob_hist: 20 bins of 0.05 = share of the field's
# pixels in each probability band, so a viewer can recompute area and status
# at any cut-off (threshold slider) without the raster.
#
# Executive rationale (inline):
#   Crop insurance, traceability and micro-credit all require a trusted,
#   independently verifiable record of WHICH fields are active, HOW MUCH of
#   each field is cultivated, and HOW CONFIDENT that determination is.
#   These six outputs supply exactly those three quantities at field resolution
#   every 10 days, replacing paper declarations and reducing fraud risk.

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_DATE="${1:-2025-01-01}"
END_DATE="${2:-2025-03-31}"
SCRATCH="${3:-${HOME}/RSDATA/alma_wheat_festival}"
CHECKPOINT="${SCRATCH}/trainer_checkpoints/epoch=12-step=754.ckpt"
PROJECT_PATH="${REPO_DIR}/olmoearth_run_data/wheat_festival"
FIELDS_GEOJSON="${HOME}/GISDATA/Sharjah_WheatFestival/Sharjah_Wheat_fields_4326.geojson"
OUT_DATE=$(date +%Y%m%d)
THRESH="${THRESH:-0.50}"     # wheat / not-wheat cut-off on the probability
UNC_LO="${UNC_LO:-0.30}"     # uncertain band, lower edge
UNC_HI="${UNC_HI:-0.70}"     # uncertain band, upper edge
OUT_DIR="${SCRATCH}/inference_outputs/${OUT_DATE}"

echo "=== ALMA Wheat Festival — Inference ==="
echo "  Season window : ${START_DATE} → ${END_DATE}"
echo "  Checkpoint    : ${CHECKPOINT}"
echo "  Output dir    : ${OUT_DIR}"
echo "  Cut-off       : ${THRESH}   uncertain band: ${UNC_LO}–${UNC_HI}"

mkdir -p "${OUT_DIR}"

# ---------------------------------------------------------------------------
# 1. Pre-flight checks
# ---------------------------------------------------------------------------
if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    echo "  Run ALMA_Train_Wheat_Festival.sh first, or set correct path."
    exit 1
fi

if [ ! -f "${FIELDS_GEOJSON}" ]; then
    echo "ERROR: Field boundaries not found: ${FIELDS_GEOJSON}"
    exit 1
fi

source "${REPO_DIR}/.venv/bin/activate"
# alma_tasks.py (wheat probability task, used by model.yaml) sits next to this script
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

python3 -c "import rasterio, geopandas, matplotlib, scipy, mlflow" 2>/dev/null || {
    echo "ERROR: Missing dependencies. Run: uv add mlflow && pip install rasterio geopandas matplotlib scipy"
    exit 1
}

# ---------------------------------------------------------------------------
# 2. Update prediction time window
# ---------------------------------------------------------------------------
echo ""
echo "[1/3] Setting prediction window ${START_DATE} → ${END_DATE}..."

python3 - << PYEOF
import json, pathlib
p = pathlib.Path("${PROJECT_PATH}/prediction_request_geometry.geojson")
fc = json.loads(p.read_text())
fc["features"][0]["properties"]["oe_start_time"] = "${START_DATE}T00:00:00+00:00"
fc["features"][0]["properties"]["oe_end_time"]   = "${END_DATE}T00:00:00+00:00"
p.write_text(json.dumps(fc, indent=2))
print(f"  Updated: {p}")
PYEOF

# ---------------------------------------------------------------------------
# 3. Run inference (fetches fresh Sentinel-2, runs sliding-window segmentation)
# ---------------------------------------------------------------------------
# The model tiles the Meilha AOI in 64×64 px (640×640 m) windows with 12.5 %
# overlap, runs the frozen OlmoEarth-Tiny encoder + decoder on each tile,
# and mosaics the results into a single GeoTIFF.
echo ""
echo "[2/3] Running model inference over Meilha AOI..."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled

python -m olmoearth_projects.main olmoearth_run olmoearth_run \
    --config_path "${PROJECT_PATH}" \
    --scratch_path "${SCRATCH}" \
    --checkpoint_path "${CHECKPOINT}"

# Find the result raster (most recently written .tif)
RESULT_TIF=$(find "${SCRATCH}/results/results_raster" -name "*.tif" \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | awk '{print $2}')

if [ -z "${RESULT_TIF}" ]; then
    echo "ERROR: No result raster found in ${SCRATCH}/results/results_raster/"
    exit 1
fi
echo "  Result raster: ${RESULT_TIF}"

# ---------------------------------------------------------------------------
# 4. Generate all six PNG outputs
# ---------------------------------------------------------------------------
echo ""
echo "[3/3] Generating PNG outputs..."

python3 << PYEOF
"""
Six-panel PNG generator for ALMA Wheat Festival inference outputs.

Each panel is independently useful for a different operational audience:
  Insurance    → 02 mask, 06 exposure
  Traceability → 03 activation
  Micro-credit → 05 credit score
  Monitoring   → 01 probability, 04 uncertainty

The reasoning: satellite-derived field status is only actionable when it is
communicated at the right granularity for each decision. Underwriters need
hectares. Credit officers need scores. Agronomists need raw probability.
This script produces all six simultaneously from one inference run.
"""

import json
import csv
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import rasterio
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from rasterio.features import shapes as rio_shapes
from rasterio.warp import transform_geom
from scipy.ndimage import label as ndi_label
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import pyproj

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
RESULT_TIF   = Path("${RESULT_TIF}")
FIELDS_FILE  = Path("${FIELDS_GEOJSON}")
OUT_DIR      = Path("${OUT_DIR}")
START_DATE   = "${START_DATE}"
END_DATE     = "${END_DATE}"

THRESH       = float("${THRESH}")
UNC_LO       = float("${UNC_LO}")
UNC_HI       = float("${UNC_HI}")
N_BINS       = 20          # prob_hist bins of 0.05

DPI          = 150
FIG_W, FIG_H = 12, 9    # inches → 1800 × 1350 at 150 dpi

# ── Load raster ───────────────────────────────────────────────────────────
print("  Loading result raster...")
with rasterio.open(RESULT_TIF) as src:
    logits   = src.read().astype(np.float32)   # (2, H, W)
    transform = src.transform
    crs       = src.crs
    src_nodata = src.nodata
    extent    = [src.bounds.left, src.bounds.right,
                 src.bounds.bottom, src.bounds.top]  # for imshow

# Since 0.0.4 the model writes the wheat probability (softmax of class 1),
# one float32 band in 0–1. The old argmax output (0/1 only) made
# active_frac == mean_prob and uncertainty_frac == 0, so refuse it.
nodata = src_nodata
if logits.shape[0] == 2:
    exp  = np.exp(logits - logits.max(axis=0, keepdims=True))
    prob = (exp / exp.sum(axis=0, keepdims=True))[1]
else:
    prob = logits[0].copy()
invalid = ~np.isfinite(prob) | (prob < 0) | (prob > 1)
if nodata is not None:
    invalid |= (prob == nodata)
prob[invalid] = np.nan
valid = prob[~np.isnan(prob)]
if valid.size == 0:
    raise SystemExit("ERROR: the result raster has no valid pixels")
n_mid = int(((valid > 0.001) & (valid < 0.999)).sum())
print(f"  Raster: {valid.size} valid px, {n_mid} px strictly between 0 and 1, "
      f"min {valid.min():.3f} max {valid.max():.3f}")
if n_mid == 0:
    raise SystemExit(
        "ERROR: the raster holds only 0/1 class labels, not probabilities.\n"
        "  model.yaml must name  class_path: alma_tasks.WheatProbSegmentationTask\n"
        "  for wheat_seg (see patch_model_yaml_prob_task.py), and dataset_0 must not\n"
        "  hold an old 0/1 prediction. Fix and run the inference again.")

# Reproject field boundaries to raster CRS (UTM 40N)
print("  Loading field boundaries...")
fields_wgs = gpd.read_file(FIELDS_FILE)
fields     = fields_wgs.to_crs(crs)
n_fields   = len(fields)

# Compute zoom bbox: tight around field cluster + 800 m margin
MARGIN = 800   # metres — enough context without showing noisy desert
fminx, fminy, fmaxx, fmaxy = fields.total_bounds
zoom_xlim = (fminx - MARGIN, fmaxx + MARGIN)
zoom_ylim = (fminy - MARGIN, fmaxy + MARGIN)

def zoom_to_fields(ax):
    ax.set_xlim(*zoom_xlim)
    ax.set_ylim(*zoom_ylim)

# ── Helper: field mask for one row ────────────────────────────────────────
from rasterio.features import geometry_mask

def field_pixels(geom):
    """Probabilities of the valid pixels inside one field (1-D array)."""
    mask = ~geometry_mask([mapping(geom)], transform=transform,
                          out_shape=prob.shape, all_touched=False)
    px = prob[mask]
    return px[~np.isnan(px)]

PIXEL_AREA_M2 = abs(transform.a * transform.e)

def field_prob_stats(px):
    """Return (mean_prob, active_frac, uncertainty_frac, area_ha, active_area_ha, prob_hist)."""
    if px.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, [0.0] * N_BINS
    mean_p    = float(px.mean())
    n_act     = int((px >= THRESH).sum())
    act_f     = n_act / px.size
    unc_f     = float(((px >= UNC_LO) & (px <= UNC_HI)).sum() / px.size)
    area_ha   = px.size * PIXEL_AREA_M2 / 10_000
    active_ha = n_act * PIXEL_AREA_M2 / 10_000
    counts, _ = np.histogram(px, bins=N_BINS, range=(0.0, 1.0))
    hist      = [round(float(c) / px.size, 4) for c in counts]
    return mean_p, act_f, unc_f, area_ha, active_ha, hist

# Compute per-field stats
print("  Computing per-field statistics...")
records = []
field_px = {}
for fid, (_, row) in enumerate(fields.iterrows(), start=1):
    px = field_pixels(row.geometry)
    mean_p, act_f, unc_frac, area_ha, active_ha, hist = field_prob_stats(px)
    field_px[fid] = px

    if act_f >= 0.60:
        status = "active"
    elif act_f >= 0.20:
        status = "partial"
    else:
        status = "inactive"

    # Credit score: weighted composite (higher = more creditworthy)
    # Rationale: lenders need a single number per field for automated
    # loan-tranche decision. Active area fraction is the primary driver
    # (50 %), mean probability captures canopy density (30 %), and
    # consistency (inverse of uncertainty) captures reliability (20 %).
    credit = int(round(
        0.50 * act_f * 100 +
        0.30 * mean_p * 100 +
        0.20 * max(0, 1 - unc_frac) * 100
    ))

    records.append({
        "field_id":       fid,
        "diam_km":        float(row.get("Diam_km", 0)),
        "status":         status,
        "active_frac":    round(act_f, 3),
        "mean_prob":      round(mean_p, 3),
        "uncertainty_frac": round(unc_frac, 3),
        "area_ha":        round(area_ha, 2),
        "active_area_ha": round(active_ha, 2),
        "credit_score":   credit,
        "threshold":      THRESH,
        "prob_hist":      hist,
        "geometry":       row.geometry,
    })

# ── Write CSV ─────────────────────────────────────────────────────────────
csv_path = OUT_DIR / "field_stats.csv"
fieldnames = [k for k in records[0].keys() if k != "geometry"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in records:
        w.writerow({k: (" ".join(f"{v:.4f}" for v in r[k]) if k == "prob_hist" else r[k])
                    for k in fieldnames})
print(f"  Wrote {csv_path.name}")

# ── Write GeoJSON ─────────────────────────────────────────────────────────
# Same per-field figures as the CSV, with the field polygons, so GIS tools
# and dashboards can map every theme without a join on field_id.
geojson_path = OUT_DIR / "field_stats.geojson"
_g4326 = gpd.GeoSeries([r["geometry"] for r in records], crs=crs).to_crs(4326)
_feats = []
for r, g in zip(records, _g4326):
    props = {k: v for k, v in r.items() if k != "geometry"}
    _feats.append({"type": "Feature", "properties": props, "geometry": mapping(g)})
_fc = {"type": "FeatureCollection",
       "name": "field_stats",
       "alma_run": {"script_version": "${ALMA_INFER_VERSION}",
                    "threshold": THRESH, "uncertain_band": [UNC_LO, UNC_HI],
                    "prob_hist_bins": [round(i / N_BINS, 2) for i in range(N_BINS + 1)],
                    "season": [START_DATE, END_DATE]},
       "features": _feats}
geojson_path.write_text(json.dumps(_fc, default=float))
print(f"  Wrote {geojson_path.name}")

# ── Threshold sweep ───────────────────────────────────────────────────────
# How the headline numbers move with the cut-off: the basis for choosing it.
sweep_path = OUT_DIR / "threshold_sweep.csv"
with open(sweep_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["threshold", "active_area_ha", "fields_active", "fields_partial", "fields_inactive"])
    for t in np.round(np.arange(0.10, 0.91, 0.05), 2):
        ha, na, npar, nin = 0.0, 0, 0, 0
        for px in field_px.values():
            if px.size == 0:
                nin += 1
                continue
            n = int((px >= t).sum())
            ha += n * PIXEL_AREA_M2 / 10_000
            fr = n / px.size
            if fr >= 0.60:
                na += 1
            elif fr >= 0.20:
                npar += 1
            else:
                nin += 1
        w.writerow([f"{t:.2f}", round(ha, 1), na, npar, nin])
print(f"  Wrote {sweep_path.name}")

# ── Probability raster copy, next to the tables ───────────────────────────
import shutil
shutil.copy2(RESULT_TIF, OUT_DIR / "wheat_probability_epsg32640.tif")
print("  Wrote wheat_probability_epsg32640.tif")

# ── Colormaps ─────────────────────────────────────────────────────────────
wheat_cmap = LinearSegmentedColormap.from_list(
    "wheat", ["#d4b483", "#f5e6a3", "#c8e89a", "#4fa847"])
unc_cmap   = LinearSegmentedColormap.from_list(
    "unc",   ["#ffffff00", "#ff8c00aa", "#cc0000cc"])

STATUS_COLORS = {"active": "#3aa832", "partial": "#f5c518", "inactive": "#d42020"}

def base_fig(title):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")
    fig.suptitle(title, color="white", fontsize=13, fontweight="bold", y=0.97)
    return fig, ax

def add_fields_outline(ax, alpha=0.6):
    fields.boundary.plot(ax=ax, color="white", linewidth=0.6, alpha=alpha)

def add_caption(fig, text):
    fig.text(0.01, 0.01, text, color="#aaaaaa", fontsize=7,
             verticalalignment="bottom", wrap=True,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d0d1a", alpha=0.7))

# ── 01 Wheat probability ──────────────────────────────────────────────────
# Rationale: the raw probability surface is the primary scientific output.
# It shows WHERE wheat is growing AND how confidently the model knows it.
# Agronomists and remote-sensing specialists use this for QA and calibration.
print("  01 wheat probability...")
fig, ax = base_fig(f"Wheat Probability  |  Meilha, Sharjah  |  {START_DATE} → {END_DATE}")
im = ax.imshow(prob, extent=extent, cmap=wheat_cmap, vmin=0, vmax=1,
               origin="upper", interpolation="bilinear")
add_fields_outline(ax)
zoom_to_fields(ax)
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("P(wheat)", color="white", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)
ax.set_xlabel("Easting (m, UTM 40N)", color="#aaaaaa", fontsize=8)
ax.set_ylabel("Northing (m, UTM 40N)", color="#aaaaaa", fontsize=8)
add_caption(fig,
    "Source: OlmoEarth-Tiny fine-tuned on 37 Meilha fields | Sentinel-2 L2A 10 m")
fig.savefig(OUT_DIR / "01_wheat_probability.png", dpi=DPI,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)

# ── 02 Binary wheat mask ──────────────────────────────────────────────────
# Rationale: the binary mask is the authoritative area figure.  Insurance
# underwriters and government statisticians require a yes/no classification
# per pixel, not a probability. This is the input to area calculations.
print("  02 binary wheat mask...")
mask_arr = (np.nan_to_num(prob, nan=0.0) >= THRESH).astype(np.uint8)
fig, ax = base_fig(f"Wheat Mask (threshold {THRESH:.2f})  |  {START_DATE} → {END_DATE}")
mask_rgba = np.zeros((*mask_arr.shape, 4), dtype=np.float32)
mask_rgba[mask_arr == 1] = [0.24, 0.66, 0.20, 0.85]    # green wheat
mask_rgba[mask_arr == 0] = [0.60, 0.55, 0.40, 0.30]    # faint sand background
ax.imshow(mask_rgba, extent=extent, origin="upper")
add_fields_outline(ax, alpha=0.9)
zoom_to_fields(ax)
# Annotate active area per field
for r in records:
    if r["active_area_ha"] > 0.01:
        c = r["geometry"].centroid
        ax.text(c.x, c.y, f"{r['active_area_ha']:.1f} ha",
                color="white", fontsize=5.5, ha="center", va="center",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="#00000066", ec="none"))
patches = [
    mpatches.Patch(color="#3daa33", label=f"Wheat (≥ {THRESH:.2f})"),
    mpatches.Patch(color="#99887766", label="Background"),
]
ax.legend(handles=patches, loc="lower right", fontsize=8,
          facecolor="#1a1a2e", edgecolor="#555555", labelcolor="white")
total_ha = sum(r["active_area_ha"] for r in records)
ax.set_title(f"Total active wheat: {total_ha:.1f} ha  |  "
             f"{sum(1 for r in records if r['status']=='active')} of {n_fields} fields fully active",
             color="#cccccc", fontsize=9, pad=4)
add_caption(fig, "Use: insurance area baseline, government crop statistics")
fig.savefig(OUT_DIR / "02_wheat_mask.png", dpi=DPI,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)

# ── 03 Field activation map ───────────────────────────────────────────────
# Rationale: traceability and micro-credit require a field-level verdict, not
# a pixel-level one. A field is either "in production" or it isn't.  This map
# is the digital origin certificate: it proves field X was active in season Y.
# For export traceability it links the flour batch to a GPS polygon with a date.
print("  03 field activation...")
fig, ax = base_fig(
    f"Field Activation Status  |  {START_DATE} → {END_DATE}\n"
    "Green=Active (≥60 %)  Yellow=Partial (20–60 %)  Red=Inactive (<20 %)"
)
# Light background
ax.imshow([[0.1, 0.1], [0.1, 0.1]], extent=extent,
          cmap="gray", vmin=0, vmax=1, origin="upper", alpha=0.1)
zoom_to_fields(ax)
for r in records:
    color = STATUS_COLORS[r["status"]]
    gpd.GeoSeries([r["geometry"]], crs=crs).plot(
        ax=ax, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
    c = r["geometry"].centroid
    ax.text(c.x, c.y, str(r["field_id"]),
            color="white", fontsize=5, ha="center", va="center", fontweight="bold")
patches = [mpatches.Patch(color=c, label=l) for l, c in [
    ("Active — full credit eligible",   STATUS_COLORS["active"]),
    ("Partial — reduced tranche",       STATUS_COLORS["partial"]),
    ("Inactive — hold / field visit",   STATUS_COLORS["inactive"]),
]]
ax.legend(handles=patches, loc="lower right", fontsize=8,
          facecolor="#1a1a2e", edgecolor="#555555", labelcolor="white")
n_act = sum(1 for r in records if r["status"] == "active")
n_par = sum(1 for r in records if r["status"] == "partial")
n_ina = sum(1 for r in records if r["status"] == "inactive")
ax.set_title(f"Active: {n_act}  |  Partial: {n_par}  |  Inactive: {n_ina}",
             color="#cccccc", fontsize=9, pad=4)
add_caption(fig,
    "Use: micro-credit tranche release, export traceability, government registry update")
fig.savefig(OUT_DIR / "03_field_activation.png", dpi=DPI,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)

# ── 04 Uncertainty map ────────────────────────────────────────────────────
# Rationale: any automated financial decision needs a confidence qualifier.
# Pixels in the 0.30–0.70 range are genuinely ambiguous — the model cannot
# reliably assign them to wheat or background.  Fields with high uncertainty
# fractions MUST be visited before credit or insurance decisions are finalised.
# This map operationalises the "human-in-the-loop" requirement.
print("  04 uncertainty...")
unc_mask = ((prob >= UNC_LO) & (prob <= UNC_HI)).astype(np.float32)   # NaN compares False
fig, ax = base_fig(f"Model Uncertainty  |  Uncertain pixels: {UNC_LO:.2f} ≤ P ≤ {UNC_HI:.2f}")
# Base: dim probability
ax.imshow(prob, extent=extent, cmap=wheat_cmap, vmin=0, vmax=1,
          origin="upper", alpha=0.35)
# Overlay uncertainty in orange
zoom_to_fields(ax)
unc_rgba = np.zeros((*unc_mask.shape, 4), dtype=np.float32)
unc_rgba[..., 0] = 1.0
unc_rgba[..., 1] = 0.55
unc_rgba[..., 2] = 0.0
unc_rgba[..., 3] = unc_mask * 0.85
ax.imshow(unc_rgba, extent=extent, origin="upper")
add_fields_outline(ax)
# Flag high-uncertainty fields
for r in records:
    if r["uncertainty_frac"] > 0.25:
        c = r["geometry"].centroid
        ax.plot(c.x, c.y, marker="^", color="#ff4444", markersize=10, zorder=5)
        ax.text(c.x, c.y + 200, f"⚠ {r['uncertainty_frac']:.0%}",
                color="#ff8888", fontsize=6, ha="center")
pct_unc = float((unc_mask.sum() / max(1, int(np.isfinite(prob).sum()))) * 100)
ax.set_title(f"{pct_unc:.1f} % of AOI pixels are uncertain  "
             f"|  ⚠ = fields requiring physical verification",
             color="#cccccc", fontsize=9, pad=4)
add_caption(fig,
    "Use: audit trigger — ⚠ fields must be visited before credit/insurance decision")
fig.savefig(OUT_DIR / "04_uncertainty.png", dpi=DPI,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)

# ── 05 Credit score map ───────────────────────────────────────────────────
# Rationale: micro-credit institutions in the UAE disburse loans against
# confirmed crop activity. A single 0–100 score per field — combining active
# fraction, canopy density, and model confidence — lets a credit officer
# process all 37 fields in seconds rather than days of field visits.
# Fields scoring ≥75 receive full tranche; 50–74 receive 60 %; <50 are held.
print("  05 credit score...")
score_cmap = LinearSegmentedColormap.from_list(
    "score", ["#8b0000", "#ff6600", "#ffd700", "#7ccd7c", "#006400"])
fig, ax = base_fig(f"Micro-Credit Score (0–100)  |  {START_DATE} → {END_DATE}")
ax.imshow([[0]], extent=extent, cmap="gray", alpha=0.05, origin="upper")
zoom_to_fields(ax)
scores = [r["credit_score"] for r in records]
norm   = mcolors.Normalize(vmin=0, vmax=100)
sm     = plt.cm.ScalarMappable(cmap=score_cmap, norm=norm)
for r in records:
    gpd.GeoSeries([r["geometry"]], crs=crs).plot(
        ax=ax, color=score_cmap(norm(r["credit_score"])),
        alpha=0.85, edgecolor="white", linewidth=0.5)
    c = r["geometry"].centroid
    ax.text(c.x, c.y, str(r["credit_score"]),
            color="white" if r["credit_score"] < 80 else "black",
            fontsize=6, ha="center", va="center", fontweight="bold")
cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Credit score", color="white", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)
thresholds = [
    mpatches.Patch(color=score_cmap(norm(87)), label="≥75 — Full loan tranche"),
    mpatches.Patch(color=score_cmap(norm(62)), label="50–74 — 60% tranche"),
    mpatches.Patch(color=score_cmap(norm(25)), label="<50 — Hold: field inspection"),
]
ax.legend(handles=thresholds, loc="lower right", fontsize=8,
          facecolor="#1a1a2e", edgecolor="#555555", labelcolor="white")
mean_score = np.mean(scores)
ax.set_title(f"Mean score: {mean_score:.0f}/100  |  "
             f"Full tranche: {sum(1 for s in scores if s>=75)}  "
             f"Partial: {sum(1 for s in scores if 50<=s<75)}  "
             f"Hold: {sum(1 for s in scores if s<50)}",
             color="#cccccc", fontsize=9, pad=4)
add_caption(fig,
    "Score = 50%×active_fraction + 30%×mean_probability + 20%×(1−uncertainty)  "
    "| Weights set for arid small-field context")
fig.savefig(OUT_DIR / "05_credit_score.png", dpi=DPI,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)

# ── 06 Insurance exposure map ─────────────────────────────────────────────
# Rationale: multi-peril crop insurance (MPCI) requires knowing the ACTUAL
# planted area at the START of the insured period, not just the registered area.
# Farmers sometimes under-declare (tax avoidance) or over-declare (inflated
# claims). This map establishes the satellite-confirmed active area per field
# BEFORE any weather or pest event. Claims can only be settled against this
# baseline, preventing over-claim fraud and ensuring premium fairness.
print("  06 insurance exposure...")
exp_cmap = LinearSegmentedColormap.from_list(
    "exposure", ["#e8f4f8", "#9ecae1", "#2171b5", "#08306b"])
fig, ax = base_fig(f"Insurance Exposure — Active Area per Field (ha)  |  {START_DATE} → {END_DATE}")
ax.imshow([[0]], extent=extent, cmap="gray", alpha=0.05, origin="upper")
zoom_to_fields(ax)
active_has = [r["active_area_ha"] for r in records]
max_ha     = max(active_has) if max(active_has) > 0 else 1
norm_exp   = mcolors.Normalize(vmin=0, vmax=max_ha)
sm_exp     = plt.cm.ScalarMappable(cmap=exp_cmap, norm=norm_exp)
for r in records:
    gpd.GeoSeries([r["geometry"]], crs=crs).plot(
        ax=ax, color=exp_cmap(norm_exp(r["active_area_ha"])),
        alpha=0.90, edgecolor="white", linewidth=0.5)
    c = r["geometry"].centroid
    ax.text(c.x, c.y, f"{r['active_area_ha']:.2f}",
            color="white", fontsize=5.5, ha="center", va="center", fontweight="bold")
cbar = fig.colorbar(sm_exp, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Active area (ha)", color="white", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)
total_exp   = sum(active_has)
total_reg   = sum(r["area_ha"] for r in records)
coverage_r  = total_exp / total_reg if total_reg > 0 else 0
ax.set_title(f"Total insurable exposure: {total_exp:.1f} ha  "
             f"|  Coverage ratio: {coverage_r:.0%} of registered area",
             color="#cccccc", fontsize=9, pad=4)
add_caption(fig,
    "Insurable exposure = satellite-confirmed active area at season start  "
    "| Baseline locked before any weather event for anti-fraud claim settlement")
fig.savefig(OUT_DIR / "06_insurance_exposure.png", dpi=DPI,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)

# ── Summary ───────────────────────────────────────────────────────────────
print("")
print("  ┌─────────────────────────────────────────────┐")
print(f"  │  Season: {START_DATE} → {END_DATE}   cut-off {THRESH:.2f}")
print(f"  │  Total active wheat : {total_exp:.1f} ha")
print(f"  │  Active fields      : {sum(1 for r in records if r['status']=='active')} / {n_fields}")
print(f"  │  Mean credit score  : {np.mean(scores):.0f} / 100")
print(f"  │  Fields flagged ⚠   : {sum(1 for r in records if r['uncertainty_frac'] > 0.25)}")
print("  └─────────────────────────────────────────────┘")
PYEOF

# ---------------------------------------------------------------------------
# 4b. Run manifest — what produced these numbers (insurers need the same
#     answer from the same inputs years later)
# ---------------------------------------------------------------------------
mkdir -p "${OUT_DIR}/provenance"
cp "${PROJECT_PATH}/model.yaml" "${PROJECT_PATH}/olmoearth_run.yaml" \
   "${PROJECT_PATH}/dataset.json" "${PROJECT_PATH}/prediction_request_geometry.geojson" \
   "${OUT_DIR}/provenance/" 2>/dev/null || true
cp "$(dirname "${CHECKPOINT}")/run_config.json" "${OUT_DIR}/provenance/" 2>/dev/null || true
cp "${REPO_DIR}/alma_tasks.py" "${OUT_DIR}/provenance/" 2>/dev/null || true
# Sentinel-2 scenes actually used (rslearn writes items.json per window)
find "${SCRATCH}" -path "*dataset_0*" -name items.json 2>/dev/null \
  | tar czf "${OUT_DIR}/provenance/imagery_items.tgz" -T - 2>/dev/null || true

python3 - << PYEOF2
import hashlib, json, os, pathlib, platform, datetime
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
out = pathlib.Path("${OUT_DIR}")
m = {
  "script": "ALMA_Inference_Wheat_Festival.sh",
  "script_version": "${ALMA_INFER_VERSION}",
  "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
  "host": platform.node(),
  "season": ["${START_DATE}", "${END_DATE}"],
  "threshold": float("${THRESH}"),
  "uncertain_band": [float("${UNC_LO}"), float("${UNC_HI}")],
  "checkpoint": {"path": "${CHECKPOINT}", "sha256": sha("${CHECKPOINT}")},
  "fields_file": {"path": "${FIELDS_GEOJSON}", "sha256": sha("${FIELDS_GEOJSON}")},
  "result_raster": {"path": "${RESULT_TIF}", "sha256": sha("${RESULT_TIF}")},
  "outputs": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
}
try:
    import rslearn, torch
    m["versions"] = {"rslearn": getattr(rslearn, "__version__", "?"), "torch": torch.__version__}
except Exception:
    pass
(out / "run_manifest.json").write_text(json.dumps(m, indent=2))
print("  Wrote run_manifest.json")
PYEOF2

# ---------------------------------------------------------------------------
# 5. Final report
# ---------------------------------------------------------------------------
echo ""
echo "=== Outputs written to ${OUT_DIR}/ ==="
ls -lh "${OUT_DIR}/"
echo ""
echo "Next steps:"
echo "  View PNGs : eog ${OUT_DIR}/*.png &"
echo "  Load in QGIS: ${RESULT_TIF}"
echo "  Import CSV to dashboard: ${OUT_DIR}/field_stats.csv"
echo "  Per-field map layer   : ${OUT_DIR}/field_stats.geojson"
echo "  Cut-off sweep         : ${OUT_DIR}/threshold_sweep.csv"
echo "  Provenance            : ${OUT_DIR}/run_manifest.json  ${OUT_DIR}/provenance/"
echo "  View metrics: mlflow ui --backend-store-uri sqlite:////home/yann/RSDATA/alma_wheat_festival/alma_training.db"
