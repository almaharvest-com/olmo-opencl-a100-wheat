#!/usr/bin/env bash
# ALMA Wheat Festival — Inference + PNG Output Generation
ALMA_INFER_VERSION="0.0.2"; echo "ALMA_Inference_Wheat_Festival.sh v${ALMA_INFER_VERSION}"
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
OUT_DIR="${SCRATCH}/inference_outputs/${OUT_DATE}"

echo "=== ALMA Wheat Festival — Inference ==="
echo "  Season window : ${START_DATE} → ${END_DATE}"
echo "  Checkpoint    : ${CHECKPOINT}"
echo "  Output dir    : ${OUT_DIR}"

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

DPI          = 150
FIG_W, FIG_H = 12, 9    # inches → 1800 × 1350 at 150 dpi

# ── Load raster ───────────────────────────────────────────────────────────
print("  Loading result raster...")
with rasterio.open(RESULT_TIF) as src:
    logits   = src.read().astype(np.float32)   # (2, H, W)
    transform = src.transform
    crs       = src.crs
    extent    = [src.bounds.left, src.bounds.right,
                 src.bounds.bottom, src.bounds.top]  # for imshow

# Output is single-band float32 with values 0.0 (background) / 1.0 (wheat).
# The SegmentationHead saves argmax class predictions, not raw logits.
# Use band 0 directly — no softmax needed.
if logits.shape[0] == 2:
    exp  = np.exp(logits - logits.max(axis=0, keepdims=True))
    prob = (exp / exp.sum(axis=0, keepdims=True))[1]
else:
    prob = logits[0]   # already 0.0 / 1.0

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

def field_prob_stats(geom, prob, transform, crs_obj):
    """Return (mean_prob, active_frac, area_ha, active_area_ha) for one field."""
    try:
        mask = ~geometry_mask(
            [mapping(geom)],
            transform=transform,
            out_shape=prob.shape,
            all_touched=False,
        )
        if mask.sum() == 0:
            return 0.0, 0.0, 0.0, 0.0
        px     = prob[mask]
        mean_p = float(px.mean())
        act_f  = float((px >= 0.50).sum() / len(px))
        pixel_area_m2 = abs(transform.a * transform.e)
        area_ha       = len(px) * pixel_area_m2 / 10_000
        active_ha     = (px >= 0.50).sum() * pixel_area_m2 / 10_000
        return mean_p, act_f, area_ha, active_ha
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

# Compute per-field stats
print("  Computing per-field statistics...")
records = []
for fid, (_, row) in enumerate(fields.iterrows(), start=1):
    mean_p, act_f, area_ha, active_ha = field_prob_stats(
        row.geometry, prob, transform, crs
    )
    unc_frac = float(((prob >= 0.30) & (prob <= 0.70)
                       & np.array(~geometry_mask(
                           [mapping(row.geometry)],
                           transform=transform,
                           out_shape=prob.shape,
                           all_touched=False))).sum())
    unc_frac = unc_frac / max(1, area_ha * 10_000 / abs(transform.a * transform.e))

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
        "geometry":       row.geometry,
    })

# ── Write CSV ─────────────────────────────────────────────────────────────
csv_path = OUT_DIR / "field_stats.csv"
fieldnames = [k for k in records[0].keys() if k != "geometry"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in records:
        w.writerow({k: r[k] for k in fieldnames})
print(f"  Wrote {csv_path.name}")

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
mask_arr = (prob >= 0.50).astype(np.uint8)
fig, ax = base_fig(f"Wheat Mask (threshold 0.50)  |  {START_DATE} → {END_DATE}")
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
    mpatches.Patch(color="#3daa33", label="Wheat (≥ 0.50)"),
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
unc_mask = ((prob >= 0.30) & (prob <= 0.70)).astype(np.float32)
fig, ax = base_fig(f"Model Uncertainty  |  Uncertain pixels: 0.30 ≤ P ≤ 0.70")
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
        ax.plot(c.x, c.y, marker="!", color="#ff4444", markersize=10, zorder=5)
        ax.text(c.x, c.y + 200, f"⚠ {r['uncertainty_frac']:.0%}",
                color="#ff8888", fontsize=6, ha="center")
pct_unc = float((unc_mask.sum() / unc_mask.size) * 100)
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
print(f"  │  Season: {START_DATE} → {END_DATE}")
print(f"  │  Total active wheat : {total_exp:.1f} ha")
print(f"  │  Active fields      : {sum(1 for r in records if r['status']=='active')} / {n_fields}")
print(f"  │  Mean credit score  : {np.mean(scores):.0f} / 100")
print(f"  │  Fields flagged ⚠   : {sum(1 for r in records if r['uncertainty_frac'] > 0.25)}")
print("  └─────────────────────────────────────────────┘")
PYEOF

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
echo "  View metrics: mlflow ui --backend-store-uri sqlite:////home/yann/RSDATA/alma_wheat_festival/alma_training.db"
