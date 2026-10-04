#!/usr/bin/env bash
# ALMA Wheat Festival — Data Import & Project Setup
#
# Prepares the olmoearth_run_data/wheat_festival/ project and populates the
# rslearn dataset from Microsoft Planetary Computer Sentinel-2 imagery.
#
# Run from $HOME/dev/olmoearth_projects/ with the venv activated.
# Usage: bash ALMA_Import_Data.sh [optional: /path/to/scratch]

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GISDATA_DIR="${HOME}/GISDATA/Sharjah_WheatFestival"
PROJECT_PATH="${REPO_DIR}/olmoearth_run_data/wheat_festival"
ALMA_SCRATCH="${1:-${HOME}/RSDATA/alma_wheat_festival}"   # override by passing a second arg

echo "=== ALMA Wheat Festival — Import Data ==="
echo "  Repo:     ${REPO_DIR}"
echo "  GIS data: ${GISDATA_DIR}"
echo "  Project:  ${PROJECT_PATH}"
echo "  Scratch:  ${ALMA_SCRATCH}"
echo ""

# ---------------------------------------------------------------------------
# 1. Sanity checks
# ---------------------------------------------------------------------------
if [ ! -f "${GISDATA_DIR}/Sharjah_Wheat_fields_4326.geojson" ]; then
    echo "ERROR: Wheat fields GeoJSON not found at ${GISDATA_DIR}/Sharjah_Wheat_fields_4326.geojson"
    exit 1
fi

if ! python -c "import rslearn; import olmoearth_projects" 2>/dev/null; then
    echo "ERROR: rslearn / olmoearth_projects not importable. Activate the venv first."
    echo "       source ${REPO_DIR}/.venv/bin/activate"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Create project directory and write config files
# ---------------------------------------------------------------------------
mkdir -p "${PROJECT_PATH}"
mkdir -p "${ALMA_SCRATCH}"

echo "[1/6] Writing project config files to ${PROJECT_PATH}/"

# --- dataset.json ---
cat > "${PROJECT_PATH}/dataset.json" << 'EOF'
{
  "layers": {
    "label": {
      "band_sets": [
        {
          "bands": ["category"],
          "dtype": "int32"
        }
      ],
      "type": "raster"
    },
    "output": {
      "band_sets": [
        {
          "bands": ["output"],
          "dtype": "float32"
        }
      ],
      "type": "raster"
    },
    "sentinel2": {
      "band_sets": [
        {
          "bands": ["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12","B01","B09"],
          "dtype": "uint16"
        }
      ],
      "data_source": {
        "class_path": "rslearn.data_sources.planetary_computer.Sentinel2",
        "ingest": false,
        "init_args": {
          "cache_dir": "cache/planetary_computer",
          "harmonize": true,
          "sort_by": "eo:cloud_cover"
        },
        "query_config": {
          "max_matches": 6,
          "min_matches": 3,
          "period_duration": "30d",
          "space_mode": "PER_PERIOD_MOSAIC"
        }
      },
      "type": "raster"
    }
  }
}
EOF

# --- olmoearth_run.yaml ---
cat > "${PROJECT_PATH}/olmoearth_run.yaml" << 'EOF'
window_prep:
  labeled_window_preparer:
    class_path: olmoearth_run.runner.tools.labeled_window_preparers.polygon_to_raster_window_preparer.PolygonToRasterWindowPreparer
    init_args:
      window_resolution: 10.0  # 10 m/px, matching Sentinel-2

  data_splitter:
    class_path: olmoearth_run.runner.tools.data_splitters.random_data_splitter.RandomDataSplitter
    init_args:
      train_prop: 0.70
      val_prop: 0.15
      test_prop: 0.15
      seed: 42

  label_layer: "label"
  label_property: "label"     # matches the oe_labels key from annotation_features.geojson
  group_name: "random_split"
  split_property: "split"

partition_strategies:
  partition_request_geometry:
    class_path: olmoearth_run.runner.tools.partitioners.grid_partitioner.GridPartitioner
    init_args:
      grid_size: 0.25          # degrees; tiles the AOI into ~25km² partitions

  prepare_window_geometries:
    class_path: olmoearth_run.runner.tools.partitioners.grid_partitioner.GridPartitioner
    init_args:
      grid_size: 1024          # pixels; sliding-window tiles at inference time
      output_projection:
        class_path: rslearn.utils.geometry.Projection
        init_args:
          crs: EPSG:32640      # UTM zone 40N covering UAE
          x_resolution: 10
          y_resolution: -10
      use_utm: true

postprocessing_strategies:
  process_dataset:
    class_path: olmoearth_run.runner.tools.postprocessors.combine_geotiff.CombineGeotiff
    init_args:
      nodata_value: 255

  process_partition:
    class_path: olmoearth_run.runner.tools.postprocessors.combine_geotiff.CombineGeotiff
    init_args:
      nodata_value: 255

  process_window:
    class_path: olmoearth_run.runner.tools.postprocessors.noop_raster.NoopRaster

inference_results_config:
  data_type: RASTER
  classification_fields:
    - property_name: wheat_seg
      band_index: 1
      allowed_values:
        - value: 0
          label: background
          color: [200, 180, 140]   # sand colour
        - value: 1
          label: wheat_field
          color: [180, 220, 80]    # green
  detection_objects: null
  regression_fields: null
EOF

# --- model.yaml ---
cat > "${PROJECT_PATH}/model.yaml" << 'EOF'
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.multitask.MultiTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.olmoearth_pretrain.model.OlmoEarth
            init_args:
              model_id: OLMOEARTH_V1_BASE   # upgrade to OLMOEARTH_V1_1_BASE when available
              patch_size: 4
        decoders:
          wheat_seg:
            - class_path: rslearn.models.upsample.Upsample
              init_args:
                scale_factor: 4
            - class_path: rslearn.models.conv.Conv
              init_args:
                in_channels: 768
                out_channels: 2       # class 0 = background, class 1 = wheat
                kernel_size: 1
                activation:
                  class_path: torch.nn.Identity
            - class_path: rslearn.train.tasks.segmentation.SegmentationHead
    lr: 0.0001
    plateau: true
    plateau_factor: 0.2
    plateau_patience: 2
    plateau_min_lr: 0
    plateau_cooldown: 10
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    path: ${DATASET_PATH}
    inputs:
      sentinel2_l2a:
        data_type: "raster"
        layers: ["sentinel2"]
        bands: ["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12","B01","B09"]
        passthrough: true
        dtype: FLOAT32
        load_all_item_groups: true
        load_all_layers: true
      label:
        data_type: "raster"
        layers: ["label"]
        bands: ["category"]
        is_target: true
        dtype: INT32
    task:
      class_path: rslearn.train.tasks.multi_task.MultiTask
      init_args:
        tasks:
          wheat_seg:
            class_path: rslearn.train.tasks.segmentation.SegmentationTask
            init_args:
              num_classes: 2
              zero_is_invalid: false
              nodata_value: 255
              metric_kwargs:
                average: "micro"
              other_metrics:
                wheat_precision:
                  class_path: rslearn.train.tasks.segmentation.SegmentationMetric
                  init_args:
                    metric:
                      class_path: torchmetrics.classification.MulticlassPrecision
                      init_args:
                        num_classes: 2
                        average: null
                    class_idx: 1
                wheat_recall:
                  class_path: rslearn.train.tasks.segmentation.SegmentationMetric
                  init_args:
                    metric:
                      class_path: torchmetrics.classification.MulticlassRecall
                      init_args:
                        num_classes: 2
                        average: null
                    class_idx: 1
                wheat_f1:
                  class_path: rslearn.train.tasks.segmentation.SegmentationMetric
                  init_args:
                    metric:
                      class_path: torchmetrics.classification.MulticlassF1Score
                      init_args:
                        num_classes: 2
                        average: null
                    class_idx: 1
        input_mapping:
          wheat_seg:
            label: "targets"
    batch_size: 4
    num_workers: ${NUM_WORKERS}
    default_config:
      transforms:
        - class_path: rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize
          init_args:
            band_names:
              sentinel2_l2a: ["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12","B01","B09"]
      patch_size: 128
    train_config:
      transforms:
        - class_path: rslearn.train.transforms.flip.Flip
          init_args:
            image_selectors: ["sentinel2_l2a", "target/wheat_seg/classes", "target/wheat_seg/valid"]
        - class_path: rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize
          init_args:
            band_names:
              sentinel2_l2a: ["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12","B01","B09"]
      groups: ["random_split"]
      patch_size: 128
      tags:
        split: "train"
    val_config:
      groups: ["random_split"]
      patch_size: 128
      load_all_patches: true
      tags:
        split: "val"
    test_config:
      groups: ["random_split"]
      patch_size: 128
      load_all_patches: true
      tags:
        split: "test"
    predict_config:
      groups: ["group_partition_0"]
      patch_size: 128
      load_all_patches: true
      overlap_ratio: 0.125    # 16/128
      skip_targets: true
      transforms:
        - class_path: rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize
          init_args:
            band_names:
              sentinel2_l2a: ["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12","B01","B09"]
trainer:
  max_epochs: 80
  logger:
    class_path: lightning.pytorch.loggers.WandbLogger
    init_args:
      project: ${WANDB_PROJECT}
      name: ${WANDB_NAME}
      entity: ${WANDB_ENTITY}
  callbacks:
    - class_path: lightning.pytorch.callbacks.LearningRateMonitor
      init_args:
        logging_interval: "epoch"
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        dirpath: ${TRAINER_DATA_PATH}
        save_top_k: 3
        save_last: true
        monitor: val_wheat_seg/wheat_f1
        mode: max
    - class_path: rslearn.train.callbacks.freeze_unfreeze.FreezeUnfreeze
      init_args:
        module_selector: ["model", "encoder", 0]
        unfreeze_at_epoch: 15   # freeze encoder for first 15 epochs, then fine-tune end-to-end
        unfreeze_lr_factor: 10
    - class_path: rslearn.train.prediction_writer.RslearnWriter
      init_args:
        path: ${DATASET_PATH}
        output_layer: ${PREDICTION_OUTPUT_LAYER}
        selector: ["wheat_seg"]
        merger:
          class_path: rslearn.train.prediction_writer.RasterMerger
          init_args:
            padding: 2
# unused: ${EXTRA_FILES_PATH}
EOF

# --- prediction_request_geometry.geojson (full Meilha AOI for inference) ---
cat > "${PROJECT_PATH}/prediction_request_geometry.geojson" << 'EOF'
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [55.870, 25.065],
          [55.870, 25.140],
          [55.940, 25.140],
          [55.940, 25.065],
          [55.870, 25.065]
        ]]
      },
      "properties": {
        "oe_start_time": "2025-01-01T00:00:00+00:00",
        "oe_end_time": "2025-03-31T00:00:00+00:00"
      }
    }
  ]
}
EOF

echo "   Config files written."

# ---------------------------------------------------------------------------
# 3. Prepare annotated GeoJSON (add label, time, background polygons)
# ---------------------------------------------------------------------------
echo "[2/6] Preparing annotated GeoJSON with wheat labels and background rings..."

python3 << PYEOF
import json
import math
from pathlib import Path
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform
import pyproj

SRC = Path("${GISDATA_DIR}/Sharjah_Wheat_fields_4326.geojson")
OUT = Path("${PROJECT_PATH}/wheat_fields_annotated.geojson")

with open(SRC) as f:
    fc = json.load(f)

# Growing seasons to include (Nov–Apr covers emergence → harvest in UAE)
seasons = [
    ("2022-11-01T00:00:00+00:00", "2023-04-30T00:00:00+00:00"),
    ("2023-11-01T00:00:00+00:00", "2024-04-30T00:00:00+00:00"),
    ("2024-11-01T00:00:00+00:00", "2025-04-30T00:00:00+00:00"),
]

features_out = []
field_id = 1

# WGS84 <-> UTM 40N projector for metric buffering
wgs84 = pyproj.CRS("EPSG:4326")
utm40n = pyproj.CRS("EPSG:32640")
to_utm = pyproj.Transformer.from_crs(wgs84, utm40n, always_xy=True).transform
to_wgs = pyproj.Transformer.from_crs(utm40n, wgs84, always_xy=True).transform

for feat in fc["features"]:
    geom_wgs = shape(feat["geometry"])
    geom_utm = shp_transform(to_utm, geom_wgs)
    # Ring buffer: 50m gap then 300m width — provides background context
    inner = geom_utm.buffer(50)
    outer = geom_utm.buffer(350)
    ring_utm = outer.difference(inner)
    ring_wgs = shp_transform(to_wgs, ring_utm)

    for start_t, end_t in seasons:
        # Wheat field polygon
        features_out.append({
            "type": "Feature",
            "geometry": mapping(geom_wgs),
            "properties": {
                "field_id": field_id,
                "label": "wheat",
                "start_time": start_t,
                "end_time": end_t,
                "Diam_km": feat["properties"].get("Diam_km"),
                "x_coord": feat["properties"].get("x_coord"),
                "y_coord": feat["properties"].get("y_coord"),
            }
        })

        # Background ring polygon
        features_out.append({
            "type": "Feature",
            "geometry": mapping(ring_wgs),
            "properties": {
                "field_id": field_id + 10000,
                "label": "background",
                "start_time": start_t,
                "end_time": end_t,
                "Diam_km": feat["properties"].get("Diam_km"),
                "x_coord": feat["properties"].get("x_coord"),
                "y_coord": feat["properties"].get("y_coord"),
            }
        })

    field_id += 1

out_fc = {"type": "FeatureCollection", "features": features_out}
OUT.write_text(json.dumps(out_fc, indent=2))
print(f"  Written {len(features_out)} annotated features "
      f"({len(features_out)//2} wheat + {len(features_out)//2} background) "
      f"across {len(seasons)} seasons to {OUT}")
PYEOF

# ---------------------------------------------------------------------------
# 4. Run oer_annotation_creation.py
# ---------------------------------------------------------------------------
echo "[3/6] Running oer_annotation_creation.py..."

python "${REPO_DIR}/scripts/oer_annotation_creation.py" \
    "${PROJECT_PATH}/wheat_fields_annotated.geojson" \
    --id-col field_id \
    --start-col start_time \
    --end-col end_time \
    --label-cols label \
    --buffer 200 \
    --outdir "${PROJECT_PATH}"

echo "   annotation_features.geojson and annotation_task_features.geojson written."

# ---------------------------------------------------------------------------
# 5. Prepare labeled windows (rslearn dataset creation)
# ---------------------------------------------------------------------------
echo "[4/6] Preparing labeled windows (rasterizes polygons → label rasters)..."

python -m olmoearth_projects.main olmoearth_run prepare_labeled_windows \
    --project_path "${PROJECT_PATH}" \
    --scratch_path "${ALMA_SCRATCH}"

# ---------------------------------------------------------------------------
# 6. Fetch Sentinel-2 imagery from Planetary Computer
# ---------------------------------------------------------------------------
echo "[5/6] Building dataset from windows (fetches Sentinel-2 from Planetary Computer)..."
echo "      This may take 30–90 minutes depending on network speed."

python -m olmoearth_projects.main olmoearth_run build_dataset_from_windows \
    --project_path "${PROJECT_PATH}" \
    --scratch_path "${ALMA_SCRATCH}"

# ---------------------------------------------------------------------------
# 7. Report split counts
# ---------------------------------------------------------------------------
echo "[6/6] Dataset split summary:"
GROUP="random_split"
find "${ALMA_SCRATCH}/dataset/windows/${GROUP}" -maxdepth 2 -name "metadata.json" \
    -exec cat {} \; 2>/dev/null \
    | grep -oE '"split": *"[^"]+"' \
    | sort | uniq -c \
    || echo "  (No windows found — check ${ALMA_SCRATCH}/dataset/)"

echo ""
echo "=== Import complete ==="
echo "Dataset location : ${ALMA_SCRATCH}"
echo "Project config   : ${PROJECT_PATH}"
echo ""
echo "Next step: bash ALMA_Train_Wheat_Festival.sh [${ALMA_SCRATCH}]"
