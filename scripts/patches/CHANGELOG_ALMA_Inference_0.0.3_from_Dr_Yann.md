# Changelog

Changes to the ALMA Wheat Festival scripts in `olmoearth_projects`.

## ALMA_Inference_Wheat_Festival.sh 0.0.3 (2026-10-04)

### Added

- `field_stats.geojson` in `inference_outputs/YYYYMMDD/`, next to
  `field_stats.csv`. It holds one feature per field (37), with the field
  polygon in EPSG:4326 and the same per-field figures as the CSV as
  properties. GIS tools and map-based dashboards can now draw every theme
  directly, without joining the CSV to the boundary file.

  | Property | Theme / map |
  |---|---|
  | `field_id` | 1-based row order of `Sharjah_Wheat_fields_4326.geojson` |
  | `diam_km` | field diameter, copied from the boundary file |
  | `status` | 03 field activation (`active` / `partial` / `inactive`) |
  | `active_frac`, `mean_prob` | 03 field activation, 05 credit score |
  | `uncertainty_frac` | 04 uncertainty (share of pixels at 0.30–0.70) |
  | `area_ha`, `active_area_ha` | 06 insurance exposure |
  | `credit_score` | 05 credit score (0–100) |

  Maps 01 (wheat probability) and 02 (wheat mask) are drawn from the
  probability raster itself; it is still listed in the final report as
  the QGIS layer.

- `ALMA_Inference_Wheat_Festival.md`: the output listing, the property
  table above, and the data-flow diagram mention the GeoJSON.

### Unchanged

- The CSV, the six PNGs, their values, and the inference itself.
- No new dependency: the script already imported `geopandas`.

### Note

`field_id` is the position of a field in
`Sharjah_Wheat_fields_4326.geojson`, not an ID stored in that file. If
the boundary file is reordered or edited, the IDs of a new run no longer
match those of earlier runs. The GeoJSON carries its own polygons, so it
is not affected; joins of the CSV to the boundary file are.

### Applying

From the `olmoearth_projects` root:

```sh
patch -p1 < ALMA_inference_geojson_2026-10-04.diff
```

Tested on the 37 Sharjah fields and the 2026-06-04 run's figures: 37
features, polygons identical to the input boundaries (largest difference
1e-14 degrees after the UTM 40N round trip), all nine properties present.
The full inference was not re-run.
