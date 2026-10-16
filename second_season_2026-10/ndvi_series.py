import os, sys
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "5")
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.windows import from_bounds
from rasterio.features import rasterize
from rasterio.enums import Resampling
import pystac_client, planetary_computer as pc

start, end, out = sys.argv[1:4]
f = gpd.read_file(os.path.expanduser("~/GISDATA/Sharjah_WheatFestival/Sharjah_Wheat_fields_4326.geojson"))
f["fid"] = range(1, len(f) + 1)
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
items = list(cat.search(collections=["sentinel-2-l2a"], bbox=list(f.total_bounds),
                        datetime=f"{start}/{end}", query={"eo:cloud_cover": {"lt": 40}}).items())
items.sort(key=lambda i: i.datetime)
print(len(items), "scenes", flush=True)
BAD_SCL = [0, 1, 3, 8, 9, 10]   # nodata, saturated, shadow, cloud med/high, cirrus
rows, seen = [], set()
for n, it in enumerate(items, 1):
    d = it.datetime.strftime("%Y-%m-%d")
    tile = it.properties.get("s2:mgrs_tile", "")
    try:
        it = pc.sign(it)
        off = 1000 if float(it.properties.get("s2:processing_baseline", "0")) >= 4 else 0
        with rasterio.open(it.assets["B04"].href) as r4, rasterio.open(it.assets["B08"].href) as r8, \
             rasterio.open(it.assets["SCL"].href) as rs:
            fu = f.to_crs(r4.crs)
            u = fu.geometry.unary_union
            ring = u.buffer(1000).difference(u.buffer(100))
            b = ring.bounds
            win = from_bounds(*b, r4.transform).round_offsets().round_lengths()
            red_raw = r4.read(1, window=win, boundless=True, fill_value=0).astype(float)
            nir_raw = r8.read(1, window=win, boundless=True, fill_value=0).astype(float)
            scl = rs.read(1, window=from_bounds(*b, rs.transform), out_shape=red_raw.shape,
                          boundless=True, fill_value=0, resampling=Resampling.nearest)
            tr = r4.window_transform(win)
        red, nir = red_raw - off, nir_raw - off
        ok = (red_raw > 0) & (nir_raw > 0) & ~np.isin(scl, BAD_SCL) & ((red + nir) > 0)
        ndvi = np.where(ok, (nir - red) / np.where((red + nir) > 0, red + nir, 1), np.nan)
        shapes = [(ring, 0)] + list(zip(fu.geometry, fu.fid))
        lab = rasterize(shapes, out_shape=red_raw.shape, transform=tr, fill=-1, dtype="int32")
        for i in [0] + list(fu.fid):
            if (i, d) in seen: continue
            m = lab == i; tot = m.sum(); v = m & ok
            if tot == 0 or v.sum() < 0.8 * tot: continue      # outside scene or >20 % cloud
            x = ndvi[v]
            rows.append(dict(date=d, tile=tile, id=i, ndvi=round(float(np.median(x)), 3),
                             green_frac=round(float((x > 0.3).mean()), 3), valid_px=int(v.sum())))
            seen.add((i, d))
    except Exception as e:
        print("skip", d, tile, type(e).__name__, e, flush=True)
    if n % 10 == 0:
        print(f"{n}/{len(items)} scenes done", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
pd.DataFrame(rows).to_csv(out, index=False)
print("wrote", out, len(rows), "rows")
