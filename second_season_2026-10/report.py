import os, pandas as pd, geopandas as gpd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = os.path.expanduser("~/second_season")
df = pd.read_csv(D + "/ndvi_timeseries.csv", parse_dates=["date"])
f = gpd.read_file(os.path.expanduser("~/GISDATA/Sharjah_WheatFestival/Sharjah_Wheat_fields_4326.geojson"))
f["fid"] = range(1, len(f) + 1)
area = dict(zip(f.fid, f.to_crs(32640).area / 1e4))
rows = []
for y in (2024, 2025, 2026):
    s = df[(df.date >= f"{y}-05-01") & (df.date <= f"{y}-09-15")]     # after harvest, before sowing
    ring = s[s.id == 0]
    for fid, g in s[s.id > 0].groupby("id"):
        gr = g[g.ndvi > 0.3]
        rows.append(dict(year=y, field=fid, area_ha=round(area[fid], 1), scenes=len(g),
            max_ndvi=round(g.ndvi.max(), 2), green_scenes=len(gr),
            first_green=gr.date.min().date() if len(gr) else "", last_green=gr.date.max().date() if len(gr) else "",
            peak_green_ha=round(g.green_frac.max() * area[fid], 1),
            desert_max_ndvi=round(ring.ndvi.max(), 2) if len(ring) else None))
t = pd.DataFrame(rows); t.to_csv(D + "/summer_by_field.csv", index=False)
print("Fields green (NDVI > 0.3) in at least 2 clear scenes between 1 May and 15 Sep:")
print(t[t.green_scenes >= 2].to_string(index=False))
print("\nDesert ring, max NDVI per summer:", t.groupby("year").desert_max_ndvi.first().to_dict())
print("Clear scenes per summer:", t.groupby("year").scenes.max().to_dict())
fig, axs = plt.subplots(7, 6, figsize=(26, 24), sharex=True, sharey=True)
r = df[df.id == 0]
for ax, fid in zip(axs.flat, range(1, 38)):
    g = df[df.id == fid]
    ax.plot(r.date, r.ndvi, color="grey", lw=.7, label="desert ring")
    ax.plot(g.date, g.ndvi, ".-", ms=3, lw=.8, color="tab:green", label="field")
    for y in (2024, 2025, 2026):
        ax.axvspan(pd.Timestamp(f"{y}-05-01"), pd.Timestamp(f"{y}-09-15"), color="orange", alpha=.15)
    ax.axhline(.3, ls=":", color="k", lw=.6); ax.set_ylim(-.05, 1); ax.set_title(f"field {fid}")
for ax in axs.flat[37:]: ax.axis("off")
axs.flat[0].legend(fontsize=8)
fig.suptitle("Mleiha fields, Sentinel-2 NDVI Nov 2023 - Oct 2026 (orange = summer, 1 May - 15 Sep)", fontsize=16)
fig.autofmt_xdate(); fig.tight_layout(rect=[0, 0, 1, .97]); fig.savefig(D + "/ndvi_all_fields.png", dpi=80)
print("wrote", D + "/summer_by_field.csv and ndvi_all_fields.png")
