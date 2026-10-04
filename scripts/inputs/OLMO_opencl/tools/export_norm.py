#!/usr/bin/env python3
# SPDX-License-Identifier: Unlicense
"""Export OlmoEarth's Sentinel-2 L2A normalisation constants for olmo_cl.

Reads olmoearth_pretrain's computed.json (the file behind
olmoearth_pretrain.data.normalize.load_computed_config()) and writes one
line per band, in OlmoEarth band order: band, mean, std. olmo_cl applies
rslearn's OlmoEarthNormalize: (x - (mean - 2 std)) / (4 std).

Usage:
    export_norm.py [COMPUTED_JSON] > data/s2_norm.tsv

Without an argument, the file is located through the installed
olmoearth_pretrain package.
"""

import json
import sys

BANDS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12",
         "B01", "B09"]


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            config = json.load(f)
    else:
        from olmoearth_pretrain.data.normalize import load_computed_config

        config = load_computed_config()
    s2 = config["sentinel2_l2a"]
    missing = [b for b in BANDS if b not in s2]
    if missing or len(s2) != len(BANDS):
        sys.exit(f"sentinel2_l2a bands {sorted(s2)} do not match {BANDS}")
    print("# band\tmean\tstd")
    for band in BANDS:
        print(f"{band}\t{s2[band]['mean']!r}\t{s2[band]['std']!r}")


if __name__ == "__main__":
    main()
