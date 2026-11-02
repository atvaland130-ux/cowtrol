"""Build data_realcow/mmcows_10min.csv from the raw MmCows files. Run once, on a laptop (the raw files are ~40 MB).
The CSV it writes is what train_realcow.py uses, so the Pi only needs that CSV.

Usage:  python realcow_prep.py <folder with the MmCows cbt/, thi/ and uwb/ folders>
        (from sensor_data.zip at https://huggingface.co/datasets/neis-lab/mmcows)

Output columns: time (unix s, start of each 10-minute step), cow, thi (barn average), clustering (other tagged cows
within 2.5 m, averaged over the step), rise (body temperature minus the cow's median when THI < 68, deg C).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # find cowtrol_common.py in the parent folder
from cowtrol_common import path

RADIUS_M = 2.5
STEP = "10min"


def series(f, col):
    d = pd.read_csv(f)
    d.index = pd.to_datetime(d["timestamp"], unit="s")
    return d[col]


def main(src):
    thi = series(os.path.join(src, "thi", "average.csv"), "THI").resample(STEP).mean()

    # cow positions: tags T01-T10 are on cows C01-C10 (T13/T14 are not cows and are skipped)
    tags = [f"T{i:02d}" for i in range(1, 11)]
    pos = {}
    for t in tags:
        u = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(src, "uwb", t, "*.csv")))])
        u.index = pd.to_datetime(u["timestamp"], unit="s")
        pos[t] = u[["coord_x_cm", "coord_y_cm"]].resample("1min").mean() / 100.0
    idx = pos[tags[0]].index
    for t in tags[1:]:
        idx = idx.union(pos[t].index)
    x = np.stack([pos[t]["coord_x_cm"].reindex(idx).values for t in tags], 1)
    y = np.stack([pos[t]["coord_y_cm"].reindex(idx).values for t in tags], 1)
    dist = np.sqrt((x[:, :, None] - x[:, None, :]) ** 2 + (y[:, :, None] - y[:, None, :]) ** 2)
    n = np.nansum((dist < RADIUS_M) & (dist > 0), axis=2).astype(float)
    n[np.isnan(x)] = np.nan
    clus = pd.DataFrame(n, index=idx, columns=[f"C{t[1:]}" for t in tags]).resample(STEP).mean()

    out = []
    for f in sorted(glob.glob(os.path.join(src, "cbt", "C*.csv"))):
        cow = os.path.basename(f)[:3]
        cbt = series(f, "temperature_C")
        cbt = cbt.where(cbt > 37).resample(STEP).mean()       # drop sensor glitches (e.g. 21.9 C)
        base = cbt[thi.reindex(cbt.index) < 68].median()
        d = pd.DataFrame({"thi": thi, "clustering": clus.get(cow), "rise": cbt - base})
        d["cow"] = cow
        d["time"] = (d.index - pd.Timestamp(0)) // pd.Timedelta("1s")
        out.append(d.dropna(subset=["thi"]))
    df = pd.concat(out)[["time", "cow", "thi", "clustering", "rise"]].round(3)
    os.makedirs(path("data_realcow"), exist_ok=True)
    df.to_csv(path("data_realcow/mmcows_10min.csv"), index=False)
    print(f"wrote data_realcow/mmcows_10min.csv: {len(df)} rows, {df.cow.nunique()} cows")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
