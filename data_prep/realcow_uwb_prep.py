"""Add movement and time at the water trough to data_realcow/mmcows_10min.csv, from the MmCows UWB position tags.
Run once, on a laptop, after realcow_prep.py (and realcow_extra_prep.py).

Usage:  python realcow_uwb_prep.py <folder with the MmCows uwb/ folder>
        (uwb.zip from main_data in sensor_data.zip at https://huggingface.co/datasets/neis-lab/mmcows)

Added columns (per cow, per 10-minute step):
  move_m      metres walked: one position per minute (median of the tag fixes), steps under 30 cm ignored (tag jitter)
  trough_min  minutes spent in a water-trough zone

Why: drinking cold water drops body temperature for up to ~1.5 h, and walking cows are not lying. Both help predict
ONE cow's body temperature (+0.03 R2), but add nothing for the herd average the fans are controlled on, so the
current model does not use them. Kept for testing (both are things the camera could see).

Trough zones: MmCows does not mark the water troughs. Positions 10-40 min before sudden body temperature drops
(> 0.4 C in 20 min) are 4-8x more common in two spots, one at each end of the pen, which is where we take the
troughs to be (TROUGHS below, cm in the MmCows barn coordinates).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # find cowtrol_common.py in the parent folder
from cowtrol_common import path

STEP = "10min"
JITTER_M = 0.3
TROUGHS = [((912, 1100), (-121, 79)), ((-1100, -888), (-121, 79))]     # ((x min, x max), (y min, y max)), cm


def tag(folder, t):
    d = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(folder, t, "*.csv"))))
    d.index = pd.to_datetime(d["timestamp"], unit="s")
    m = d.sort_index()[["coord_x_cm", "coord_y_cm"]].resample("1min").median()
    step = np.hypot(m["coord_x_cm"].diff(), m["coord_y_cm"].diff()) / 100
    step = step.where(step > JITTER_M, 0).where(step.notna())
    at = pd.Series(False, index=m.index)
    for (x0, x1), (y0, y1) in TROUGHS:
        at |= m["coord_x_cm"].between(x0, x1) & m["coord_y_cm"].between(y0, y1)
    at = at.astype(float).where(m["coord_x_cm"].notna())
    out = pd.DataFrame({"move_m": step.resample(STEP).sum(min_count=3),
                        "trough_min": at.resample(STEP).sum(min_count=3)})
    out["time"] = (out.index - pd.Timestamp(0)) // pd.Timedelta("1s")
    return out


def main(src):
    df = pd.read_csv(path("data_realcow/mmcows_10min.csv"))
    df = df.drop(columns=[c for c in ("move_m", "trough_min") if c in df])
    rows = []
    for i in range(1, 11):                      # tags T01-T10 are on cows C01-C10 (T13/T14 are not cows)
        u = tag(os.path.join(src, "uwb"), f"T{i:02d}")
        u["cow"] = f"C{i:02d}"
        rows.append(u)
    df = df.merge(pd.concat(rows), on=["time", "cow"], how="left")
    df.round(3).to_csv(path("data_realcow/mmcows_10min.csv"), index=False)
    print(f"wrote {len(df)} rows; filled: move_m {df['move_m'].notna().mean():.0%}, "
          f"trough_min {df['trough_min'].notna().mean():.0%}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
