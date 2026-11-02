"""Add lying, outdoor weather (solar, wind, temperature) and milk yield to data_realcow/mmcows_10min.csv.
Run once, on a laptop, after realcow_prep.py.

Usage:  python realcow_extra_prep.py <folder with the MmCows ankle/, weather/ and milk/ folders>
        (from main_data in sensor_data.zip at https://huggingface.co/datasets/neis-lab/mmcows)

Added columns:
  lying      fraction of the 10-minute step the cow was lying (ankle sensor, 0-1)
  solar_wm2  outdoor solar radiation, W/m2 (KWIARLIN5 weather station near the barn)
  wind_m_s   outdoor wind speed, m/s (same station; NOT the air speed inside the barn)
  out_temp_c outdoor air temperature, deg C
  milk_kg    that cow's milk yield for the day (one value per day)
"""
import glob
import os
import sys

import pandas as pd

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # find cowtrol_common.py in the parent folder
from cowtrol_common import path

STEP = "10min"
TZ = "America/Chicago"           # weather sheets and ankle datetimes are local barn time (Wisconsin)


def num(col):
    return pd.to_numeric(col.astype(str).str.extract(r"(-?[\d.]+)")[0], errors="coerce")


def weather(folder):
    out = []
    for f in sorted(glob.glob(os.path.join(folder, "*.xlsx"))):
        day = pd.to_datetime(os.path.basename(f)[:10], format="%m_%d_%Y")
        w = pd.read_excel(f)
        t = (day + pd.to_timedelta(w["Time"].astype(str))).dt.tz_localize(TZ, ambiguous="NaT").dt.tz_convert(None)
        out.append(pd.DataFrame({"solar_wm2": num(w["Solar"]).values,
                                 "wind_m_s": (num(w["Speed"]) * 0.44704).values,
                                 "out_temp_c": ((num(w["Temperature"]) - 32) / 1.8).values}, index=t))
    return pd.concat(out).sort_index().resample(STEP).mean().interpolate(limit=3)


def main(src):
    df = pd.read_csv(path("data_realcow/mmcows_10min.csv"))
    df = df.drop(columns=[c for c in ("lying", "solar_wm2", "wind_m_s", "out_temp_c", "milk_kg") if c in df])
    t = pd.to_datetime(df["time"], unit="s")

    w = weather(os.path.join(src, "weather"))
    df = df.join(w.reindex(t).reset_index(drop=True))

    lying = []
    for cow_dir in sorted(glob.glob(os.path.join(src, "ankle", "C*"))):
        a = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(cow_dir, "*.csv")))])
        a.index = pd.to_datetime(a["timestamp"], unit="s")
        s = a["lying"].resample(STEP).mean().rename("lying").to_frame()
        s["cow"] = os.path.basename(cow_dir)
        s["time"] = (s.index - pd.Timestamp(0)) // pd.Timedelta("1s")
        lying.append(s)
    df = df.merge(pd.concat(lying), on=["time", "cow"], how="left")

    milk = []
    for f in sorted(glob.glob(os.path.join(src, "milk", "C*.csv"))):
        m = pd.read_csv(f)
        day = pd.to_datetime(m["timestamp"], unit="s").dt.tz_localize("UTC").dt.tz_convert(TZ).dt.date
        milk.append(pd.DataFrame({"cow": os.path.basename(f)[:3], "day": day, "milk_kg": m["milk_weight_kg"]}))
    df["day"] = t.dt.tz_localize("UTC").dt.tz_convert(TZ).dt.date
    df = df.merge(pd.concat(milk), on=["cow", "day"], how="left").drop(columns="day")

    df.round(3).to_csv(path("data_realcow/mmcows_10min.csv"), index=False)
    print(f"wrote {len(df)} rows; filled: " +
          ", ".join(f"{c} {df[c].notna().mean():.0%}" for c in ("lying", "solar_wm2", "wind_m_s", "milk_kg")))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
