"""Train the real-cow model (predicted rise in the herd's average body temperature) from data_realcow/mmcows_10min.csv.

Usage:  python train_realcow.py      (on the Pi; takes a minute or two)
Output: models/realcow_model.joblib, used by controller.py to choose fan speed (steady-state prediction)
        and for the dashboard's last-hour estimate.

The fans serve a whole zone, so the model is trained on the HERD average: every 10 minutes, the inputs and the body
temperature rise are averaged over the cows. Tested two ways on data the model never saw: 5-fold cross-validation
by day (every day tested once), and a time-ordered 80/20 split (train on the first 80%, predict the last 20%, as in
real use).
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from cowtrol_common import path
from realcow import FEATURES, HISTORY_H, MONOTONIC, PER_HOUR, feature_frame

MIN_COWS = 8                     # a time step counts for the herd average only if at least this many cows have data


def load():
    """One row per cow per 10 minutes."""
    df = pd.read_csv(path("data_realcow/mmcows_10min.csv"))
    df.index = pd.to_datetime(df["time"], unit="s")
    rows = []
    for cow, d in df.groupby("cow"):
        d = d.asfreq("10min")
        hour = d.index.tz_localize("UTC").tz_convert("America/Chicago")          # barn local time (Wisconsin)
        f = feature_frame(d["thi"], d["clustering"], d["lying"], pd.Series(hour.hour + hour.minute / 60, index=d.index))
        f["rise"], f["cow"], f["day"] = d["rise"], cow, hour.date.astype(str)
        rows.append(f.iloc[HISTORY_H * PER_HOUR:])            # skip the first hour (incomplete 1 h averages)
    return pd.concat(rows).dropna()


def herd(cows):
    """One row per 10 minutes: inputs and rise averaged over the cows."""
    g = cows.groupby(level=0)
    h = g[FEATURES + ["rise"]].mean()
    h["day"], h["n_cows"] = g["day"].first(), g.size()
    return h[h["n_cows"] >= MIN_COWS]


def lying_by_thi(data):
    """Typical share of time lying per THI band, for the standing-at-high-THI flag."""
    bins = np.array([0, 65, 68, 72, 75, 78])
    band = np.searchsorted(bins, data["thi_now"].values, side="right") - 1
    return bins, [float(data["lying_1h"].values[band == i].mean()) for i in range(len(bins))]


def xgb():
    """XGBoost (gradient-boosted trees): 300 trees of depth 3, each correcting the errors of the ones before.
    Monotonic: more THI, clustering or lying never predicts a LOWER body temperature (realcow.MONOTONIC), so a
    faster fan (lower effective THI) never predicts a hotter herd. Time of day is learned freely."""
    return XGBRegressor(max_depth=3, learning_rate=0.05, n_estimators=300, min_child_weight=30, subsample=0.8,
                        random_state=0, n_jobs=-1, monotone_constraints=tuple(MONOTONIC.get(c, 0) for c in FEATURES))


def cv(data, k):
    """Train on some days, predict the others; repeat so every day is predicted once."""
    y, p = data["rise"].values, np.zeros(len(data))
    for tr, te in GroupKFold(k).split(data, y, data["day"]):
        p[te] = xgb().fit(data[FEATURES].values[tr], y[tr]).predict(data[FEATURES].values[te])
    return mean_absolute_error(y, p), r2_score(y, p)


def split_80_20(data):
    """Train on the first 80% of the timeline, predict the last 20% (later days, as in real use)."""
    data = data.sort_index()
    n = int(len(data) * 0.8)
    tr, te = data.iloc[:n], data.iloc[n:]
    p = xgb().fit(tr[FEATURES].values, tr["rise"].values).predict(te[FEATURES].values)
    return mean_absolute_error(te["rise"], p), r2_score(te["rise"], p)


def main():
    if not os.path.exists(path("data_realcow/mmcows_10min.csv")):
        sys.exit("data_realcow/mmcows_10min.csv not found (make it with realcow_prep.py).")
    cows = load()
    data = herd(cows)
    print(f"{len(data)} herd time steps ({cows.cow.nunique()} cows, {data['day'].nunique()} days)")
    mae, r2 = cv(data, 5)
    print(f"Tested on unseen days (5-fold cross-validation): R2 {r2:.2f}, MAE {mae:.3f} C")
    mae, r2 = split_80_20(data)
    print(f"Tested on later days (train first 80%, test last 20%): R2 {r2:.2f}, MAE {mae:.3f} C")

    model = xgb().fit(data[FEATURES].values, data["rise"].values)
    bundle = {"model": model, "features": FEATURES, "target": "herd average",
              "max_cluster": float(data["cluster_1h"].quantile(0.99)),
              "lying_by_thi": lying_by_thi(cows),
              "model_name": "XGBoost (monotonic), herd average", "trained_rows": len(data),
              "source": "MmCows 2023 (10 cows, 14 days, herd average)"}
    os.makedirs(path("models"), exist_ok=True)
    joblib.dump(bundle, path("models/realcow_model.joblib"))
    print("saved models/realcow_model.joblib")


if __name__ == "__main__":
    main()
