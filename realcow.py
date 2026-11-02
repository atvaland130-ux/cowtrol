"""Real-cow model: predicted rise in the HERD's average body temperature (CBT) from THI, clustering, lying and the
time of day.

Trained on MmCows (10 cows, one barn, 14 days in 2023): measured body temperature, barn THI, cow positions
(UWB tags: clustering) and ankle-sensor lying. The target is the average over the cows of each cow's body
temperature above her own normal (her median when THI < 68), deg C, every 10 minutes. The fans serve a whole zone,
so the model predicts the zone's average cow, not one cow: one cow's temperature jumps with every drink or meal,
and those jumps mostly cancel out across the herd (R2 0.53 for single cows vs 0.78 for the herd average).
Walking and drinking (realcow_uwb_prep.py) were tried: they help for single cows but add nothing for the herd.

steady_state_many(): "CBT rise if current conditions hold" (the current inputs fed as if they had lasted 1 h),
one prediction per candidate fan speed. This is what sets fan speed.

Inputs, built the same way for training (train_realcow.py) and on the Pi, on a 10-minute grid, averaged over the
cows (on the Pi: one value per zone):
  thi_now    = current THI (past THI averages over 1-12 h were tried: no gain once time of day is in, since time
             of day already tells the model where the barn is in its daily heat cycle),
  cluster_1h = average number of other cows within 2.5 m over the past hour,
  lying_1h   = share of the past hour lying (camera: share of cows lying; MmCows: ankle sensor),
  hour_sin, hour_cos = local time of day (body temperature has a daily rhythm, highest in the evening; the
             biggest input after THI: R2 0.63 without it).
In MmCows a lying cow runs ~0.1-0.2 C hotter at the same THI (lying traps heat), and cows stand more as THI rises.
So lying raises the predicted CBT, and unusual standing at high THI is flagged separately (standing_flag()).
THI, clustering and lying are monotonic (more never predicts a lower CBT); time of day is learned freely.
On the benchtop, the clock can be overridden on the dashboard for demos.
"""
import os
import time

import joblib
import numpy as np
import pandas as pd

from cowtrol_common import path

PER_HOUR = 6
FEATURES = ["thi_now", "cluster_1h", "lying_1h", "hour_sin", "hour_cos"]
# +1: more never predicts a LOWER body temperature; time of day is learned freely.
MONOTONIC = {"thi_now": 1, "cluster_1h": 1, "lying_1h": 1}
HISTORY_H = 1                    # clustering and lying are averaged over the past hour


def hour_parts(hour):
    """Local hour (0-24, float or array) -> (sin, cos), so 23:50 and 00:10 are close."""
    a = 2 * np.pi * np.asarray(hour, dtype=float) / 24
    return np.sin(a), np.cos(a)


def feature_frame(thi, cluster, lying, hour):
    """All inputs: pandas Series on a regular 10-minute index; hour = local hour (0-24).
    Returns one row of FEATURES per time step.
    Windows use whatever history exists (min_periods=1), so the Pi can predict before it has 1 h."""
    f = pd.DataFrame({"thi_now": thi})
    f["cluster_1h"] = cluster.rolling(PER_HOUR, min_periods=1).mean()
    f["lying_1h"] = lying.rolling(PER_HOUR, min_periods=1).mean()
    f["hour_sin"], f["hour_cos"] = hour_parts(hour)
    return f[FEATURES]


def steady_features(thi, clustering, lying, hour):
    """FEATURES for 'these conditions have held for the past hour', at this time of day (local hour)."""
    hs, hc = hour_parts(hour)
    return [thi, clustering, lying, float(hs), float(hc)]


def local_hour(t=None):
    lt = time.localtime(time.time() if t is None else t)
    return lt.tm_hour + lt.tm_min / 60


class RealCow:
    """Loads the trained model and predicts."""

    def __init__(self, cfg):
        rc = cfg.get("realcow", {})
        self.flag_cfg = rc.get("standing_flag", {})
        self.bundle = None
        mpath = path(rc.get("model_path", "models/realcow_model.joblib"))
        if os.path.exists(mpath):
            self.bundle = joblib.load(mpath)
        self.max_cluster = (self.bundle or {}).get("max_cluster")

    @property
    def ready(self):
        return self.bundle is not None

    def _clip(self, clustering):
        return min(clustering, self.max_cluster) if self.max_cluster else clustering

    def steady_state_many(self, thi_list, z):
        """Predicted CBT rise (deg C) for each THI in thi_list (already corrected for air speed and solar) if it and
        this zone's clustering and lying share held for 1 h, at z["hour"]. One THI per candidate fan speed."""
        x = np.array([steady_features(t, self._clip(z["clustering"]), z["lying"], z["hour"]) for t in thi_list])
        return self.bundle["model"].predict(x)

    def standing_flag(self, thi, lying):
        """Heat-stress sign: fewer cows lying than MmCows cows did at this THI. Returns a message or None."""
        if not self.ready or lying is None or thi < self.flag_cfg.get("min_thi", 72):
            return None
        bins, exp = self.bundle["lying_by_thi"]
        i = min(max(int(np.searchsorted(bins, thi, side="right")) - 1, 0), len(exp) - 1)
        if lying < exp[i] - self.flag_cfg.get("margin", 0.2):
            return (f"Cows standing more than usual for THI {thi:.0f} "
                    f"({lying:.0%} lying vs {exp[i]:.0%} typical): possible heat stress")
        return None
