"""Effective THI (the THI the cows feel) and the air speed map.

The body temperature model was trained on MmCows, which has no air speed or solar data, so both enter as
corrections to THI before the model:

    T_eff   = T - cooling(v, T) [+ heating(n, v) if heat_load.clustering.enabled]
    THI_eff = NRC_THI(T_eff, RH) + solar_term

cooling(v, T) = coef * v^exponent * (body_temp - T)        air speed term of the ETIC index for dairy cattle
                                                           (Wang et al. 2018, J. Therm. Biol.), v in m/s
solar_term    = thi_per_wm2 * solar (W/m2)                 radiation term of Mader et al. (2006) adjusted THI
heating(n, v) = n * q / (rho * cp * A * max(v, v_min))     optional physics estimate of neighbours' body heat
                                                           (off by default: the model learned clustering itself)
All coefficients live in config.yaml -> heat_load.
"""
import csv
import os

import numpy as np

from cowtrol_common import nrc_thi, path

RHO_AIR = 1.2      # kg/m3
CP_AIR = 1005.0    # J/(kg K)


def air_speed_cooling_c(temp_c, v, cfg):
    p = cfg["heat_load"]["air_speed"]
    v = np.maximum(np.asarray(v, dtype=float), 0.0)
    return p["coef"] * v ** p["exponent"] * np.maximum(p["body_temp_c"] - temp_c, 0.0)


def clustering_heating_c(n, v, cfg):
    p = cfg["heat_load"]["clustering"]
    v = np.maximum(np.asarray(v, dtype=float), p["min_air_speed_m_s"])
    if not p.get("enabled", False):
        return 0.0 * v
    return max(float(n), 0.0) * p["sensible_heat_w"] / (RHO_AIR * CP_AIR * p["shared_air_area_m2"] * v)


def effective_thi(temp_c, rh, v, n, solar_wm2, cfg):
    """v may be a number or an array of candidate air speeds (m/s). Returns a dict of THI parts."""
    v = np.asarray(v, dtype=float)
    cool = air_speed_cooling_c(temp_c, v, cfg)
    heat = clustering_heating_c(n, v, cfg)
    solar = cfg["heat_load"]["solar"]["thi_per_wm2"] * max(float(solar_wm2), 0.0)
    thi_ambient = nrc_thi(temp_c, rh) + 0.0 * v
    thi_after_air = nrc_thi(temp_c - cool, rh)
    thi_after_heat = nrc_thi(temp_c - cool + heat, rh)
    return {"thi_ambient": thi_ambient,
            "air_speed_effect": thi_after_air - thi_ambient,       # negative = cooling
            "clustering_effect": thi_after_heat - thi_after_air,   # 0 unless the physics estimate is on
            "solar_effect": solar + 0.0 * v,
            "thi_eff": thi_after_heat + solar}


class AirspeedMap:
    """Measured air speed (m/s) in each quadrant at each fan %, from airspeed_map.csv.

    CSV columns: quadrant, fan_pct, point, air_speed_m_s  (one row per reading; points are averaged).
    Measure each quadrant with both fans of its zone running at that %. A quadrant with fewer than 2 measured
    fan speeds falls back to a straight-line placeholder.
    """

    def __init__(self, cfg):
        m = cfg["airspeed_map"]
        self.placeholder_max = float(m["placeholder_max_air_speed_m_s"])
        self.tables, self.measured = {}, {}
        raw = {}
        f = path(m["file"])
        if os.path.exists(f):
            with open(f, newline="") as fh:
                for row in csv.DictReader(fh):
                    try:
                        q = (row.get("quadrant") or row["area"]).strip()
                        pct = float(row["fan_pct"])
                        val = float(row["air_speed_m_s"])
                    except (KeyError, ValueError, AttributeError):
                        continue                                    # blank or unfilled row
                    raw.setdefault(q, {}).setdefault(pct, []).append(val)
        for q in [q for a in cfg["areas"] for q in a["quadrants"]]:
            pts = raw.get(q, {})
            if len(pts) >= 2:
                pcts = np.array(sorted(pts))
                spd = np.maximum.accumulate(np.array([np.mean(pts[p]) for p in pcts]))   # never decreasing
                self.tables[q], self.measured[q] = (pcts, spd), True
            else:
                self.tables[q] = (np.array([0.0, 100.0]), np.array([0.0, self.placeholder_max]))
                self.measured[q] = False

    def speed(self, quadrants, fan_pct):
        """Average air speed over these quadrants at each fan % (air mixes, so an average, not the worst)."""
        fan_pct = np.asarray(fan_pct, dtype=float)
        return np.mean([np.interp(fan_pct, *self.tables[q]) for q in quadrants], axis=0)

    @property
    def all_measured(self):
        return all(self.measured.values())
