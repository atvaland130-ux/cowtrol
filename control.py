"""Decision pipeline (no hardware). Used by controller.py (live) and the dashboard check page.

Each zone (config.yaml -> areas) is one side of the barn: 3 quadrants (each with a temperature/humidity sensor)
and 2 fans that run together. For each zone:
  Steps 1-5 run for EACH quadrant with cows, on its own sensor and its own measured air speed:
  1. Ambient temperature/humidity from the quadrant's sensor; for every possible fan speed, the measured air speed
     in that quadrant.
  2. Effective THI = ambient THI - air speed cooling + solar   (heatload.py)
  3. Body temperature model (realcow.py): predicted CBT rise if these conditions held, from effective THI,
     the zone's clustering, share of cows lying and the time of day.
  4. Pick the LOWEST fan speed (not below the goal's minimum) whose predicted rise is under the goal's target
     (100 % if no speed gets there). A zone with no cows: fans off.
  5. Hard safety rules (always win; for an empty zone only those in config.yaml -> empty_zone).
  The zone's fans then run at the speed its HOTTEST occupied quadrant needs, so no group of cows is under-cooled
  (under-cooling costs milk, fertility and welfare; over-cooling only costs fan energy).
"""
from datetime import datetime

import numpy as np

from heatload import effective_thi
from realcow import local_hour


def sensor_ids(cfg):
    return sorted(set(cfg["quadrant_sensors"].values()))


def side_of(cfg, quadrant):
    return next(a["name"] for a in cfg["areas"] if quadrant in a["quadrants"])


def n_fans(cfg):
    return max(f for a in cfg["areas"] for f in a["fans"])


def zone_inputs(cfg, manual, camera=None):
    """Per-zone model inputs. manual = dashboard/config values; camera = {zone: {...}} from the camera when it
    is connected (it wins over manual for what it reports). Returns {zone: {clustering, lying, hour, quadrants,
    cows_present, solar_wm2}}. hour = manual clock_hour if set, else the Pi's clock."""
    hour = manual.get("clock_hour")
    hour = local_hour() if hour is None else float(hour)
    out = {}
    for a in cfg["areas"]:
        cam = (camera or {}).get(a["name"], {})
        occupied = cam.get("quadrants", [q for q in a["quadrants"] if q in manual["occupied_quadrants"]])
        out[a["name"]] = {
            "clustering": float(cam.get("clustering", manual["clustering"])),
            "lying": float(cam.get("lying", manual["lying"])),
            "hour": hour,
            "quadrants": occupied or list(a["quadrants"]),       # no cows reported -> whole zone
            "cows_present": bool(occupied),
            "solar_wm2": float(manual["solar_wm2"]),
        }
    return out


def check_hour(clock_hour):
    """None if fine, else an error message (dashboard clock override)."""
    if clock_hour is not None and not 0 <= clock_hour < 24:
        return "clock hour 0-23.99 (or blank for the real clock)"
    return None


def resolve_readings(cfg, sensor_values):
    """Give every quadrant a temperature and humidity.

    sensor_values = {sensor id: (temp_c, rh)} for the sensors that are answering.
    A quadrant whose sensor is not answering borrows the average of the working sensors on its side
    (or of all working sensors if none on its side work).
    Returns ({quadrant: {"temp_c", "rh", "sensor", "borrowed"}}, warnings), or (None, warnings) if none answer.
    """
    qs = cfg["quadrant_sensors"]
    up = {k: v for k, v in sensor_values.items() if v is not None}
    if not up:
        return None, ["No sensor is answering"]
    out, warnings = {}, []
    for q, sid in qs.items():
        if sid in up:
            out[q] = {"temp_c": float(up[sid][0]), "rh": float(up[sid][1]), "sensor": sid, "borrowed": False}
            continue
        side = [qs[o] for o in qs if side_of(cfg, o) == side_of(cfg, q) and qs[o] in up]
        src = sorted(set(side)) or sorted(up)
        out[q] = {"temp_c": float(np.mean([up[i][0] for i in src])), "rh": float(np.mean([up[i][1] for i in src])),
                  "sensor": sid, "borrowed": True}
        warnings.append(f"{q}: sensor {sid} is not answering, using the average of sensor(s) "
                        f"{', '.join(map(str, src))}")
    return out, warnings


def reading_valid(cfg, temp_c, rh):
    lo_t, hi_t = cfg["safety"]["valid_temp_c"]
    lo_h, hi_h = cfg["safety"]["valid_rh_pct"]
    return lo_t <= temp_c <= hi_t and lo_h <= rh <= hi_h


def candidate_duties(cfg):
    return np.concatenate([[0.0], np.arange(cfg["fans"]["min_duty_pct"], 101, 1.0)])


def clamp_goal(cfg, target_rise_c, min_duty_pct):
    lim = cfg["goal"]["limits"]
    t = float(np.clip(target_rise_c, *lim["target_rise_c"]))
    m = float(np.clip(min_duty_pct, *lim["min_duty_pct"]))
    return {"target_rise_c": round(t, 3), "min_duty_pct": round(m, 1)}


def rise_table(model, cfg, amap, temp_c, rh, z, duties=None):
    """Air speed, THI parts and predicted CBT rise for each candidate fan speed in one zone."""
    duties = candidate_duties(cfg) if duties is None else np.asarray(duties, dtype=float)
    v = amap.speed(z["quadrants"], duties)
    parts = effective_thi(temp_c, rh, v, z["clustering"], z["solar_wm2"], cfg)
    rise = model.steady_state_many(parts["thi_eff"], z)
    return {"duty": duties, "air_speed": v, "rise": rise, **parts}


def decide_area(model, cfg, amap, area, temp_c, rh, z, goal):
    s = cfg["safety"]
    tab = rise_table(model, cfg, amap, temp_c, rh, z)
    thi = float(tab["thi_ambient"][0])
    overrides, last = [], len(tab["duty"]) - 1

    # 4. lowest fan speed (at or above the goal's minimum) whose predicted rise is under the target
    allowed = tab["duty"] >= goal["min_duty_pct"] - 1e-9
    ok = allowed & (tab["rise"] <= goal["target_rise_c"])
    reached = bool(ok.any())
    ml_i = int(np.argmax(ok)) if reached else last
    rules = "all"
    if not z["cows_present"]:                      # nobody to cool
        ml_i, rules = 0, cfg.get("empty_zone", {}).get("thi_rules", "none")
    i = ml_i

    # 5. hard safety rules
    if rules == "all" and thi >= s["stress_onset_thi"]:
        enough = tab["air_speed"] >= s["min_air_speed_m_s"] - 1e-9
        if not enough.any():
            if i != last:
                overrides.append(f"THI {thi:.1f} >= {s['stress_onset_thi']}: fans cannot reach "
                                 f"{s['min_air_speed_m_s']} m/s in zone {area['name']}, running 100%")
            i = last
        elif not enough[i]:
            i = int(np.argmax(enough))
            overrides.append(f"THI {thi:.1f} >= {s['stress_onset_thi']}: air speed raised to the "
                             f"{s['min_air_speed_m_s']} m/s minimum")
    if rules in ("all", "emergency") and thi >= s["emergency_thi"] and tab["duty"][i] < 100:
        i = last
        overrides.append(f"THI {thi:.1f} >= {s['emergency_thi']} (emergency): fans forced to 100%")

    warnings = []
    flag = model.standing_flag(thi, z["lying"]) if z["cows_present"] else None
    if flag:
        warnings.append(f"[{area['name']}] {flag}")
    return {
        "area": area["name"], "fans": area["fans"],
        "quadrants_used": z["quadrants"], "cows_present": z["cows_present"],
        "clustering": round(z["clustering"], 2), "lying": round(z["lying"], 2), "solar_wm2": round(z["solar_wm2"]),
        "hour": round(z["hour"], 2),
        "fan_duty_pct": float(tab["duty"][i]), "ml_duty_pct": float(tab["duty"][ml_i]),
        "air_speed_m_s": round(float(tab["air_speed"][i]), 2),
        "thi_ambient": round(thi, 2),
        "air_speed_effect": round(float(tab["air_speed_effect"][i]), 2),
        "solar_effect": round(float(tab["solar_effect"][i]), 2),
        "clustering_effect": round(float(tab["clustering_effect"][i]), 2),
        "thi_eff": round(float(tab["thi_eff"][i]), 2),
        "rise_c": round(float(tab["rise"][i]), 3),
        "rise_fan_off_c": round(float(tab["rise"][0]), 3),
        "rise_fan_max_c": round(float(tab["rise"][-1]), 3),
        "target_rise_c": goal["target_rise_c"], "min_duty_pct": goal["min_duty_pct"],
        "target_reached": bool(tab["rise"][i] <= goal["target_rise_c"]) or not z["cows_present"],
        "standing_flag": flag,
        "air_speed_measured": all(amap.measured[q] for q in z["quadrants"]),
        "overrides": overrides, "warnings": warnings,
    }


def decide(model, cfg, amap, goal, readings, inputs, stale=False, warnings=None):
    """readings = output of resolve_readings (one per quadrant); inputs = output of zone_inputs.
    goal = {"target_rise_c", "min_duty_pct", ...}. stale=True -> no sensor answering: failsafe.
    A fan listed in several zones runs at the highest speed any of them needs."""
    nf = n_fans(cfg)
    rec = {"timestamp": datetime.now().isoformat(timespec="seconds"), "goal": dict(goal),
           "calibrated": amap.all_measured, "overrides": [], "warnings": list(warnings or [])}

    if stale or not model.ready:
        why = (f"No valid sensor readings for {cfg['safety']['stale_data_s']} s" if stale
               else "No body temperature model (run train_realcow.py)")
        rec.update(areas=[], rise_c=None, thi_eff=None, fan_duty_pct=[100.0] * nf, safety_status="Safe with Override")
        rec["overrides"].append(f"{why} -> fans 100% (failsafe)")
        return rec

    results = []
    for a in cfg["areas"]:
        z = inputs[a["name"]]
        best = None
        for q in z["quadrants"]:                   # each quadrant with cows on its own; the hottest sets the fans
            rd = readings[q]
            r = decide_area(model, cfg, amap, a, rd["temp_c"], rd["rh"], dict(z, quadrants=[q]), goal)
            r.update(temp_c=round(rd["temp_c"], 2), rh=round(rd["rh"], 1), sensor=rd["sensor"],
                     sensor_borrowed=rd["borrowed"], hottest_quadrant=q)
            if best is None or (r["fan_duty_pct"], r["rise_fan_off_c"]) > (best["fan_duty_pct"], best["rise_fan_off_c"]):
                best = r
        best.update(quadrants_used=z["quadrants"],
                    air_speed_measured=all(amap.measured[q] for q in z["quadrants"]))
        results.append(best)

    duties = [0.0] * nf
    for r in results:
        for f in r["fans"]:
            duties[f - 1] = max(duties[f - 1], r["fan_duty_pct"])
        rec["overrides"] += [f"[{r['area']}] {o}" for o in r["overrides"]]
        rec["warnings"] += r["warnings"]
    worst = max(results, key=lambda r: (r["cows_present"], r["rise_c"], r["thi_eff"]))
    rec.update(worst_area=worst["area"], areas=results, fan_duty_pct=duties,
               rise_c=worst["rise_c"], thi_eff=worst["thi_eff"], thi=worst["thi_ambient"],
               safety_status="Safe with Override" if rec["overrides"] else "Safe")
    return rec


def fan_curve(model, cfg, amap, temp_c, rh, z, duties=(0, 7, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)):
    """Small table for the check page: what each fan speed would do in one zone."""
    lo = float(cfg["fans"]["min_duty_pct"])
    duties = sorted(set([0.0, lo] + [float(d) for d in duties if d == 0 or d >= lo]))
    tab = rise_table(model, cfg, amap, temp_c, rh, z, duties)
    return [{"fan_pct": float(d), "air_speed_m_s": round(float(v), 2), "thi_eff": round(float(t), 1),
             "rise_c": round(float(r), 3)}
            for d, v, t, r in zip(tab["duty"], tab["air_speed"], tab["thi_eff"], tab["rise"])]
