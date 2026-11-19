"""Cowtrol main program (runs on the Raspberry Pi).
 
- Reads the Arduino, runs the decision pipeline every decision_interval_s, sends fan speeds, logs.
- Serves a live dashboard at http://<pi-hostname>.local:<port>
  (readings, predicted body temperature rise, fans, safety, goal, manual inputs, LLM chat, and a page for
  checking inputs against outputs).
 
Arduino -> Pi:  S,<temp_c>,<rh_pct>            one sensor (counts as sensor 1), every ~2 s
                S,<sensor>,<temp_c>,<rh_pct>   several sensors, one line each
                E,<msg> on a sensor error, A,<d1>,... acknowledges a fan command
Pi -> Arduino:  F,<duty1>[,<duty2>,...]        duty in % (0-100), one value per fan
 
Usage:
  python controller.py              # real Arduino
  python controller.py --simulate   # fake sensor, no hardware
"""
import argparse
import collections
import csv
import json
import os
import random
import sys
import threading
import time
 
import numpy as np
from flask import Flask, Response, jsonify, request, send_file
 
from control import (check_hour, clamp_goal, decide, fan_curve, n_fans, reading_valid, resolve_readings, sensor_ids,
                     zone_inputs)
from cowtrol_common import load_config, nrc_thi, path
from heatload import AirspeedMap
from llm import LLM
from rag import Guidelines
from realcow import RealCow
from safety import InputGuard, ManualOverride, camera_health
 
 
# ---------- hardware links ----------
class ArduinoLink:
    def __init__(self, port, baud):
        self.port, self.baud, self.ser = port, baud, None
        self._open()
 
    def _candidates(self):
        """The configured port first, then the stable by-id paths, then any /dev/ttyACM*/ttyUSB*,
        so a reconnect still finds the Arduino if it re-enumerated on a different index."""
        import glob
        seen, out = set(), []
        for p in [self.port, *sorted(glob.glob("/dev/serial/by-id/*")),
                  *sorted(glob.glob("/dev/ttyACM*")), *sorted(glob.glob("/dev/ttyUSB*"))]:
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return out
 
    def _open(self):
        import serial
        last = None
        for p in self._candidates():
            try:
                self.ser = serial.Serial(p, self.baud, timeout=1)
                time.sleep(2)                    # the Arduino resets when the port opens
                self.ser.reset_input_buffer()
                self.port = p
                return True
            except Exception as e:               # port busy / not present yet -> try the next
                last = e
        print(f"[serial] could not open any port ({last})")
        return False
 
    def reopen(self):
        """Close and re-open after a serial error so a USB/cable hiccup self-heals (no manual reset)."""
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        self.ser = None
        return self._open()
 
    def read_line(self):
        if self.ser is None and not self._open():   # still down -> let run() back off and retry
            time.sleep(1)
            return None
        raw = self.ser.readline().decode(errors="ignore").strip()   # SerialException bubbles to run()
        return raw or None
 
    def send_fans(self, duties):
        if self.ser is None:
            return
        self.ser.write(("F," + ",".join(f"{d:.0f}" for d in duties) + "\n").encode())
 
 
class SimLink:
    """Random-walk temperature/humidity for testing without hardware (one stream per configured sensor)."""
 
    def __init__(self, sensor_ids=(1,)):
        self.ids = list(sensor_ids)
        self.t, self.rh, self.i = 28.0, 60.0, 0
        self.mode = "ok"        # dashboard test buttons (--simulate only): ok | silent | out_of_range | flatline | far
 
    def read_line(self):
        time.sleep(0.5 / len(self.ids))
        sid = self.ids[self.i % len(self.ids)]
        self.i += 1
        if sid == self.ids[0]:
            self.t = min(40, max(15, self.t + random.uniform(-0.3, 0.4)))
            self.rh = min(90, max(20, self.rh + random.uniform(-1, 1)))
        t = self.t + 0.8 * self.ids.index(sid)             # each sensor a little different
        if sid == self.ids[0] and self.mode != "ok":
            if self.mode == "silent":
                return None
            t = {"out_of_range": 150.0, "flatline": 25.0, "far": t + 8}.get(self.mode, t)
            if self.mode == "flatline":
                self.rh = 55
        if len(self.ids) == 1:
            return f"S,{t:.1f},{self.rh:.0f}"                # single-sensor line format
        return f"S,{sid},{t:.1f},{self.rh:.0f}"
 
    def send_fans(self, duties):
        pass
 
 
# ---------- logging ----------
def log(rec, cfg):
    for key in ("log_csv", "log_jsonl"):
        os.makedirs(os.path.dirname(path(cfg["loop"][key])), exist_ok=True)
    with open(path(cfg["loop"]["log_jsonl"]), "a") as f:
        f.write(json.dumps(rec) + "\n")
    areas = {a["name"]: {} for a in cfg["areas"]}
    areas.update({a["area"]: a for a in rec["areas"]})
    g = rec["goal"]
    row = {"timestamp": rec["timestamp"], "target_rise_c": g["target_rise_c"], "min_duty_pct": g["min_duty_pct"],
           "goal_note": g.get("note")}
    for name, a in areas.items():
        for k in ("sensor", "sensor_borrowed", "temp_c", "rh", "thi_ambient", "solar_wm2", "clustering", "lying",
                  "hour", "cows_present", "fan_duty_pct", "ml_duty_pct", "air_speed_m_s", "air_speed_effect", "solar_effect",
                  "thi_eff", "rise_c", "rise_fan_off_c", "target_reached", "standing_flag"):
            row[f"{name}_{k}"] = a.get(k)
        row[f"{name}_quadrants_used"] = " ".join(a.get("quadrants_used", []))
    row.update({"safety_status": rec["safety_status"], "overrides": " | ".join(rec["overrides"]),
                "faults": " | ".join(rec.get("faults", [])),
                "manual_override_pct": (rec.get("manual_override") or {}).get("duty_pct"),
                "warnings": " | ".join(rec.get("warnings", [])),
                "air_speed_measured": rec["calibrated"]})
    csv_path = path(cfg["loop"]["log_csv"])
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row), extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
 
 
# ---------- controller ----------
class Controller:
    def __init__(self, cfg, link, interval):
        self.cfg, self.link, self.interval = cfg, link, interval
        self.model = RealCow(cfg)
        if not self.model.ready:
            sys.exit("No body temperature model found. Run train_realcow.py first.")
        self.amap = AirspeedMap(cfg)
        missing = [q for q, measured in self.amap.measured.items() if not measured]
        if missing:
            print(f"[warn] quadrants {', '.join(missing)}: no air speed measurements in {cfg['airspeed_map']['file']}; "
                  f"using a straight-line placeholder (0 to {self.amap.placeholder_max} m/s).")
        self.lock = threading.Lock()
        self.goal = {**clamp_goal(cfg, **cfg["goal"]["default"]), "note": "default", "source": "config"}
        self.manual = dict(cfg["manual_inputs"])
        self.decisions = collections.deque(maxlen=50)
        self.sensor_ids = sensor_ids(cfg)
        now = time.time()
        # per sensor: readings in the current window, last window average, time of last valid reading, live values
        self.sensors = {i: {"window": [], "last": None, "seen": now, "down": False,
                            "latest": {"temp_c": None, "rh": None, "thi": None, "time": None}}
                        for i in self.sensor_ids}
        self.events = collections.deque(maxlen=20)
        self.last_valid = now
        self.decide_now = threading.Event()
        # FMEA safeguards (safety.py)
        self.override = ManualOverride(cfg)
        self.guard = InputGuard(cfg, self.sensor_ids)
        self.faults = []
        self.camera, self.camera_status = None, {"enabled": False}
        cam_cfg = cfg["safety_checks"]["camera"]
        if isinstance(link, SimLink) or cam_cfg["enabled"]:
            try:
                from camera import Camera, SimCamera
                self.camera = SimCamera(cfg) if isinstance(link, SimLink) else Camera(cfg)
                self.camera_status = {"enabled": True}
            except Exception as e:                    # camera enabled but not working = a fault
                self.camera_status = {"enabled": True, "error": f"camera not available: {e}"}
 
        # Camera -> model inputs (vision_inputs.py). Off unless config vision.enabled; when on it
        # owns the USB camera (the dashboard live view then reads frames from it).
        self.vision = None
        if cfg.get("vision", {}).get("enabled") and not isinstance(link, SimLink):
            try:
                from vision_inputs import VisionInputs
                self.vision = VisionInputs(cfg)
                print("[vision] camera inputs ON (model:", cfg["vision"]["model_path"], ")")
            except Exception as e:
                print(f"[vision] disabled: {e}")   # missing model/opencv/ultralytics -> fall back to manual
 
    def set_goal(self, target_rise_c, min_duty_pct, note, source):
        g = {**clamp_goal(self.cfg, target_rise_c, min_duty_pct), "note": note, "source": source}
        with self.lock:
            self.goal = g
        self.event(f"Goal -> rise under {g['target_rise_c']} C, fans at least {g['min_duty_pct']:.0f}% ({note})")
        self.decide_now.set()
        return g
 
    def set_manual(self, values):
        with self.lock:
            self.manual.update(values)
        self.event("Manual inputs -> " + ", ".join(f"{k} {v}" for k, v in values.items()))
        self.decide_now.set()
 
    def event(self, msg):
        with self.lock:
            self.events.appendleft({"time": time.strftime("%H:%M:%S"), "msg": msg})
        print(f"[event] {msg}")
 
    def handle_reading(self, line):
        """S,<temp>,<rh>  or  S,<sensor>,<temp>,<rh>"""
        parts = line.split(",")
        try:
            if len(parts) == 3:
                sid, t, h = 1, float(parts[1]), float(parts[2])
            else:
                sid, t, h = int(parts[1]), float(parts[2]), float(parts[3])
        except (ValueError, IndexError):
            return
        if sid not in self.sensors:
            return                                    # a sensor no area uses
        off = (self.cfg.get("sensor_offsets") or {}).get(sid) or {}
        t = t + float(off.get("temp_c", 0.0))          # HOBO-derived calibration (config sensor_offsets)
        h = min(100.0, max(0.0, h + float(off.get("rh_pct", 0.0))))
        if not self.guard.add(sid, t, h):              # out-of-range: a fault, not just ignored
            self.event(f"Out-of-range reading from sensor {sid}: {t} C, {h}% RH")
            return
        sn = self.sensors[sid]
        with self.lock:
            sn["window"].append((t, h))
            sn["seen"] = self.last_valid = time.time()
            sn["latest"] = {"temp_c": t, "rh": h, "thi": round(float(nrc_thi(t, h)), 1),
                            "time": time.strftime("%H:%M:%S")}
 
    def sensor_values(self):
        """Window average per sensor. A sensor silent for stale_data_s counts as not answering (None)."""
        now, out, changes = time.time(), {}, []
        limit = self.cfg["safety"]["stale_data_s"]
        with self.lock:
            for sid, sn in self.sensors.items():
                if sn["window"]:
                    sn["last"] = tuple(float(v) for v in np.mean(sn["window"], axis=0))
                    sn["window"] = []
                silent = now - sn["seen"] >= limit
                if silent != sn["down"]:
                    sn["down"] = silent
                    changes.append(f"Sensor {sid} {'is not answering' if silent else 'is answering again'}")
                out[sid] = None if (silent or sn["last"] is None) else sn["last"]
        for c in changes:
            self.event(c)
        return out
 
    def check_inputs(self, values):
        """Alt 2: validate every input before the model runs. Returns a list of fault messages."""
        faults = [f"Sensor {sid}: {why}" for sid, why in self.guard.faults(values).items()]
        if self.camera_status.get("enabled"):
            if self.camera is None:
                faults.append(self.camera_status["error"])
            else:
                try:
                    frame = self.camera.frame()
                    if self.camera.reference is None and frame is not None:
                        self.camera.set_reference(frame)          # first good view = normal view
                        self.event("Camera reference frame saved")
                    why, bright, sim = ("camera returned no image", 0.0, None) if frame is None else \
                        camera_health(frame, self.camera.reference, self.cfg)
                    self.camera_status.update(brightness=round(bright, 1),
                                              similarity=None if sim is None else round(sim, 2), fault=why)
                    if why:
                        faults.append(why[0].upper() + why[1:])
                except Exception as e:
                    faults.append(f"Camera error: {e}")
        new, cleared = set(faults) - set(self.faults), set(self.faults) - set(faults)
        for f in sorted(new):
            self.event(f"FAULT: {f} -> fans 100%")
        for f in sorted(cleared):
            self.event(f"Fault cleared: {f}")
        with self.lock:
            self.faults = faults
        return faults
 
    def apply_safeguards(self, rec, faults):
        """Final say on the fan command. Order: input fault (100 %) > manual override > AI decision."""
        n_fans = len(rec["fan_duty_pct"])
        thi = [a["thi_ambient"] for a in rec["areas"]]
        with self.lock:
            msg = self.override.check(max(thi) if thi else None)       # Alt 1: expiry and THI revoke
        if msg:
            self.event(msg)
        rec["faults"] = list(faults)
        if faults and self.cfg["safety"].get("enabled", True):
            # Only blast the fans if there are cows to protect. If every zone is empty (camera), a sensor
            # fault leaves the fans off. When the stale guard already handled the command (no areas in the
            # record) keep that command and just flag the fault.
            cows = any(a.get("cows_present") for a in rec["areas"])
            if rec["areas"] and not cows:
                rec["fan_duty_pct"] = [0.0] * n_fans
                rec["overrides"].insert(0, "Input fault, but no cows -> fans off: " + "; ".join(faults))
                rec["safety_status"] = "Fault (no cows, fans off)"
            elif rec["areas"]:
                rec["fan_duty_pct"] = [100.0] * n_fans
                rec["overrides"].insert(0, "Input fault -> fans 100% (fail-safe): " + "; ".join(faults))
                rec["safety_status"] = "FAULT: fans 100%"
            else:
                rec["overrides"].insert(0, "Input fault: " + "; ".join(faults))
                rec["safety_status"] = "FAULT"
        elif self.override.active:
            rec["manual_override"] = {"duty_pct": self.override.duty[0],
                                      "remaining_min": round(self.override.remaining_s() / 60, 1)}
            rec["fan_duty_pct"] = list(self.override.duty[:n_fans]) + [0.0] * (n_fans - len(self.override.duty))
            rec["overrides"].insert(0, f"Manual override: fans {self.override.duty[0]:.0f}% "
                                       f"({rec['manual_override']['remaining_min']} min left)")
            rec["safety_status"] = "Manual override"
 
    def start_override(self, duty):
        n = n_fans(self.cfg)
        with self.lock:
            msg = self.override.start(duty, n)
        self.event(msg)
        self.decide_now.set()
 
    def stop_override(self, reason):
        with self.lock:
            msg = self.override.stop(reason) if self.override.active else None
        if msg:
            self.event(msg)
        self.decide_now.set()
 
    def run(self):
        t0 = time.time()
        while True:
            try:
                line = self.link.read_line()
            except Exception as e:                       # serial dropped (USB autosuspend, unplug, Arduino hang)
                self.event(f"Serial read error: {e} - reconnecting")
                if hasattr(self.link, "reopen"):
                    self.link.reopen()                   # self-heal instead of needing a manual reset
                time.sleep(1)
                line = None
            if line is None:
                time.sleep(0.2)
            if line and line.startswith("S,"):
                self.handle_reading(line)
            elif line and line.startswith("E,"):
                self.event(f"Arduino: {line[2:]}")
 
            if time.time() - t0 >= self.interval or self.decide_now.is_set():
                self.decide_now.clear()
                stale = time.time() - self.last_valid >= self.cfg["safety"]["stale_data_s"]
                values = self.sensor_values()
                readings, warnings = resolve_readings(self.cfg, values)
                faults = self.check_inputs(values)
                if readings is None and not stale and not faults:
                    continue                  # nothing yet; keep the last command
                t0 = time.time()
                with self.lock:
                    goal, manual = dict(self.goal), dict(self.manual)
                camera = self.vision.camera_dict() if self.vision else None   # None -> manual inputs
                inputs = zone_inputs(self.cfg, manual, camera=camera)
                rec = decide(self.model, self.cfg, self.amap, goal, readings, inputs,
                             stale=readings is None, warnings=warnings)
                self.apply_safeguards(rec, faults)
                self.link.send_fans(rec["fan_duty_pct"])
                log(rec, self.cfg)
                with self.lock:
                    self.decisions.append(rec)
                print(f"{rec['timestamp']} rise={rec['rise_c']} C target={goal['target_rise_c']} "
                      f"duty={rec['fan_duty_pct']} {rec['safety_status']}")
 
 
# ---------- dashboard ----------
def make_app(ctl, cfg):
    app = Flask(__name__)
    app.json.sort_keys = False          # keep stress levels in severity order
    llm = LLM(cfg)
    guides = Guidelines(cfg)
    chat_history = collections.deque(maxlen=6)
    status = {"llm_online": False, "checked": 0}
 
    def llm_status():
        if time.time() - status["checked"] > 15:
            status["llm_online"], status["checked"] = llm.online(), time.time()
        return status["llm_online"]
 
    @app.get("/")
    def index():
        return send_file(path("dashboard.html"))
 
    # --- live camera view (optional) -------------------------------------------------
    # Opens the USB camera on first request and streams JPEG snapshots to the dashboard.
    # Separate from the safety health-check camera (safety_checks.camera); when that is
    # enabled with source: usb they would share one device, so the view stays off then.
    cam_view = {"cap": None, "frame": None, "started": False, "ok": False}
 
    def _open_view_camera():
        import glob
        import cv2
        nodes = sorted(glob.glob("/dev/v4l/by-id/*")) + sorted(glob.glob("/dev/video*"))
        for dev in nodes:
            cap = cv2.VideoCapture(dev)
            if cap.isOpened() and cap.read()[0]:
                cap.set(3, 640)
                cap.set(4, 480)
                return cap
            cap.release()
        return None
 
    def _view_loop():
        while cam_view["cap"] is not None:
            ok, f = cam_view["cap"].read()
            if ok:
                cam_view["frame"] = f
            else:
                time.sleep(0.1)
 
    @app.get("/camera.jpg")
    def camera_jpg():
        try:
            import cv2
            if ctl.vision is not None:                     # vision owns the camera; share its frame
                f = ctl.vision.source.frame()
                if f is None:
                    return ("", 503)
                ok, buf = cv2.imencode(".jpg", f)
                return Response(buf.tobytes(), mimetype="image/jpeg")
            if not cam_view["started"]:
                cam_view["started"] = True
                if not ctl.camera_status.get("enabled"):   # don't fight the safety camera for the device
                    cam_view["cap"] = _open_view_camera()
                    cam_view["ok"] = cam_view["cap"] is not None
                    if cam_view["ok"]:
                        threading.Thread(target=_view_loop, daemon=True).start()
            f = cam_view["frame"]
            if f is None:
                return ("", 503)
            ok, buf = cv2.imencode(".jpg", f)
            return Response(buf.tobytes(), mimetype="image/jpeg")
        except Exception:
            return ("", 503)                                # no OpenCV / no camera -> dashboard shows "no camera"
 
    @app.get("/api/state")
    def state():
        with ctl.lock:
            sensors = [{"id": sid, "areas": [q for q, i in cfg["quadrant_sensors"].items() if i == sid],
                        "down": sn["down"], **sn["latest"]} for sid, sn in ctl.sensors.items()]
            return jsonify(sensors=sensors, goal=ctl.goal, goal_limits=cfg["goal"]["limits"],
                           goal_default=cfg["goal"]["default"], manual=ctl.manual,
                           cbt={"baseline_c": cfg["goal"].get("baseline_c"),
                                "stress_threshold_c": cfg["goal"].get("stress_threshold_c")},
                           decisions=list(ctl.decisions)[-cfg["llm"]["history"]:], events=list(ctl.events),
                           calibrated=ctl.amap.all_measured, model=ctl.model.bundle.get("source", "MmCows"),
                           areas=[{"name": a["name"], "fans": a["fans"], "quadrants": a["quadrants"],
                                   "fan_label": a.get("fan_label", "+".join(map(str, a["fans"])))}
                                  for a in cfg["areas"]],
                           camera_inputs=ctl.vision is not None,   # True when the camera feeds the model
                           vision=(ctl.vision.last if ctl.vision else None),  # cows/quadrant + confidence
                           mode=cfg.get("control", {}).get("mode", "optimizer"),
                           llm_host=llm.host, llm_online=llm_status(),
                           seconds_since_valid=round(time.time() - ctl.last_valid),
                           faults=ctl.faults, camera=ctl.camera_status,
                           override={"active": ctl.override.active,
                                     "duty_pct": ctl.override.duty[0] if ctl.override.active else None,
                                     "remaining_min": None if not ctl.override.active
                                     else round(ctl.override.remaining_s() / 60, 1),
                                     "max_minutes": cfg["safety_checks"]["override"]["max_minutes"],
                                     "revoke_thi": cfg["safety_checks"]["override"]["revoke_thi"]},
                           simulate=isinstance(ctl.link, SimLink))
 
    @app.post("/api/goal_numbers")         # typed in: works even when the LLM is offline
    def goal_numbers():
        j = request.json or {}
        try:
            t, d = float(j["target_rise_c"]), float(j.get("min_duty_pct", 0))
        except (KeyError, TypeError, ValueError):
            return jsonify(error="Enter numbers for the target and minimum fan speed."), 400
        if j.get("reset"):
            t, d = cfg["goal"]["default"]["target_rise_c"], cfg["goal"]["default"]["min_duty_pct"]
        return jsonify(ctl.set_goal(t, d, "default" if j.get("reset") else "set by hand", "manual"))
 
    @app.post("/api/goal")                 # free text -> LLM -> clamped numbers
    def goal():
        text = request.json.get("text", "").strip()
        with ctl.lock:
            current = dict(ctl.goal)
            last = ctl.decisions[-1] if ctl.decisions else None
        try:
            out = llm.set_goal(text, cfg, current, last)
        except Exception as e:
            return jsonify(error=f"LLM unavailable or invalid output: {e}"), 502
        g = ctl.set_goal(out["target_rise_c"], out["min_duty_pct"], f"LLM: \"{text}\" -> {out['reason']}", "llm")
        return jsonify({**g, "reason": out["reason"], "clamped": out["clamped"]})
 
    @app.post("/api/goal/reset")           # drop any operator/LLM goal, go back to config default
    def goal_reset():
        d = cfg["goal"]["default"]
        g = ctl.set_goal(d["target_rise_c"], d.get("min_duty_pct", 0), "default", "config")
        return jsonify({**g, "reason": "back to default", "clamped": False})
 
    @app.post("/api/ask/clear")            # forget the assistant chat so far
    def ask_clear():
        chat_history.clear()
        return jsonify(ok=True)
 
    @app.post("/api/manual")               # inputs we do not sense yet (solar, clustering, lying, occupied quadrants)
                                           # and the clock override
    def manual():
        j = request.json or {}
        quads = [q for a in cfg["areas"] for q in a["quadrants"]]
        vals = {}
        try:
            if "solar_wm2" in j:
                vals["solar_wm2"] = float(j["solar_wm2"])
            if "clustering" in j:
                vals["clustering"] = float(j["clustering"])
            if "lying" in j:
                vals["lying"] = float(j["lying"])
            if "occupied_quadrants" in j:
                vals["occupied_quadrants"] = [q for q in j["occupied_quadrants"] if q in quads]
            if "clock_hour" in j:
                vals["clock_hour"] = None if j["clock_hour"] in (None, "") else float(j["clock_hour"])
        except (TypeError, ValueError):
            return jsonify(error="Enter numbers."), 400
        if not (0 <= vals.get("solar_wm2", 0) <= 1400 and vals.get("clustering", 0) >= 0
                and 0 <= vals.get("lying", 0) <= 1):
            return jsonify(error="Out of range (solar 0-1400 W/m2, clustering 0 or more, lying 0-1)."), 400
        err = check_hour(vals.get("clock_hour"))
        if err:
            return jsonify(error=f"Out of range ({err})."), 400
        ctl.set_manual(vals)
        return jsonify(ok=True)
 
    @app.post("/api/ask")                  # question -> RAG + decision log -> LLM explanation
    def ask():
        q = request.json.get("text", "").strip()
        with ctl.lock:
            decs = list(ctl.decisions)[-cfg["llm"]["history"]:]
        last = decs[-1] if decs else {}
        retrieved = guides.search(f"{q} {' '.join(last.get('overrides', []))} {' '.join(last.get('warnings', []))}")
        try:
            ans = llm.explain(q, decs, retrieved, cfg, history=list(chat_history))
        except Exception as e:
            return jsonify(error=f"LLM unavailable: {e}"), 502
        chat_history.extend([{"role": "user", "content": q}, {"role": "assistant", "content": ans}])
        return jsonify(answer=ans, sources=[{"id": d["id"], "source": d["source"]} for d in retrieved])
 
    @app.post("/api/whatif")               # check page: try inputs, nothing is sent to the fan or logged
    def whatif():
        j = request.json or {}
        try:
            temp, rh = float(j["temp_c"]), float(j["rh"])
            z = {"clustering": float(j["clustering"]), "lying": float(j["lying"]),
                 "solar_wm2": float(j["solar_wm2"]), "cows_present": True, "hour": float(j["hour"])}
            goal = clamp_goal(cfg, float(j["target_rise_c"]), float(j.get("min_duty_pct", 0)))
        except (KeyError, TypeError, ValueError):
            return jsonify(error="Enter a number in every box."), 400
        area = next((a for a in cfg["areas"] if a["name"] == j.get("area")), cfg["areas"][0])
        z["quadrants"] = [q for q in j.get("quadrants", []) if q in area["quadrants"]] or list(area["quadrants"])
        if not (reading_valid(cfg, temp, rh) and z["clustering"] >= 0 and 0 <= z["lying"] <= 1
                and 0 <= z["solar_wm2"] <= 1400):
            return jsonify(error="Inputs out of range (humidity 0-100, clustering 0 or more, lying 0-1, "
                                 "solar 0-1400 W/m2)."), 400
        err = check_hour(z["hour"])
        if err:
            return jsonify(error=f"Inputs out of range ({err})."), 400
        readings = {q: {"temp_c": temp, "rh": rh, "sensor": sid, "borrowed": False}
                    for q, sid in cfg["quadrant_sensors"].items()}
        inputs = {a["name"]: z if a is area else {**z, "quadrants": list(a["quadrants"])} for a in cfg["areas"]}
        rec = decide(ctl.model, cfg, ctl.amap, goal, readings, inputs)
        result = next(a for a in rec["areas"] if a["area"] == area["name"])
        return jsonify(result=result, safety_status=rec["safety_status"],
                       curve=fan_curve(ctl.model, cfg, ctl.amap, temp, rh, z,
                                       duties=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                                               result["fan_duty_pct"], result["ml_duty_pct"])))
 
    @app.post("/api/override")             # Alt 1: manual fan speed, time-limited and auto-revoked
    def override():
        j = request.json or {}
        if j.get("stop"):
            ctl.stop_override("cancelled by operator")
            return jsonify(ok=True)
        try:
            duty = float(j["duty_pct"])
        except (KeyError, TypeError, ValueError):
            return jsonify(error="Enter a fan speed from 0 to 100."), 400
        if not 0 <= duty <= 100:
            return jsonify(error="Enter a fan speed from 0 to 100."), 400
        ctl.start_override(duty)
        return jsonify(ok=True)
 
    @app.post("/api/camera_reference")     # re-take the camera's normal view after mounting it
    def camera_reference():
        if ctl.camera is None:
            return jsonify(error="No camera."), 400
        frame = ctl.camera.frame()
        if frame is None:
            return jsonify(error="Camera returned no image."), 400
        ctl.camera.set_reference(frame)
        ctl.event("Camera reference frame re-taken")
        ctl.decide_now.set()
        return jsonify(ok=True)
 
    @app.post("/api/sim_fault")            # --simulate only: test buttons for the safeguards
    def sim_fault():
        if not isinstance(ctl.link, SimLink):
            return jsonify(error="Only in --simulate."), 400
        j = request.json or {}
        if "camera" in j:
            ctl.camera.state = j["camera"]
        if "sensor" in j:
            ctl.link.mode = j["sensor"]
        if "thi_bump" in j:
            ctl.link.t += float(j["thi_bump"])
        ctl.decide_now.set()
        return jsonify(ok=True)
 
    @app.post("/api/llm_host")             # hotspot IPs change: update without editing config
    def set_host():
        llm.host = request.json.get("host", "").strip().rstrip("/")
        status["checked"] = 0
        return jsonify(ok=True, online=llm_status())
 
    return app
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true", help="fake sensor, no Arduino")
    ap.add_argument("--interval", type=float, help="override decision interval (s)")
    args = ap.parse_args()
 
    cfg = load_config()
    link = SimLink(sensor_ids(cfg)) if args.simulate else ArduinoLink(cfg["serial"]["port"], cfg["serial"]["baud"])
    ctl = Controller(cfg, link, args.interval or cfg["loop"]["decision_interval_s"])
    threading.Thread(target=ctl.run, daemon=True).start()
    print(f"Dashboard: http://<this-pi>.local:{cfg['dashboard']['port']}")
    make_app(ctl, cfg).run(host="0.0.0.0", port=cfg["dashboard"]["port"], threaded=True)
 
 
if __name__ == "__main__":
    main()
 
