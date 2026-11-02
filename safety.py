"""FMEA safeguards (hazard: operator override left active, or sensors / camera moved or blocked).

Alt 1  ManualOverride: the operator can set the fans by hand from the dashboard, but the override
       expires after override.max_minutes (60) and is revoked at once if any area's ambient THI reaches
       override.revoke_thi (72). Control then returns to the AI and the event is logged.
Alt 2  InputGuard + camera_health: every control loop checks the inputs before the model runs.
       Temperature/humidity sensor fault = no reading, out-of-range value, flatline, or far from the
       median of the other sensors. Camera fault = too dark (blocked) or too different from the reference
       frame (moved). On ANY fault the fans go to 100 % (fail-safe) and the dashboard shows an alert.
All limits are in config.yaml -> safety_checks.
"""
import collections
import time

import numpy as np


class ManualOverride:
    def __init__(self, cfg):
        c = cfg["safety_checks"]["override"]
        self.max_s = c["max_minutes"] * 60
        self.revoke_thi = c["revoke_thi"]
        self.duty = None          # list of fan duties while active
        self.started = None
        self.note = ""

    @property
    def active(self):
        return self.duty is not None

    def start(self, duty, n_fans, now=None):
        self.duty = [float(min(100, max(0, duty)))] * n_fans
        self.started = now or time.time()
        return f"Manual override ON: fans {self.duty[0]:.0f}% (expires in {self.max_s // 60:.0f} min)"

    def stop(self, reason):
        self.duty, self.started = None, None
        return f"Manual override OFF ({reason}); control returned to the AI"

    def remaining_s(self, now=None):
        return None if not self.active else max(0, self.max_s - ((now or time.time()) - self.started))

    def check(self, max_thi, now=None):
        """Called every control loop. Returns an event message if the override was ended, else None."""
        if not self.active:
            return None
        if max_thi is not None and max_thi >= self.revoke_thi:
            return self.stop(f"revoked: area THI {max_thi:.1f} >= {self.revoke_thi}")
        if self.remaining_s(now) <= 0:
            return self.stop(f"expired after {self.max_s // 60:.0f} min")
        return None


class InputGuard:
    """Per-sensor checks for the temperature/humidity sensors."""

    def __init__(self, cfg, sensor_ids):
        c = cfg["safety_checks"]["sensors"]
        self.no_reading_s = c["no_reading_s"]
        self.flat_s = c["flatline_minutes"] * 60
        self.flat_t, self.flat_rh = c["flatline_temp_c"], c["flatline_rh_pct"]
        self.max_dev = c["max_dev_from_median_c"]
        self.lo_t, self.hi_t = cfg["safety"]["valid_temp_c"]
        self.lo_h, self.hi_h = cfg["safety"]["valid_rh_pct"]
        now = time.time()
        self.seen = {i: now for i in sensor_ids}
        self.history = {i: collections.deque() for i in sensor_ids}   # (time, temp, rh)
        self.bad_value = {i: None for i in sensor_ids}

    def add(self, sid, t, h, now=None):
        """Record one raw reading. Out-of-range readings are kept as a fault, not silently dropped."""
        now = now or time.time()
        self.seen[sid] = now
        if not (self.lo_t <= t <= self.hi_t and self.lo_h <= h <= self.hi_h):
            self.bad_value[sid] = (now, t, h)
            return False
        self.bad_value[sid] = None
        q = self.history[sid]
        q.append((now, t, h))
        while q and q[0][0] < now - self.flat_s - 60:
            q.popleft()
        return True

    def faults(self, latest, now=None):
        """latest = {sensor id: (temp, rh) or None}. Returns {sensor id: reason} for every faulted sensor."""
        now = now or time.time()
        out = {}
        for sid in self.seen:
            if now - self.seen[sid] >= self.no_reading_s:
                out[sid] = f"no reading for {now - self.seen[sid]:.0f} s"
            elif self.bad_value[sid] is not None:
                _, t, h = self.bad_value[sid]
                out[sid] = f"out-of-range value ({t} C, {h} % RH)"
            else:
                q = [r for r in self.history[sid] if r[0] >= now - self.flat_s]
                spans = len(self.history[sid]) and now - self.history[sid][0][0] >= self.flat_s
                if spans and q and (max(r[1] for r in q) - min(r[1] for r in q) < self.flat_t
                                    and max(r[2] for r in q) - min(r[2] for r in q) < self.flat_rh):
                    out[sid] = f"flatline: no change for {self.flat_s // 60:.0f} min"
        # median check needs at least 2 OTHER working sensors, otherwise it cannot tell which one is wrong
        ok = {s: v for s, v in latest.items() if v is not None and s not in out}
        for sid, v in ok.items():
            others = [o[0] for s, o in ok.items() if s != sid]
            if len(others) >= 2 and abs(v[0] - float(np.median(others))) > self.max_dev:
                out[sid] = f"{v[0]:.1f} C is more than {self.max_dev} C from the other sensors' median " \
                           f"({float(np.median(others)):.1f} C)"
        return out


def camera_health(frame_gray, reference_gray, cfg):
    """frame_gray / reference_gray: 2-D uint8 arrays. Returns (fault reason or None, brightness, similarity).
    Similarity = correlation of the two images shrunk to 64x48 (1 = identical view, ~0 = unrelated)."""
    c = cfg["safety_checks"]["camera"]
    brightness = float(np.mean(frame_gray))
    if brightness < c["min_brightness"]:
        return f"camera blocked (brightness {brightness:.0f}/255 < {c['min_brightness']})", brightness, None
    if reference_gray is None:
        return None, brightness, None
    a, b = _small(frame_gray), _small(reference_gray)
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    sim = float((a * b).sum() / den) if den > 0 else 0.0
    if sim < c["min_similarity"]:
        return f"camera moved (similarity to reference {sim:.2f} < {c['min_similarity']})", brightness, sim
    return None, brightness, sim


def _small(img, w=64, h=48):
    img = np.asarray(img, dtype=float)
    ys = (np.arange(h) * img.shape[0] / h).astype(int)
    xs = (np.arange(w) * img.shape[1] / w).astype(int)
    return img[np.ix_(ys, xs)]
