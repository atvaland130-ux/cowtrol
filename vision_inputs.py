"""Camera -> control inputs for Cowtrol.
 
Turns the fine-tuned YOLO detections (standing / lying model cows) into the per-zone
numbers the controller already knows how to use:
 
    {zone_name: {"quadrants": [...], "clustering": float, "lying": float}}
 
which is exactly the `camera=` argument `control.zone_inputs()` accepts (camera wins over
the dashboard "Manual inputs" for whatever it reports; a zone with no cows reports an empty
quadrant list, so that side's fans turn off).
 
NOTHING here is live until the camera is MOUNTED and calibrated, because two constants
depend on the final camera position:
  - vision.floor_quad_px : the 4 image-pixel corners of the barn floor (perspective map ->
                           which quadrant a cow is in).
  - vision.barn_wh_m     : the real floor size in metres (pixels -> metres, so the 2.5 m
                           clustering radius is real distance, not pixels).
Use calibrate_vision.py after mounting to set both. Until then this stays off
(config vision.enabled: false) and the controller keeps using the manual inputs.
 
Design choices (all overridable in config `vision:`):
  - A cow's floor position = the centre of its box (good for an overhead view; set
    ground_point: bottom for an oblique/corner view, which uses the box's bottom edge).
  - The floor is split 2 columns (left/right) x 3 rows (near/mid/far) -> 6 quadrants,
    matching config `areas`. Mapping in `quadrant_grid`.
  - clustering = average number of other cows within `cluster_radius_m` of each cow,
    computed over the whole herd (same definition the model trained on) and reported to
    both zones, exactly like the single manual clustering value.
  - lying = per zone, the share of that zone's cows whose class name contains "lying".
"""
import glob
import threading
import time
 
import numpy as np
 
 
# ---------------------------------------------------------------- camera device
class CameraSource:
    """Owns the USB camera and keeps the latest frame in a background thread.
 
    One process can hold the camera at a time, so when vision is on this is the single
    owner; the dashboard live view reads frames from here instead of opening its own.
    Opens by the stable /dev/v4l/by-id path first (the /dev/videoN index shifts on replug).
    """
 
    def __init__(self, width=640, height=480):
        self.width, self.height = width, height
        self._frame = None
        self._lock = threading.Lock()
        self._cap = None
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()
 
    def _open(self):
        import cv2
        for dev in sorted(glob.glob("/dev/v4l/by-id/*")) + sorted(glob.glob("/dev/video*")):
            cap = cv2.VideoCapture(dev)
            if cap.isOpened() and cap.read()[0]:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                print("[vision] camera opened:", dev)
                return cap
            cap.release()
        return None
 
    def _loop(self):
        while not self._stop:
            if self._cap is None:
                self._cap = self._open()
                if self._cap is None:
                    time.sleep(2)
                    continue
            ok, f = self._cap.read()
            if not ok:
                self._cap.release()
                self._cap = None
                time.sleep(0.5)
                continue
            with self._lock:
                self._frame = f
            time.sleep(0.03)
 
    def frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()
 
    def close(self):
        self._stop = True
 
 
# ---------------------------------------------------------------- floor geometry
class FloorMap:
    """Maps an image pixel to (column, row) -> quadrant, and to floor metres.
 
    floor_quad_px is the barn floor's 4 corners in the image, in this order:
        [near-left, near-right, far-right, far-left]
    where NEAR = the edge closest to the fans (the Q1/Q4 row) and LEFT = the Q1-Q3 side.
    They map to a unit square u in [0,1] (left->right), v in [0,1] (near->far).
    """
 
    def __init__(self, cfg):
        v = cfg["vision"]
        self.quad_px = np.array(v["floor_quad_px"], dtype="float32")
        self.barn_w, self.barn_l = (float(x) for x in v["barn_wh_m"])
        self.grid = v["quadrant_grid"]          # {"left": {"near":"Q1",...}, "right": {...}}
        dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype="float32")
        import cv2
        self._H = cv2.getPerspectiveTransform(self.quad_px, dst)
 
    def to_uv(self, pts_px):
        """pts_px: (N,2) image points -> (N,2) unit-square (u,v)."""
        import cv2
        pts = np.asarray(pts_px, dtype="float32").reshape(-1, 1, 2)
        uv = cv2.perspectiveTransform(pts, self._H).reshape(-1, 2)
        return uv
 
    def quadrant(self, u, v):
        col = "left" if u < 0.5 else "right"
        row = "near" if v < 1 / 3 else ("mid" if v < 2 / 3 else "far")
        return self.grid[col][row]
 
    def metres(self, uv):
        xy = np.asarray(uv, dtype=float).copy()
        xy[:, 0] *= self.barn_w
        xy[:, 1] *= self.barn_l
        return xy
 
    def assign(self, pts_px):
        """pts_px (N,2) -> (list of quadrant names, (N,2) metres)."""
        if len(pts_px) == 0:
            return [], np.zeros((0, 2))
        uv = self.to_uv(pts_px)
        quads = [self.quadrant(float(np.clip(u, 0, 1)), float(np.clip(v, 0, 1))) for u, v in uv]
        return quads, self.metres(uv)
 
 
def quadrant_order(cfg):
    """Quadrant names in a fixed order with their (column, row) cell, from `areas` + grid."""
    grid = cfg["vision"]["quadrant_grid"]
    out = []
    for col in ("left", "right"):
        for row in ("near", "mid", "far"):
            out.append((grid[col][row], col, row))
    return out
 
 
class CenterMap:
    """Classify a cow by the NEAREST hand-placed quadrant centre (image space).
 
    Robust to lens distortion / a corner mount, where a straight 4-corner grid can't line up:
    you drop one point in the middle of each quadrant and a cow belongs to whichever centre it's
    closest to. Real distances (for clustering) come from a homography fit to the 6 centre points
    and their known floor positions (barn_wh_m + the 2x3 layout).
    """
 
    _COL_U = {"left": 0.25, "right": 0.75}
    _ROW_V = {"near": 1 / 6, "mid": 0.5, "far": 5 / 6}
 
    def __init__(self, cfg):
        v = cfg["vision"]
        centers = v["quadrant_centers_px"]                      # {Q1: [x,y], ...}
        bw, bl = (float(x) for x in v["barn_wh_m"])
        self.names, px, mx = [], [], []
        for q, col, row in quadrant_order(cfg):
            self.names.append(q)
            px.append([float(centers[q][0]), float(centers[q][1])])
            mx.append([self._COL_U[col] * bw, self._ROW_V[row] * bl])
        self.px = np.array(px, dtype="float32")                 # image centres
        self.mx = np.array(mx, dtype="float32")                 # real centres (m)
        import cv2
        self._H, _ = cv2.findHomography(self.px, self.mx)       # image px -> metres
 
    def assign(self, pts_px):
        if len(pts_px) == 0:
            return [], np.zeros((0, 2))
        pts = np.asarray(pts_px, dtype="float32")
        d = np.linalg.norm(pts[:, None, :] - self.px[None, :, :], axis=2)   # to each centre
        quads = [self.names[i] for i in d.argmin(axis=1)]
        import cv2
        metres = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), self._H).reshape(-1, 2) \
            if self._H is not None else pts
        return quads, metres
 
 
# ---------------------------------------------------------------- vision -> inputs
class VisionInputs:
    """Runs YOLO on the latest frame and returns the per-zone camera dict (or None)."""
 
    def __init__(self, cfg, source=None):
        self.cfg = cfg
        v = cfg["vision"]
        self.conf = float(v.get("conf", 0.5))
        self.imgsz = int(v.get("imgsz", 320))
        self.radius_m = float(v.get("cluster_radius_m", 2.5))
        self.lying_token = v.get("lying_class_contains", "lying").lower()
        self.ground = v.get("ground_point", "center")
        self.areas = cfg["areas"]
        from ultralytics import YOLO
        self.model = YOLO(v["model_path"])
        self.names = self.model.names
        # nearest-centre mapping when quadrant_centers_px is set (robust to lens distortion),
        # otherwise the 4-corner perspective map.
        self.qmap = CenterMap(cfg) if v.get("quadrant_centers_px") else FloorMap(cfg)
        self.source = source or CameraSource(
            width=int(v.get("frame_w", 640)), height=int(v.get("frame_h", 480)))
        self.last = {"cows": 0, "error": None}
 
    # --- one detection pass -------------------------------------------------
    def _ground_points_and_classes(self, frame):
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        pts, lying, conf = [], [], []
        for b in res.boxes:
            x1, y1, x2, y2 = (float(t) for t in b.xyxy[0])
            cx = (x1 + x2) / 2
            cy = y2 if self.ground == "bottom" else (y1 + y2) / 2
            pts.append((cx, cy))
            name = str(self.names[int(b.cls[0])]).lower()
            lying.append(self.lying_token in name)
            conf.append(float(b.conf[0]))
        return (np.array(pts, dtype="float32"), np.array(lying, dtype=bool),
                np.array(conf, dtype="float32"))
 
    def _clustering(self, metres):
        """Average number of OTHER cows within radius_m of each cow (0 if <2 cows)."""
        n = len(metres)
        if n < 2:
            return 0.0
        d = np.linalg.norm(metres[:, None, :] - metres[None, :, :], axis=2)
        within = (d <= self.radius_m).sum(axis=1) - 1          # minus itself
        return float(within.mean())
 
    def camera_dict(self):
        """{zone: {quadrants, clustering, lying}} or None if no frame yet.
 
        Returns a dict for BOTH zones every call so it fully overrides the manual inputs
        when vision is on. A zone with no detected cows gets quadrants: [] -> fans off."""
        frame = self.source.frame()
        if frame is None:
            self.last = {"cows": 0, "error": "no frame"}
            return None
        try:
            pts, lying, conf = self._ground_points_and_classes(frame)
        except Exception as e:                                  # bad model / frame -> fall back to manual
            self.last = {"cows": 0, "error": f"detect failed: {e}"}
            return None
 
        clustering = 0.0
        per_quad = {}           # quadrant -> [lying bools]
        summary = {}            # quadrant -> {n, conf(%), lying} for the dashboard
        if len(pts):
            quads, metres = self.qmap.assign(pts)
            clustering = self._clustering(metres)
            for q, is_lying, cf in zip(quads, lying, conf):
                per_quad.setdefault(q, []).append(bool(is_lying))
                e = summary.setdefault(q, {"n": 0, "conf": 0, "lying": 0})
                e["n"] += 1
                e["conf"] = max(e["conf"], round(float(cf) * 100))
                e["lying"] += 1 if is_lying else 0
 
        out = {}
        for a in self.areas:
            occ = [q for q in a["quadrants"] if q in per_quad]
            flags = [f for q in occ for f in per_quad[q]]
            zone = {"quadrants": occ, "clustering": clustering}
            if flags:                                           # only report lying when cows are seen here
                zone["lying"] = sum(flags) / len(flags)
            out[a["name"]] = zone
        self.last = {"cows": int(len(pts)), "error": None,
                     "clustering": round(clustering, 2), "per_quad": summary}
        return out
 
