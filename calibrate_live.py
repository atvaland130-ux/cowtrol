#!/usr/bin/env python3
"""Live, drag-to-set camera -> quadrant calibration (browser tool).

Open http://<pi>:8003 and you get the live camera feed with the barn-floor quad drawn on
top as 4 draggable handles. Drag them to the real floor corners; the Q1-Q6 grid (correctly
perspective-warped, so the lines go diagonal for a corner-mounted camera) updates live, and
any cows the model sees are dotted and labelled with the quadrant they fall in. Hit Save and
it writes vision.floor_quad_px straight into config.yaml (comments preserved).

Run on the Pi (stop demo.py / the controller first — one program holds the camera):
    cd ~/cowtrol
    source venv/bin/activate
    python3 calibrate_live.py
Then set vision.barn_wh_m by hand (measure the floor) and flip vision.enabled: true.

Only the floor corners are set here; everything else lives in config.yaml under `vision:`.
"""
import io
import re

import numpy as np
from flask import Flask, Response, request, jsonify

from cowtrol_common import load_config, path
from vision_inputs import CameraSource, FloorMap, VisionInputs

PORT = 8003
cfg = load_config()
W = int(cfg["vision"].get("frame_w", 640))
H = int(cfg["vision"].get("frame_h", 480))

source = CameraSource(width=W, height=H)
try:
    detector = VisionInputs(cfg, source=source)     # optional live cow check
except Exception as e:
    detector = None
    print("[calibrate_live] detection off:", e)

app = Flask(__name__)


def _floor_from(pts):
    """Build a FloorMap from 4 image points [[x,y]*4] using the config grid."""
    c = {"vision": {**cfg["vision"], "floor_quad_px": pts}}
    return FloorMap(c)


def _overlay(frame, pts):
    import cv2
    quad = np.array(pts, dtype=np.int32)
    cv2.polylines(frame, [quad.reshape(-1, 1, 2)], True, (0, 255, 255), 2)

    floor = _floor_from(pts)
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype="float32")
    Hmat = cv2.getPerspectiveTransform(floor.quad_px, unit)
    Hinv = np.linalg.inv(Hmat)

    def uv_px(u, v):
        p = Hinv @ np.array([u, v, 1.0])
        return int(p[0] / p[2]), int(p[1] / p[2])

    cv2.line(frame, uv_px(0.5, 0), uv_px(0.5, 1), (0, 180, 0), 1)       # left|right
    for v in (1 / 3, 2 / 3):                                            # near|mid|far
        cv2.line(frame, uv_px(0, v), uv_px(1, v), (0, 180, 0), 1)
    for col, u in (("left", 0.25), ("right", 0.75)):
        for row, v in (("near", 1 / 6), ("mid", 0.5), ("far", 5 / 6)):
            x, y = uv_px(u, v)
            cv2.putText(frame, floor.grid[col][row], (x - 12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)

    if detector is not None:                                           # live cow -> quadrant
        try:
            dpts, lying = detector._ground_points_and_classes(frame)
            if len(dpts):
                uv = floor.to_uv(dpts)
                for (px, py), (u, v), lie in zip(dpts, uv, lying):
                    q = floor.quadrant(float(np.clip(u, 0, 1)), float(np.clip(v, 0, 1)))
                    c = (0, 180, 255) if lie else (0, 0, 255)
                    cv2.circle(frame, (int(px), int(py)), 6, c, -1)
                    cv2.putText(frame, q, (int(px) + 6, int(py) + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        except Exception:
            pass
    return frame


def _current_pts():
    return [[float(x), float(y)] for x, y in cfg["vision"]["floor_quad_px"]]


@app.get("/frame")
def frame():
    import cv2
    f = source.frame()
    if f is None:
        return ("", 503)
    pts = request.args.get("pts")
    pts = [[float(v) for v in p.split(",")] for p in pts.split(";")] if pts else _current_pts()
    try:
        f = _overlay(f.copy(), pts)
    except Exception as e:
        cv2.putText(f, f"bad corners: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.post("/save")
def save():
    pts = request.json["pts"]                       # [[x,y]*4] in [near-left,near-right,far-right,far-left]
    ipts = [[int(round(x)), int(round(y))] for x, y in pts]
    literal = "[" + ", ".join(f"[{x}, {y}]" for x, y in ipts) + "]"
    p = path("config.yaml")
    text = open(p).read()
    new, n = re.subn(r"(\n[ \t]*floor_quad_px:[ \t]*)\[.*\]", r"\g<1>" + literal, text, count=1)
    if n == 0:
        return jsonify(ok=False, msg="floor_quad_px line not found in config.yaml"), 400
    open(p, "w").write(new)
    cfg["vision"]["floor_quad_px"] = ipts           # so the live view keeps the saved corners
    return jsonify(ok=True, saved=literal)


@app.get("/")
def index():
    pts = _current_pts()
    import json
    return f"""<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>Quadrant calibration</title><style>
 body{{margin:0;background:#111;color:#eee;font:15px system-ui;padding:10px;text-align:center}}
 #wrap{{position:relative;display:inline-block;touch-action:none}}
 #v{{display:block;width:{W}px;max-width:100%;border-radius:6px}}
 svg{{position:absolute;inset:0;width:100%;height:100%}}
 circle{{fill:#ff3b30;stroke:#fff;stroke-width:2;cursor:grab}}
 .bar{{margin:10px}} button{{font-size:15px;padding:8px 16px;margin:0 4px;border:0;border-radius:6px;background:#2563eb;color:#fff;cursor:pointer}}
 #msg{{color:#9aa;margin-left:8px}}
</style>
<h3>Drag the 4 dots to the floor corners</h3>
<div id=wrap>
 <img id=v>
 <svg id=s viewBox="0 0 {W} {H}" preserveAspectRatio="none">
  <circle id=p0 r=10></circle><circle id=p1 r=10></circle>
  <circle id=p2 r=10></circle><circle id=p3 r=10></circle>
 </svg>
</div>
<div class=bar>
 <button onclick="save()">Save to config</button>
 <span id=msg>near-left, near-right, far-right, far-left</span>
</div>
<script>
 const W={W},H={H}, names=['near-left','near-right','far-right','far-left'];
 let pts={json.dumps(pts)};
 const img=document.getElementById('v'), svg=document.getElementById('s');
 const dots=[0,1,2,3].map(i=>document.getElementById('p'+i));
 function place(){{dots.forEach((d,i)=>{{d.setAttribute('cx',pts[i][0]);d.setAttribute('cy',pts[i][1]);}});}}
 function refresh(){{img.src='/frame?pts='+pts.map(p=>p[0].toFixed(1)+','+p[1].toFixed(1)).join(';')+'&t='+Date.now();}}
 place(); refresh();
 setInterval(refresh, 1200);                       // keep the feed live while you drag
 let drag=-1;
 function toImg(e){{const r=svg.getBoundingClientRect();const x=(e.clientX-r.left)/r.width*W;const y=(e.clientY-r.top)/r.height*H;return [Math.max(0,Math.min(W,x)),Math.max(0,Math.min(H,y))];}}
 dots.forEach((d,i)=>d.addEventListener('pointerdown',e=>{{drag=i;d.setPointerCapture(e.pointerId);}}));
 svg.addEventListener('pointermove',e=>{{if(drag<0)return;pts[drag]=toImg(e);place();}});
 svg.addEventListener('pointerup',e=>{{if(drag<0)return;drag=-1;refresh();}});
 async function save(){{const r=await fetch('/save',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{pts}})}});
   const j=await r.json();document.getElementById('msg').textContent=j.ok?('saved '+j.saved):('ERROR: '+j.msg);}}
</script>"""


if __name__ == "__main__":
    print(f"open http://<pi-hostname>.local:{PORT}  (drag the corners, then Save)")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
