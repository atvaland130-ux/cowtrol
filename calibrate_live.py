#!/usr/bin/env python3
"""Live, drag-to-set camera -> quadrant calibration (browser tool), nearest-centre version.
 
Open http://<pi>:8003 and you get the live feed with one draggable dot per quadrant (Q1-Q6).
Drop each dot in the middle of its quadrant; every cow the model sees is labelled with the
quadrant whose centre it's nearest to, so it lines up even with a corner mount / lens
distortion (where a straight 4-corner grid can't). Hit Save and it writes
vision.quadrant_centers_px into config.yaml (comments preserved).
 
Run on the Pi (stop demo.py / the controller first — one program holds the camera):
    cd ~/cowtrol
    source venv/bin/activate
    python3 calibrate_live.py
Then make sure vision.barn_wh_m is set (floor size in metres) and flip vision.enabled: true.
"""
import re
 
import numpy as np
from flask import Flask, Response, request, jsonify
 
from cowtrol_common import load_config, path
from vision_inputs import CameraSource, CenterMap, VisionInputs, quadrant_order
 
PORT = 8003
cfg = load_config()
W = int(cfg["vision"].get("frame_w", 640))
H = int(cfg["vision"].get("frame_h", 480))
QUADS = [q for q, _, _ in quadrant_order(cfg)]        # canonical order Q1..Q6
 
source = CameraSource(width=W, height=H)
try:
    detector = VisionInputs(cfg, source=source)       # for the YOLO model only
except Exception as e:
    detector = None
    print("[calibrate_live] detection off:", e)
 
app = Flask(__name__)
 
 
def _default_centers():
    c = cfg["vision"].get("quadrant_centers_px")
    if c:
        return [[float(c[q][0]), float(c[q][1])] for q in QUADS]
    xs = {"left": W * 0.25, "right": W * 0.75}
    ys = {"near": H * 0.8, "mid": H * 0.5, "far": H * 0.2}
    return [[xs[col], ys[row]] for _, col, row in quadrant_order(cfg)]
 
 
def _centermap(pts):
    c = {"vision": {**cfg["vision"], "quadrant_centers_px": {QUADS[i]: pts[i] for i in range(len(QUADS))}}}
    return CenterMap(c)
 
 
def _overlay(frame, pts):
    import cv2
    cmap = _centermap(pts)
    # the 6 centres
    for q, (x, y) in zip(QUADS, pts):
        cv2.circle(frame, (int(x), int(y)), 5, (0, 200, 0), -1)
        cv2.putText(frame, q, (int(x) + 7, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    # cows, each labelled with its nearest-centre quadrant
    if detector is not None:
        try:
            dpts, lying = detector._ground_points_and_classes(frame)
            if len(dpts):
                quads, _ = cmap.assign(dpts)
                for (px, py), q, lie in zip(dpts, quads, lying):
                    col = (0, 180, 255) if lie else (0, 0, 255)
                    cv2.circle(frame, (int(px), int(py)), 6, col, -1)
                    cv2.putText(frame, q, (int(px) + 6, int(py) + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        except Exception:
            pass
    return frame
 
 
@app.get("/frame")
def frame():
    import cv2
    f = source.frame()
    if f is None:
        return ("", 503)
    pts = request.args.get("pts")
    pts = [[float(v) for v in p.split(",")] for p in pts.split(";")] if pts else _default_centers()
    try:
        f = _overlay(f.copy(), pts)
    except Exception as e:
        cv2.putText(f, f"err: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(buf.tobytes(), mimetype="image/jpeg")
 
 
@app.post("/save")
def save():
    pts = request.json["pts"]
    ipts = {QUADS[i]: [int(round(pts[i][0])), int(round(pts[i][1]))] for i in range(len(QUADS))}
    block = "  quadrant_centers_px:\n" + "".join(f"    {q}: [{ipts[q][0]}, {ipts[q][1]}]\n" for q in QUADS)
    p = path("config.yaml")
    text = open(p).read()
    pat = r"\n  quadrant_centers_px:\n(?:    .*\n)*"          # existing key + its 4-space children
    if re.search(pat, text):
        new = re.sub(pat, "\n" + block, text, count=1)
    elif re.search(r"\nvision:\n", text):                    # insert at the top of the vision block
        new = re.sub(r"(\nvision:\n)", r"\g<1>" + block, text, count=1)
    else:
        return jsonify(ok=False, msg="no `vision:` block in config.yaml"), 400
    open(p, "w").write(new)
    cfg["vision"]["quadrant_centers_px"] = ipts
    return jsonify(ok=True, saved=ipts)
 
 
@app.get("/")
def index():
    import json
    pts = _default_centers()
    return f"""<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>Quadrant centres</title><style>
 body{{margin:0;background:#111;color:#eee;font:15px system-ui;padding:10px;text-align:center}}
 #wrap{{position:relative;display:inline-block;touch-action:none}}
 #v{{display:block;width:{W}px;max-width:100%;border-radius:6px}}
 svg{{position:absolute;inset:0;width:100%;height:100%}}
 circle{{fill:#22c55e;stroke:#fff;stroke-width:2;cursor:grab}}
 text{{fill:#bbf7d0;font:bold 13px system-ui;pointer-events:none}}
 .bar{{margin:10px}} button{{font-size:15px;padding:8px 16px;margin:0 4px;border:0;border-radius:6px;background:#2563eb;color:#fff;cursor:pointer}}
 #msg{{color:#9aa;margin-left:8px}}
</style>
<h3>Drag each dot to the centre of its quadrant</h3>
<div id=wrap>
 <img id=v>
 <svg id=s viewBox="0 0 {W} {H}" preserveAspectRatio="none"></svg>
</div>
<div class=bar><button onclick="save()">Save to config</button><span id=msg>Q1-Q6 centres</span></div>
<script>
 const W={W},H={H}, quads={json.dumps(QUADS)};
 let pts={json.dumps(pts)};
 const img=document.getElementById('v'), svg=document.getElementById('s');
 quads.forEach((q,i)=>{{
   const c=document.createElementNS('http://www.w3.org/2000/svg','circle');c.setAttribute('r',10);c.id='c'+i;
   const t=document.createElementNS('http://www.w3.org/2000/svg','text');t.id='t'+i;t.textContent=q;
   svg.appendChild(c);svg.appendChild(t);
 }});
 function place(){{quads.forEach((q,i)=>{{const c=document.getElementById('c'+i),t=document.getElementById('t'+i);
   c.setAttribute('cx',pts[i][0]);c.setAttribute('cy',pts[i][1]);t.setAttribute('x',pts[i][0]+13);t.setAttribute('y',pts[i][1]-10);}});}}
 function refresh(){{img.src='/frame?pts='+pts.map(p=>p[0].toFixed(1)+','+p[1].toFixed(1)).join(';')+'&t='+Date.now();}}
 place(); refresh(); setInterval(refresh,1200);
 let drag=-1;
 function toImg(e){{const r=svg.getBoundingClientRect();return [Math.max(0,Math.min(W,(e.clientX-r.left)/r.width*W)),Math.max(0,Math.min(H,(e.clientY-r.top)/r.height*H))];}}
 svg.addEventListener('pointerdown',e=>{{if(e.target.id&&e.target.id[0]=='c'){{drag=+e.target.id.slice(1);e.target.setPointerCapture(e.pointerId);}}}});
 svg.addEventListener('pointermove',e=>{{if(drag<0)return;pts[drag]=toImg(e);place();}});
 svg.addEventListener('pointerup',e=>{{if(drag<0)return;drag=-1;refresh();}});
 async function save(){{const r=await fetch('/save',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{pts}})}});
   const j=await r.json();document.getElementById('msg').textContent=j.ok?'saved':'ERROR: '+j.msg;}}
</script>"""
 
 
if __name__ == "__main__":
    print(f"open http://<pi-hostname>.local:{PORT}  (drag the 6 centres, then Save)")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
 
