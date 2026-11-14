#!/usr/bin/env python3
"""
Browser-based photo capture for the USB camera (headless Pi friendly).

Open http://<pi-ip-or-hostname>:8002 and you get:
  - the live camera feed,
  - a Capture button (saves the current frame to ./captures/),
  - a gallery of everything you've shot, newest first, with a Delete on each.

Run:
    source cowtrol/venv/bin/activate
    python3 capture_web.py
Ctrl-C to stop. Only one program can hold the camera at a time.
Copy the photos off to label:  scp -r cowtrol@cowtrol.local:~/.../captures .
"""
import glob, os, time, threading
import cv2
from flask import Flask, Response, jsonify, send_from_directory

W, H, PORT = 1280, 960, 8002
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
os.makedirs(OUT, exist_ok=True)
state = {"frame": None}


def open_cam():
    for dev in sorted(glob.glob("/dev/v4l/by-id/*")) + sorted(glob.glob("/dev/video*")):
        cap = cv2.VideoCapture(dev)
        if cap.isOpened() and cap.read()[0]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
            print("camera opened:", dev)
            return cap
        cap.release()
    return None


def loop():
    cap = None
    while True:
        if cap is None:
            cap = open_cam()
            if cap is None:
                print("no camera, retrying..."); time.sleep(2); continue
        ok, frame = cap.read()
        if not ok:
            cap.release(); cap = None; time.sleep(0.5); continue
        state["frame"] = frame
        time.sleep(0.03)


app = Flask(__name__)


@app.get("/frame")
def frame():
    f = state["frame"]
    if f is None:
        return ("", 503)
    ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.post("/capture")
def capture():
    f = state["frame"]
    if f is None:
        return jsonify(error="no frame"), 503
    name = f"cow_{int(time.time()*1000)}.jpg"
    cv2.imwrite(os.path.join(OUT, name), f)
    return jsonify(name=name, count=len(_photos()))


def _photos():
    return sorted([f for f in os.listdir(OUT) if f.endswith(".jpg")], reverse=True)


@app.get("/list")
def list_photos():
    return jsonify(photos=_photos())


@app.get("/img/<name>")
def img(name):
    if name != os.path.basename(name):
        return ("", 400)
    return send_from_directory(OUT, name)


@app.post("/delete/<name>")
def delete(name):
    if name == os.path.basename(name):
        p = os.path.join(OUT, name)
        if os.path.exists(p):
            os.remove(p)
    return jsonify(photos=_photos())


@app.get("/")
def index():
    return """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>Capture</title><style>
 body{margin:0;background:#111;color:#eee;font:15px system-ui;padding:12px}
 .live{width:100%;max-width:720px;border-radius:8px;display:block;margin:0 auto 10px}
 .bar{text-align:center;margin-bottom:14px}
 button{font-size:16px;padding:10px 22px;border:0;border-radius:8px;background:#2563eb;color:#fff;cursor:pointer}
 #n{margin-left:10px;color:#9aa}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
 .cell{position:relative}
 .cell img{width:100%;border-radius:6px;display:block}
 .cell button{position:absolute;top:4px;right:4px;background:#c0392b;font-size:12px;padding:3px 8px;border-radius:5px}
</style>
<img class=live id=v>
<div class=bar><button onclick=cap()>Capture</button><span id=n></span></div>
<div class=grid id=g></div>
<script>
 const v=document.getElementById('v'),g=document.getElementById('g'),n=document.getElementById('n');
 setInterval(()=>{v.src='/frame?'+Date.now()},1000);
 async function refresh(){const r=await(await fetch('/list')).json();
   n.textContent=r.photos.length+' photos';
   g.innerHTML=r.photos.map(p=>`<div class=cell><img src="/img/${p}"><button onclick="del('${p}')">Delete</button></div>`).join('');}
 async function cap(){await fetch('/capture',{method:'POST'});refresh();}
 async function del(p){await fetch('/delete/'+p,{method:'POST'});refresh();}
 refresh();
</script>"""


threading.Thread(target=loop, daemon=True).start()
app.run(host="0.0.0.0", port=PORT)
