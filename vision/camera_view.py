"""Live camera view in the browser, with a Take photo button for collecting training photos.

Run on the Pi (over SSH), from ~/cowtrol_cbt with the venv active:
    python vision/camera_view.py                                   # live view + Take photo
    python vision/camera_view.py --model vision/best_ncnn_model    # also draws the model's detections
    python vision/camera_view.py --model yolo11n_ncnn_model        # stock model: draws only its "cow" boxes
Then open http://cowtrol.local:8001 on your laptop or phone. Ctrl+C in the SSH window stops it.
Photos are saved to vision/captures/ as JPGs. Copy them to your laptop with:
    scp -r cowtrol@cowtrol.local:~/cowtrol_cbt/vision/captures .
Only one program can use the camera at a time, so stop this before running controller.py with the camera enabled.
"""
import argparse
import io
import os
import threading
import time
from datetime import datetime

from flask import Flask, Response, jsonify
from picamera2 import Picamera2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "captures")

PAGE = """<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cowtrol camera</title><style>
body{font-family:sans-serif;background:#111;color:#eee;margin:0;padding:16px;text-align:center}
img{max-width:100%;border:1px solid #444}button{font-size:20px;padding:12px 28px;margin:12px}
</style></head><body><h2>Cowtrol camera</h2><img src="/stream"><br>
<button onclick="snap()">Take photo</button><div id="msg">%COUNT% photos saved</div>
<script>async function snap(){const r=await (await fetch('/snap',{method:'POST'})).json();
document.getElementById('msg').textContent=r.count+' photos saved (last: '+r.file+')';}</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="trained model folder or .pt file; draws its detections on the live view")
    ap.add_argument("--conf", type=float, default=0.25, help="lowest confidence to draw (try 0.1 for the stock model)")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    cam = Picamera2()
    cam.configure(cam.create_still_configuration(main={"size": (1280, 960)}))
    cam.start()
    time.sleep(2)  # let exposure and white balance settle
    lock = threading.Lock()

    model = None
    if args.model:
        import cv2
        import numpy as np
        from ultralytics import YOLO
        model = YOLO(args.model)
        # the stock model knows 80 things; keep only "cow" so people, chairs etc. are not drawn
        cow_only = [k for k, v in model.names.items() if v == "cow"] if "cow_standing" not in model.names.values() else None

    def jpeg():
        with lock:
            img = cam.capture_image("main").convert("RGB")
        if model is None:
            buf = io.BytesIO()
            img.resize((640, 480)).save(buf, format="JPEG", quality=80)
            return buf.getvalue()
        frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        drawn = model.predict(frame, imgsz=640, conf=args.conf, classes=cow_only, verbose=False)[0].plot()
        return cv2.imencode(".jpg", cv2.resize(drawn, (640, 480)))[1].tobytes()

    def count():
        return len([f for f in os.listdir(OUT) if f.endswith(".jpg")])

    app = Flask(__name__)

    @app.route("/")
    def index():
        return PAGE.replace("%COUNT%", str(count()))

    @app.route("/stream")
    def stream():
        def frames():
            while True:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg() + b"\r\n"
                time.sleep(0.1)
        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/snap", methods=["POST"])
    def snap():
        name = datetime.now().strftime("barn_%Y%m%d_%H%M%S.jpg")
        with lock:
            cam.capture_file(os.path.join(OUT, name))  # full 1280x960, no boxes drawn
        return jsonify(file=name, count=count())

    print(f"Open http://cowtrol.local:{args.port} (photos go to {OUT})")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
