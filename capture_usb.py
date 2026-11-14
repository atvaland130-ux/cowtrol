#!/usr/bin/env python3
"""
Capture training photos of the model cows from the USB webcam, for labeling
cow_standing / cow_lying and fine-tuning YOLO.

Run on the Pi (over SSH):
    source cowtrol/venv/bin/activate
    python3 capture_usb.py                 # press Enter for each photo, q + Enter to stop
    python3 capture_usb.py --every 3       # one photo every 3 s until Ctrl+C

Rearrange the cows between shots: mix standing and lying, clustered and spread out,
a few empty-barn shots, and different lighting. Shoot from roughly the angle/height
you plan to mount the camera at, so posture + position transfer. Aim for 150-300 photos.

Photos save to ./captures/ as JPGs. Copy them to your laptop to label:
    scp -r cowtrol@cowtrol.local:~/captures .
"""
import argparse, glob, os, time
import cv2

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")


def open_cam(w, h):
    for dev in sorted(glob.glob("/dev/v4l/by-id/*")) + sorted(glob.glob("/dev/video*")):
        cap = cv2.VideoCapture(dev)
        if cap.isOpened() and cap.read()[0]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            print("camera:", dev)
            return cap
        cap.release()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=0, help="seconds between photos (0 = press Enter each)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    cap = open_cam(args.width, args.height)
    if cap is None:
        print("No camera found (check the USB cable / power)."); return
    time.sleep(1)  # let exposure settle

    n = len([f for f in os.listdir(OUT) if f.endswith(".jpg")])
    print(f"Saving to {OUT} ({n} already there)")
    try:
        while True:
            if args.every:
                time.sleep(args.every)
            elif input("Enter = photo, q = quit: ").strip().lower() == "q":
                break
            for _ in range(3):           # flush stale frames so we grab a fresh one
                cap.read()
            ok, frame = cap.read()
            if not ok:
                print("read failed, retrying"); time.sleep(0.3); continue
            path = os.path.join(OUT, f"cow_{int(time.time()*1000)}.jpg")
            cv2.imwrite(path, frame)
            n += 1
            print(f"  saved {os.path.basename(path)}  (total {n})")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print(f"Done. {n} photos in {OUT}")


if __name__ == "__main__":
    main()
