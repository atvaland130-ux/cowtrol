#!/usr/bin/env python3
"""Calibrate the camera->quadrant mapping AFTER the camera is mounted.

It grabs one frame and saves an annotated image showing:
  - the floor quad you set in config (vision.floor_quad_px), as a yellow outline,
  - the 2x3 quadrant grid with Q1..Q6 labels,
  - every detected cow as a dot, labelled with the quadrant it was assigned to.

Workflow:
  1. Mount the camera in its final spot.
  2. Put a cow (or a marker) in each of the 4 floor CORNERS.
  3. Run:  python3 calibrate_vision.py
  4. Open calibration.jpg. Read off the pixel coordinates of the 4 floor corners
     (the image is 640x480 by default; top-left is 0,0) and set them in config.yaml:
         vision.floor_quad_px: [[near-left], [near-right], [far-right], [far-left]]
     where NEAR = the row closest to the fans (Q1/Q4), LEFT = the Q1-Q3 side.
  5. Measure the real floor and set vision.barn_wh_m: [width_m, length_m].
  6. Re-run until each cow's dot is labelled with the correct quadrant.
  7. Set vision.enabled: true and restart controller.py.

Only one program can hold the camera at a time, so stop demo.py / the controller first.
"""
import sys

import numpy as np

from cowtrol_common import load_config
from vision_inputs import CameraSource, FloorMap, VisionInputs


def main():
    import cv2
    cfg = load_config()
    if "vision" not in cfg:
        sys.exit("No `vision:` block in config.yaml yet — add it first (see the sample).")

    src = CameraSource(width=int(cfg["vision"].get("frame_w", 640)),
                       height=int(cfg["vision"].get("frame_h", 480)))
    print("warming up camera...")
    frame = None
    import time
    for _ in range(50):
        frame = src.frame()
        if frame is not None:
            break
        time.sleep(0.1)
    if frame is None:
        sys.exit("No camera frame (check the cable/power and that nothing else holds the camera).")

    floor = FloorMap(cfg)
    quad = floor.quad_px.astype(int)

    # floor outline
    cv2.polylines(frame, [quad.reshape(-1, 1, 2)], True, (0, 255, 255), 2)
    labels = {"near-left": quad[0], "near-right": quad[1], "far-right": quad[2], "far-left": quad[3]}
    for name, (x, y) in labels.items():
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)
        cv2.putText(frame, name, (int(x) + 6, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # quadrant grid: map a dense set of unit-square points back to pixels to draw the thirds
    import cv2 as _cv
    Hinv = np.linalg.inv(_cv.getPerspectiveTransform(
        floor.quad_px, np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype="float32")))

    def uv_to_px(u, v):
        p = Hinv @ np.array([u, v, 1.0])
        return int(p[0] / p[2]), int(p[1] / p[2])

    for u in (0.5,):                         # left|right divider
        cv2.line(frame, uv_to_px(u, 0), uv_to_px(u, 1), (0, 180, 0), 1)
    for v in (1 / 3, 2 / 3):                 # near|mid|far dividers
        cv2.line(frame, uv_to_px(0, v), uv_to_px(1, v), (0, 180, 0), 1)
    for col, u in (("left", 0.25), ("right", 0.75)):
        for row, v in (("near", 1 / 6), ("mid", 0.5), ("far", 5 / 6)):
            qx, qy = uv_to_px(u, v)
            cv2.putText(frame, floor.grid[col][row], (qx - 10, qy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)

    # detected cows, each labelled with its assigned quadrant
    try:
        vis = VisionInputs(cfg, source=src)
        pts, lying = vis._ground_points_and_classes(frame)
        if len(pts):
            uv = floor.to_uv(pts)
            for (px, py), (u, v), lie in zip(pts, uv, lying):
                q = floor.quadrant(float(np.clip(u, 0, 1)), float(np.clip(v, 0, 1)))
                c = (0, 180, 255) if lie else (0, 0, 255)
                cv2.circle(frame, (int(px), int(py)), 6, c, -1)
                cv2.putText(frame, q, (int(px) + 6, int(py) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
            print(f"detected {len(pts)} cows; clustering={vis._clustering(floor.metres(uv)):.2f}")
        else:
            print("no cows detected (put some in view to check quadrant assignment)")
    except Exception as e:
        print("detection skipped:", e)

    cv2.imwrite("calibration.jpg", frame)
    print("wrote calibration.jpg — open it and adjust vision.floor_quad_px until the labels line up.")


if __name__ == "__main__":
    main()
