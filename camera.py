"""Camera access for the health check (Alt 2). Returns grayscale frames as 2-D uint8 arrays.

Real camera: Raspberry Pi camera module through picamera2, or a USB camera through OpenCV (config.yaml ->
safety_checks -> camera -> source: picamera2 | usb). The reference frame is the camera's normal view; it is saved
to camera_reference.npy and can be re-taken from the dashboard after the camera is mounted.
SimCamera (--simulate) draws a fake barn view; the dashboard's test buttons can cover or move it.
"""
import os

import numpy as np

from cowtrol_common import path


class Camera:
    def __init__(self, cfg):
        c = cfg["safety_checks"]["camera"]
        self.ref_file = path(c.get("reference_file", "camera_reference.npy"))
        self.cam, self.kind = None, c.get("source", "picamera2")
        if self.kind == "picamera2":
            from picamera2 import Picamera2
            self.cam = Picamera2()
            self.cam.configure(self.cam.create_still_configuration(main={"size": (640, 480)}))
            self.cam.start()
        else:
            import cv2
            self.cv2 = cv2
            self.cam = cv2.VideoCapture(int(c.get("usb_index", 0)))
        self.reference = np.load(self.ref_file) if os.path.exists(self.ref_file) else None

    def frame(self):
        """Grayscale frame, or None if the camera returned nothing (counts as blocked)."""
        if self.kind == "picamera2":
            rgb = self.cam.capture_array()
            return np.asarray(rgb[..., :3].mean(axis=2), dtype=np.uint8)
        ok, img = self.cam.read()
        return self.cv2.cvtColor(img, self.cv2.COLOR_BGR2GRAY) if ok else None

    def set_reference(self, frame):
        self.reference = frame
        np.save(self.ref_file, frame)


class SimCamera(Camera):
    """Fake camera for --simulate. state: ok | covered | moved."""

    def __init__(self, cfg):
        self.ref_file = path(cfg["safety_checks"]["camera"].get("reference_file", "camera_reference.npy"))
        self.state = "ok"
        rng = np.random.default_rng(1)
        y, x = np.mgrid[0:480, 0:640]
        self.scene = np.clip(90 + 60 * np.sin(x / 50) * np.cos(y / 70) + (x > 320) * 40, 0, 255)
        self.noise = rng
        self.reference = self._view(0)

    def _view(self, shift):
        img = np.roll(self.scene, shift, axis=1) + self.noise.normal(0, 4, self.scene.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    def frame(self):
        if self.state == "covered":
            return np.full((480, 640), 4, dtype=np.uint8)
        return self._view(170 if self.state == "moved" else 0)

    def set_reference(self, frame):
        self.reference = frame
