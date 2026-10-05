"""Camera access for the turbidity rig (Raspberry Pi camera or USB/laptop webcam).

The camera runs at its native 1920x1080 (16:9). Frames are returned at the size
the camera really delivers, with no stretching. The analysis pipeline then
shrinks them to ANALYSIS_SIZE, which has the SAME 16:9 shape, so the pattern is
never distorted.
"""

import os
import sys
import time

import numpy as np
import cv2

try:
    from picamera2 import Picamera2
    _HAS_PICAMERA2 = True
except ImportError:
    _HAS_PICAMERA2 = False

CAPTURE_SIZE = (1920, 1080)    # (width, height) requested from the camera
ANALYSIS_SIZE = (640, 360)     # (width, height) used for the blur analysis, also 16:9


def to_analysis_size(img, size=ANALYSIS_SIZE):
    """Shrink (or enlarge) an image to the analysis size. INTER_AREA keeps edges clean."""
    w, h = size
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def save_png(img, path):
    """Lossless PNG save that creates the folder and raises if it fails.
    (cv2.imwrite fails silently on a missing folder or on non-ASCII paths.)"""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise IOError(f"Could not encode PNG: {path}")
    buf.tofile(path)
    return path


def _as_bgr(frame):
    """Always hand back an HxWx3 uint8 BGR array."""
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = frame[:, :, :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


class Camera:
    """Opens the camera at (width, height); self.width / self.height always hold the
    ACTUAL frame size the camera delivers, which may differ from what was requested."""

    def __init__(self, index=0, width=CAPTURE_SIZE[0], height=CAPTURE_SIZE[1],
                 backend="auto", warmup=5):
        self.width = int(width)
        self.height = int(height)
        if backend == "auto":
            backend = "picam" if _HAS_PICAMERA2 else "webcam"
        self.backend = backend
        self._picam = None
        self._cap = None

        if backend == "picam":
            if not _HAS_PICAMERA2:
                raise RuntimeError("picamera2 is not installed on this system.")
            self._picam = Picamera2()
            cfg = self._picam.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"})
            self._picam.configure(cfg)
            self._picam.start()
            time.sleep(0.5)
        else:
            if sys.platform == "darwin":
                self._cap = cv2.VideoCapture(int(index), cv2.CAP_AVFOUNDATION)
            else:
                self._cap = cv2.VideoCapture(int(index))
            if not self._cap.isOpened():
                self._cap.release()
                self._cap = None
                raise RuntimeError(
                    "Could not open the webcam. Check the camera index, and on macOS allow "
                    "camera access for Terminal/Python in System Settings > Privacy & Security > Camera.")
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Discard the first frames (auto exposure settles) and learn the real frame size.
        first = None
        for _ in range(max(1, warmup)):
            try:
                first = self.read_frame()
            except RuntimeError:
                time.sleep(0.1)
        if first is None:
            self.release()
            raise RuntimeError("Camera opened but returned no frames.")

    def read_frame(self, retries=5):
        """Return one BGR frame at the camera's real size (never stretched)."""
        frame = None
        for _ in range(max(1, retries)):
            if self.backend == "picam":
                frame = self._picam.capture_array()
            else:
                ok, frame = self._cap.read()
                if not ok:
                    frame = None
            if frame is not None and frame.size > 0:
                break
            time.sleep(0.05)
        if frame is None or frame.size == 0:
            raise RuntimeError("Failed to read a frame from the camera.")
        frame = _as_bgr(frame)
        self.height, self.width = frame.shape[:2]
        return frame

    def save_png(self, path, frame=None):
        """Capture (or take the given frame) and save it losslessly as PNG."""
        return save_png(self.read_frame() if frame is None else frame, path)

    def release(self):
        if self.backend == "picam" and self._picam is not None:
            self._picam.stop()
            self._picam = None
        elif self._cap is not None:
            self._cap.release()
            self._cap = None


if __name__ == "__main__":
    cam = Camera()
    print(f"Camera delivers {cam.width} x {cam.height}  (backend: {cam.backend})")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captured", "camera_test.png")
    cam.save_png(out)
    cam.release()
    print("Saved", out)