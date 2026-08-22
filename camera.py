"""
camera.py
Camera capture abstraction. Uses the laptop webcam via OpenCV's VideoCapture
by default. On a Raspberry Pi with picamera2 installed, the same interface
is backed by the Pi camera instead -- app.py and everything downstream never
need to know which backend is active, which is the point of keeping this
embedded-ready (Fig 1a's imager is the same code path either way).
"""

import time
import numpy as np
import cv2

try:
    from picamera2 import Picamera2
    _HAS_PICAMERA2 = True
except ImportError:
    _HAS_PICAMERA2 = False


class Camera:
    """
    Unified camera interface.

    backend="auto"  -> use Pi camera if picamera2 is available, else webcam
    backend="webcam" -> force OpenCV VideoCapture
    backend="picam"   -> force Raspberry Pi camera (requires picamera2)
    """

    def __init__(self, index=0, width=480, height=360, backend="auto"):
        self.width = width
        self.height = height
        if backend == "auto":
            backend = "picam" if _HAS_PICAMERA2 else "webcam"
        self.backend = backend

        if backend == "picam":
            if not _HAS_PICAMERA2:
                raise RuntimeError("picamera2 is not installed on this system.")
            self._picam = Picamera2()
            cfg = self._picam.create_preview_configuration(
                main={"size": (width, height), "format": "RGB888"})
            self._picam.configure(cfg)
            self._picam.start()
            time.sleep(0.5)  # let auto-exposure settle
            self._cap = None
        else:
            self._cap = cv2.VideoCapture(index)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if not self._cap.isOpened():
                raise RuntimeError(
                    "Could not open webcam. Check the camera index / permissions.")
            self._picam = None

    def read_frame(self):
        """Return a BGR uint8 frame resized to (width, height)."""
        if self.backend == "picam":
            frame = self._picam.capture_array()  # RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ok, frame = self._cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame from webcam.")
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        return frame

    def release(self):
        if self.backend == "picam" and self._picam is not None:
            self._picam.stop()
        elif self._cap is not None:
            self._cap.release()


if __name__ == "__main__":
    # Quick standalone smoke test: grab one frame and save it.
    cam = Camera()
    frame = cam.read_frame()
    cv2.imwrite("captured/camera_test.png", frame)
    cam.release()
    print("Saved captured/camera_test.png -- backend:", cam.backend)