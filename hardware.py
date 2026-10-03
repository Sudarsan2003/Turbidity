\
\
\
\
\
\
\
\

import time

import cv2

import rigcore
from camera import Camera
from patterns import generate_pattern

class Projector:
    WIN = "pattern_out"

    def __init__(self, w=1280, h=720, x=1920, y=0, windowed=False,
                 manual=False, settle=0.5):
        self.w, self.h, self.x, self.y = int(w), int(h), int(x), int(y)
        self.windowed, self.manual, self.settle = windowed, manual, settle
        self._open = False
        self.current = None

    def show(self, pattern, feat):
        self.current = (pattern, int(feat))
        if self.manual:
            input(f">>> Place printed mask '{pattern}' (feature {feat}) in the beam, "
                  f"then press Enter... ")
            return
        img = generate_pattern(pattern, self.w, self.h, int(feat))
        if not self._open:
            cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
            cv2.moveWindow(self.WIN, self.x, self.y)
            if self.windowed:
                cv2.resizeWindow(self.WIN, self.w // 2, self.h // 2)
            else:
                cv2.setWindowProperty(self.WIN, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            self._open = True
        cv2.imshow(self.WIN, img)
        cv2.waitKey(1)
        time.sleep(self.settle)
        cv2.waitKey(1)

    def close(self):
        if self._open:
            cv2.destroyWindow(self.WIN)
            cv2.waitKey(1)
            self._open = False

def add_hw_args(ap):
    g = ap.add_argument_group("hardware")
    g.add_argument("--cam", type=int, default=0, help="camera index")
    g.add_argument("--exposure", type=float, default=None,
                   help="manual exposure (driver specific; Pi cam = microseconds)")
    g.add_argument("--gain", type=float, default=None)
    g.add_argument("--n-avg", type=int, default=10,
                   help="frames averaged per capture (use the same value everywhere)")
    g.add_argument("--proj-w", type=int, default=1280)
    g.add_argument("--proj-h", type=int, default=720)
    g.add_argument("--proj-x", type=int, default=1920)
    g.add_argument("--proj-y", type=int, default=0)
    g.add_argument("--proj-windowed", action="store_true")
    g.add_argument("--manual", action="store_true", help="printed masks, no projector")
    g.add_argument("--settle", type=float, default=0.5, help="seconds after pattern change")
    return ap

def open_hardware(args):
    cam = Camera(args.cam, 480, 360)
    rigcore.lock_camera(cam, args.exposure, args.gain)
    proj = Projector(args.proj_w, args.proj_h, args.proj_x, args.proj_y,
                     args.proj_windowed, args.manual, args.settle)
    return cam, proj

def flush(cam, n=3):
    for _ in range(n):
        cam.read_frame()
