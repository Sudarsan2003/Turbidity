\
\
\
\
\
\
\
\
\
\
\
\

import csv
import os

import cv2
import numpy as np

import blur_matrix
from blur import to_gray
from utils import ensure_dir, timestamp

BASE = os.path.dirname(os.path.abspath(__file__))
RES = ensure_dir(os.path.join(BASE, "results"))
REF_DIR = ensure_dir(os.path.join(RES, "baselines"))
CAP_DIR = ensure_dir(os.path.join(BASE, "captured"))
SWEEP_CSV = os.path.join(RES, "sweep.csv")

METHODS = ("laplacian", "tenengrad", "fft")
GRID_COLS = 12
FEATURE_NAMES = ([f"{m}_{s}" for m in METHODS
                  for s in ("ratio_mean", "ratio_std", "ratio_p90", "angle_mean")]
                 + ["int_ratio", "contrast_ratio"])
META_COLS = ["timestamp", "fluid", "label", "ntu_ref", "particle_um", "pattern",
             "feat", "grid_cols", "rep", "is_blank", "sat_frac", "mean_int", "image"]
SWEEP_COLUMNS = META_COLS + FEATURE_NAMES

def lock_camera(cam, exposure=None, gain=None):
\
\

    cap = getattr(cam, "_cap", None)
    if cap is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        if exposure is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        if gain is not None:
            cap.set(cv2.CAP_PROP_GAIN, gain)
    elif getattr(cam, "_picam", None) is not None:
        ctl = {"AeEnable": False, "AwbEnable": False}
        if exposure is not None:
            ctl["ExposureTime"] = int(exposure)
        if gain is not None:
            ctl["AnalogueGain"] = float(gain)
        cam._picam.set_controls(ctl)
    for _ in range(8):
        cam.read_frame()

def capture_average(cam, n=10):
\
\

    acc = None
    for _ in range(max(1, n)):
        fr = cam.read_frame().astype(np.float32)
        acc = fr if acc is None else acc + fr
    avg = np.clip(acc / max(1, n) + 0.5, 0, 255).astype(np.uint8)
    return avg, clip_fraction(avg)

def clip_fraction(img):
    return float((img.max(axis=2) >= 253).mean()) if img.ndim == 3 \
        else float((img >= 253).mean())

def matrices(img_u8, grid_cols=GRID_COLS):
    return {m: blur_matrix.compute_blur_matrix(img_u8, grid_cols=grid_cols, method=m)
            for m in METHODS}

def make_reference(img_u8, pattern="", feat=0, grid_cols=GRID_COLS):
    mats = matrices(img_u8, grid_cols)
    masks = {m: mats[m] > 0.05 * mats[m].max() for m in METHODS}
    g = to_gray(img_u8)
    return {"pattern": pattern, "feat": int(feat), "grid_cols": grid_cols,
            "image": img_u8, "mats": mats, "masks": masks,
            "mean_int": float(g.mean()), "std_int": float(g.std())}

def ref_path(pattern, feat):
    return os.path.join(REF_DIR, f"{pattern}_f{int(feat)}.png")

def save_reference_image(img_u8, pattern, feat):
    ok, buf = cv2.imencode(".png", img_u8)
    if not ok:
        raise IOError("could not encode reference image")
    buf.tofile(ref_path(pattern, feat))

def load_reference(pattern, feat, grid_cols=GRID_COLS):
    p = ref_path(pattern, feat)
    if not os.path.exists(p):
        return None
    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    return None if img is None else make_reference(img, pattern, feat, grid_cols)

def frame_features(img_u8, ref):

    rh, rw = ref["image"].shape[:2]
    if img_u8.shape[:2] != (rh, rw):
        img_u8 = cv2.resize(img_u8, (rw, rh), interpolation=cv2.INTER_AREA)
    mats = matrices(img_u8, ref["grid_cols"])
    f = {}
    for m in METHODS:
        mask = ref["masks"][m]
        if mask.any():
            r = np.clip(1.0 - mats[m][mask] / np.maximum(ref["mats"][m][mask], 1e-9),
                        0.0, 1.0)
            f[f"{m}_ratio_mean"] = float(r.mean())
            f[f"{m}_ratio_std"] = float(r.std())
            f[f"{m}_ratio_p90"] = float(np.percentile(r, 90))
            f[f"{m}_angle_mean"] = float(np.degrees(np.arctan(r)).mean())
        else:
            for s in ("ratio_mean", "ratio_std", "ratio_p90", "angle_mean"):
                f[f"{m}_{s}"] = float("nan")
    g = to_gray(img_u8)
    f["int_ratio"] = float(g.mean() / max(ref["mean_int"], 1e-6))
    f["contrast_ratio"] = float(g.std() / max(ref["std_int"], 1e-6))
    f["mean_int"] = float(g.mean())
    return f

def append_sweep_row(row):
    new = not os.path.exists(SWEEP_CSV)
    with open(SWEEP_CSV, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(SWEEP_COLUMNS)
        out = []
        for c in SWEEP_COLUMNS:
            v = row.get(c, "")
            out.append(f"{v:.6g}" if isinstance(v, float) else v)
        w.writerow(out)
