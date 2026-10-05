"""Noise floor and repeatability of the measurement.

Capture clear water N times (each capture = n_avg averaged frames, exactly like a
normal measurement), run every capture through the same pipeline against the same
reference, and look at the spread of each ratio feature.

  std            -> repeatability (noise floor) of that feature
  mean           -> offset of clear water vs the reference (bias, not noise)
  LOD (signal)   -> 3 * std
  LOD (NTU etc.) -> 3 * std / |calibration slope|   (if a calibration is given)

CLI (on the rig / Pi):
  python noise_floor.py --ref data/references/clear.png --n 10 --n-avg 5 [--target known_NTU]
"""

import argparse
import csv
import datetime
import os

import numpy as np

import data_manager

OUT_DIR = os.path.join(data_manager.DATA_DIR, "noise_floor")


def features_for_images(ref_img, images, analysis_size=(640, 360)):
    import preprocessing
    out = []
    for img in images:
        res = preprocessing.run_preprocessing_pipeline(ref_img, img, target_size=analysis_size)
        out.append(data_manager.features_from_proc(res))
    return out


def summarize(feature_dicts, calibration=None, unit=""):
    """feature_dicts: list of dicts from data_manager.features_from_proc."""
    feature_dicts = [f for f in feature_dicts if f]
    n = len(feature_dicts)
    if n < 3:
        raise ValueError("Need at least 3 clear-water captures.")
    rows = []
    for name in data_manager.REGRESSION_FEATURE_COLUMNS:
        v = np.array([f[name] for f in feature_dicts], dtype=float)
        mean, std = float(v.mean()), float(v.std(ddof=1))
        row = {"feature": name, "mean": mean, "std": std,
               "cv_pct": (std / abs(mean) * 100.0) if abs(mean) > 1e-3 else float("nan"),
               "min": float(v.min()), "max": float(v.max()),
               "lod_signal": 3.0 * std, "lod_target": float("nan")}
        if calibration and calibration["feature"] == name and abs(calibration["slope"]) > 1e-12:
            row["lod_target"] = 3.0 * std / abs(calibration["slope"])
        rows.append(row)
    return {"n": n, "rows": rows, "unit": unit,
            "calibration_feature": calibration["feature"] if calibration else None}


def format_report(s):
    lines = [f"Noise floor / repeatability, {s['n']} clear-water captures",
             f"{'feature':26s}{'mean':>9}{'std':>9}{'CV%':>8}{'LOD(3s)':>9}  LOD in target"]
    for r in s["rows"]:
        cv = "-" if r["cv_pct"] != r["cv_pct"] else f"{r['cv_pct']:.1f}"
        lt = "" if r["lod_target"] != r["lod_target"] else f"{r['lod_target']:.3g} {s['unit']}"
        mark = "  <- calibration feature" if r["feature"] == s["calibration_feature"] else ""
        lines.append(f"{r['feature']:26s}{r['mean']:9.4f}{r['std']:9.4f}{cv:>8}"
                     f"{r['lod_signal']:9.4f}  {lt}{mark}")
    lines.append("")
    lines.append("std = repeatability. mean = clear-water offset from the reference (bias).")
    if s["calibration_feature"] is None:
        lines.append("No calibration supplied: LOD is in signal units only. Run the calibration "
                     "curve to convert it to NTU / concentration.")
    return "\n".join(lines)


def save_csv(s, path=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = path or os.path.join(OUT_DIR, f"noise_floor_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "n", "mean", "std", "cv_pct", "min", "max", "lod_signal", "lod_target"])
        for r in s["rows"]:
            w.writerow([r["feature"], s["n"]] + [f"{r[k]:.6g}" for k in
                        ("mean", "std", "cv_pct", "min", "max", "lod_signal", "lod_target")])
    return path


def capture_clear_water(cam, n=10, n_avg=5, progress=None):
    import cv2
    imgs = []
    for i in range(n):
        acc = None
        for _ in range(max(1, n_avg)):
            fr = cam.read_frame().astype(np.float32)
            acc = fr if acc is None else acc + fr
        imgs.append(np.clip(acc / max(1, n_avg) + 0.5, 0, 255).astype(np.uint8))
        if progress:
            progress(i + 1, n)
    return imgs


def main():
    import cv2
    from camera import Camera
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="clear-water reference image")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--n-avg", type=int, default=5)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--target", default=None, help="known_NTU or particle_concentration: converts LOD to those units")
    a = ap.parse_args()
    ref = cv2.imdecode(np.fromfile(a.ref, dtype=np.uint8), cv2.IMREAD_COLOR)
    if ref is None:
        raise SystemExit(f"cannot read {a.ref}")
    cam = Camera(a.cam)
    try:
        imgs = capture_clear_water(cam, a.n, a.n_avg, lambda i, n: print(f"capture {i}/{n}", flush=True))
    finally:
        cam.release()
    cal, unit = None, ""
    if a.target:
        import calibration
        try:
            cal = calibration.calibration_curve(data_manager.load_experiments(), a.target)
            unit = "NTU" if a.target == "known_NTU" else ""
        except ValueError as e:
            print("calibration unavailable:", e)
    s = summarize(features_for_images(ref, imgs), cal, unit)
    print(format_report(s))
    print("saved", save_csv(s))


if __name__ == "__main__":
    main()