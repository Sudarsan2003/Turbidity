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
\
\
\
\
\

import csv
import json
import math
import os
import re
import time

import cv2
import numpy as np

import blur_matrix
import rigcore
from blur import blur_score, to_gray
from utils import ensure_dir

DIR = ensure_dir(os.path.join(rigcore.RES, "turbidity"))
REF_DIR = ensure_dir(os.path.join(DIR, "references"))
FRAME_DIR = ensure_dir(os.path.join(DIR, "frames"))
CSV_PATH = os.path.join(DIR, "data.csv")
MODEL_PATH = os.path.join(DIR, "model.json")
REF_META = os.path.join(DIR, "current_reference.json")

METHODS = rigcore.METHODS
RAW_KEYS = ("laplacian", "tenengrad", "fft", "mean", "std")
STAT_KEYS = ("blur_mean", "blur_std", "angle_mean", "angle_std")
FEATURES = ([f"{k}_{m}" for m in METHODS for k in
             ("rel_ratio_mean", "rel_ratio_std", "rel_ratio_p90",
              "rel_angle_mean", "g_ratio", "d_angle_mean")]
            + ["int_ratio", "contrast_ratio"])
META = ["timestamp", "frame_name", "frame_image", "source", "frames_averaged",
        "ref_name", "ref_image", "ref_id", "grid_cols", "algorithm_selected",
        "concentration", "conc_unit", "ntu", "sat_frac"]
RAWC = [f"{w}_{k}" for w in ("sample", "ref") for k in RAW_KEYS]
STATC = [f"{w}_{k}_{m}" for w in ("sample", "ref") for m in METHODS for k in STAT_KEYS]
COLUMNS = META + RAWC + STATC + FEATURES
TEXT_COLS = {"timestamp", "frame_name", "frame_image", "source", "ref_name",
             "ref_image", "ref_id", "algorithm_selected", "conc_unit"}

def _save_png(img, path):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise IOError("could not encode image")
    buf.tofile(path)

def _read_png(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

def raw_values(img):
    g = to_gray(img)
    return {"laplacian": blur_score(g, "laplacian"), "tenengrad": blur_score(g, "tenengrad"),
            "fft": blur_score(g, "fft"), "mean": float(g.mean()), "std": float(g.std())}

def within_angle(mat):

    return np.degrees(np.arctan(blur_matrix.to_blur_ratio(mat)))

def rel_ratio(ms, mr, mask):
    r = np.clip(1.0 - ms / np.maximum(mr, 1e-9), 0.0, 1.0)
    return np.where(mask, r, 0.0)

def angle_pair(mr, ms, mask, mode):
\

    if mode == "ref":
        if mr is None or ms is None:
            return (None if mr is None else np.zeros_like(mr)), None
        return np.zeros_like(mr), np.degrees(np.arctan(rel_ratio(ms, mr, mask)))
    return (None if mr is None else within_angle(mr)), (None if ms is None else within_angle(ms))

def set_reference(img, name, source):
    ref_id = time.strftime("%Y%m%d_%H%M%S")
    fn = f"ref_{ref_id}.png"
    _save_png(img, os.path.join(REF_DIR, fn))
    with open(REF_META, "w") as fh:
        json.dump({"id": ref_id, "name": name, "image": fn, "source": source}, fh)
    return ref_id

def get_reference(grid_cols=rigcore.GRID_COLS):
    if not os.path.exists(REF_META):
        return None
    meta = json.load(open(REF_META))
    p = os.path.join(REF_DIR, meta["image"])
    img = _read_png(p) if os.path.exists(p) else None
    if img is None:
        return None
    ref = rigcore.make_reference(img, "ref", 0, grid_cols)
    ref.update({"id": meta["id"], "name": meta["name"], "file": meta["image"],
                "source": meta["source"], "raw": raw_values(img)})
    return ref

def analyze(img, ref):

    if img.shape[:2] != ref["image"].shape[:2]:
        img = cv2.resize(img, (ref["image"].shape[1], ref["image"].shape[0]),
                         interpolation=cv2.INTER_AREA)
    mats = rigcore.matrices(img, ref["grid_cols"])
    raw = raw_values(img)
    feat, stats = {}, {}
    for m in METHODS:
        ms, mr, mask = mats[m], ref["mats"][m], ref["masks"][m]
        was, war = within_angle(ms), within_angle(mr)
        for who, mat, ang in (("sample", ms, was), ("ref", mr, war)):
            stats[f"{who}_blur_mean_{m}"] = float(mat.mean())
            stats[f"{who}_blur_std_{m}"] = float(mat.std())
            stats[f"{who}_angle_mean_{m}"] = float(ang.mean())
            stats[f"{who}_angle_std_{m}"] = float(ang.std())
        if mask.any():
            r = np.clip(1.0 - ms[mask] / np.maximum(mr[mask], 1e-9), 0.0, 1.0)
            feat[f"rel_ratio_mean_{m}"] = float(r.mean())
            feat[f"rel_ratio_std_{m}"] = float(r.std())
            feat[f"rel_ratio_p90_{m}"] = float(np.percentile(r, 90))
            feat[f"rel_angle_mean_{m}"] = float(np.degrees(np.arctan(r)).mean())
        else:
            for k in ("rel_ratio_mean", "rel_ratio_std", "rel_ratio_p90", "rel_angle_mean"):
                feat[f"{k}_{m}"] = float("nan")
        feat[f"g_ratio_{m}"] = 1.0 - raw[m] / max(ref["raw"][m], 1e-9)
        feat[f"d_angle_mean_{m}"] = float(was.mean() - war.mean())
    feat["int_ratio"] = raw["mean"] / max(ref["raw"]["mean"], 1e-6)
    feat["contrast_ratio"] = raw["std"] / max(ref["raw"]["std"], 1e-6)
    return {"image": img, "mats": mats, "raw": raw, "feat": feat, "stats": stats,
            "sat": rigcore.clip_fraction(img)}

def clean_name(name):
    return re.sub(r"[^\w\-.]+", "_", name.strip()).strip("._") or "frame"

def unique_frame_name(name):
    base = clean_name(name)
    if base.lower().endswith(".png"):
        base = base[:-4]
    cand, i = base, 2
    while os.path.exists(os.path.join(FRAME_DIR, cand + ".png")):
        cand, i = f"{base}_{i}", i + 1
    return cand

def next_auto_name():
    i = 1
    while os.path.exists(os.path.join(FRAME_DIR, f"frame_{i:03d}.png")):
        i += 1
    return f"frame_{i:03d}"

def save_row(an, ref, name, source, n_avg, algorithm, conc, unit, ntu):

    name = unique_frame_name(name)
    _save_png(an["image"], os.path.join(FRAME_DIR, name + ".png"))
    row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "frame_name": name,
           "frame_image": name + ".png", "source": source, "frames_averaged": int(n_avg),
           "ref_name": ref["name"], "ref_image": ref["file"], "ref_id": ref["id"],
           "grid_cols": ref["grid_cols"], "algorithm_selected": algorithm,
           "concentration": "" if conc is None else float(conc), "conc_unit": unit,
           "ntu": "" if ntu is None else float(ntu), "sat_frac": an["sat"]}
    row.update({f"sample_{k}": v for k, v in an["raw"].items()})
    row.update({f"ref_{k}": v for k, v in ref["raw"].items()})
    row.update(an["stats"])
    row.update(an["feat"])
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(COLUMNS)
        w.writerow([f"{row.get(c, ''):.6g}" if isinstance(row.get(c, ""), float)
                    else row.get(c, "") for c in COLUMNS])
    return row

def load_rows():
    if not os.path.exists(CSV_PATH):
        return []
    rows = []
    with open(CSV_PATH, newline="") as fh:
        for r in csv.DictReader(fh):
            for c in COLUMNS:
                if c not in TEXT_COLS:
                    try:
                        r[c] = float(r.get(c, ""))
                    except ValueError:
                        r[c] = float("nan")
            rows.append(r)
    return rows

def algo_of(feature):
    for m in METHODS:
        if feature.endswith("_" + m):
            return m
    return "intensity"

def _rank(a):
    order = np.argsort(a, kind="mergesort")
    rk = np.empty(len(a))
    rk[order] = np.arange(len(a))
    for v in np.unique(a):
        idx = a == v
        if idx.sum() > 1:
            rk[idx] = rk[idx].mean()
    return rk

def _monotonic(coef, mu, sd, lo, hi):
    d = np.diff(np.polyval(coef, (np.linspace(lo, hi, 60) - mu) / sd))
    return bool(np.all(d >= -1e-9) or np.all(d <= 1e-9))

def candidates(rows, target, names):
    rows = [r for r in rows if not math.isnan(r[target])]
    y_all = np.array([r[target] for r in rows])
    if len(rows) < 4 or len(np.unique(y_all)) < 3:
        raise RuntimeError(f"need >= 4 saved rows with a '{target}' value and >= 3 different "
                           f"levels (have {len(rows)} rows, {len(np.unique(y_all))} levels)")
    out = []
    for name in names:
        x_all = np.array([r[name] for r in rows])
        ok = ~np.isnan(x_all)
        x, y = x_all[ok], y_all[ok]
        if len(x) < 4 or np.ptp(x) < 1e-9:
            continue
        mu, sd = float(x.mean()), float(x.std())
        z = (x - mu) / sd
        sxx = np.polyfit(y, x, 1)[0]
        blank = x[y == 0]
        noise = float(np.std(blank, ddof=1)) if len(blank) >= 3 else float("nan")
        rho = float(np.corrcoef(_rank(x), _rank(y))[0, 1])
        for deg in (1, 2):
            if deg == 2 and (len(z) < 6 or len(np.unique(y)) < 4):
                continue
            A = np.vander(z, deg + 1)
            G = np.linalg.pinv(A.T @ A)
            beta = G @ A.T @ y
            res = y - A @ beta
            h = np.clip(np.einsum("ij,jk,ik->i", A, G, A), 0, 0.999)
            loo = float(np.sqrt(np.mean((res / (1 - h)) ** 2)))
            if not _monotonic(beta, mu, sd, x.min(), x.max()):
                continue
            sst = float(((y - y.mean()) ** 2).sum()) or 1.0
            out.append({"feature": name, "algorithm": algo_of(name), "deg": deg, "mu": mu,
                        "sd": sd, "coef": [float(b) for b in beta], "loo_rmse": loo,
                        "nloo": loo / max(float(np.ptp(y)), 1e-9),
                        "r2": float(1 - (res ** 2).sum() / sst), "rho": rho,
                        "noise": noise,
                        "lod": 3 * noise / abs(sxx) if noise == noise and sxx else float("nan"),
                        "x_min": float(x.min()), "x_max": float(x.max()), "n": int(len(x))})
    if not out:
        raise RuntimeError("no usable feature - do the images differ between the levels?")
    out.sort(key=lambda c: c["loo_rmse"])
    return out

def compare_algorithms(rows, target):

    cands = candidates(rows, target, FEATURES)
    best = {}
    for c in cands:
        best.setdefault(c["algorithm"], c)
    lines = [f"Algorithm comparison  (target = {target}, lower error = better)",
             f"{'algorithm':10s} {'best feature':26s}{'deg':>3} {'R2':>6} {'err/range':>9} "
             f"{'trend':>6} {'LOD':>8}"]
    for a in list(METHODS) + ["intensity"]:
        c = best.get(a)
        if c:
            lod = "n/a" if c["lod"] != c["lod"] else f"{c['lod']:.3g}"
            lines.append(f"{a:10s} {c['feature']:26s}{c['deg']:>3} {c['r2']:6.3f} "
                         f"{c['nloo']*100:8.1f}% {c['rho']:6.2f} {lod:>8}")
    ranked = [best[a] for a in METHODS if a in best]
    rec = min(ranked, key=lambda c: c["nloo"])["algorithm"] if ranked else None
    lines.append("")
    lines.append(f"RECOMMENDED: {rec}" if rec else "no blur algorithm produced a usable feature")
    lines.append("(err/range = leave-one-out error as % of the concentration range; "
                 "trend = rank correlation, +-1 is a perfectly steady rise/fall;")
    lines.append(" LOD needs >= 3 clean-water rows with target 0. 'intensity' = brightness only, "
                 "shown for reference.)")
    return best, "\n".join(lines), rec

def fit(rows, target, algorithm=None, unit=""):

    names = FEATURES if algorithm in (None, "auto") else \
        [f for f in FEATURES if algo_of(f) == algorithm]
    cands = candidates(rows, target, names)
    m = dict(cands[0])
    rr = [r for r in rows if not math.isnan(r[target])]
    y = np.array([r[target] for r in rr])
    x = np.array([r[m["feature"]] for r in rr])
    pred = np.polyval(m["coef"], (x - m["mu"]) / m["sd"])
    blank = pred[(y == 0) & ~np.isnan(x)]
    m.update({"target": target, "unit": unit, "y_max": float(y.max()),
              "lod": 3 * float(np.std(blank, ddof=1)) if len(blank) >= 3 else float("nan"),
              "n_refs": len({r["ref_id"] for r in rr})})
    with open(MODEL_PATH, "w") as fh:
        json.dump(m, fh, indent=2)
    lines = [f"data: {len(rr)} rows, {len(np.unique(y))} levels of {target}",
             "top features:"]
    for c in cands[:5]:
        lines.append(f"  {c['feature']:26s} deg{c['deg']}  R2={c['r2']:.3f}  "
                     f"LOO error={c['loo_rmse']:.3g}")
    lines.append(f"USING {m['feature']} (degree {m['deg']}, {m['algorithm']})")
    lines.append(f"detection limit: {m['lod']:.3g} {unit}" if m["lod"] == m["lod"]
                 else "detection limit: save >= 3 clean-water rows with target 0")
    if m["n_refs"] > 1:
        lines.append(f"WARNING: rows use {m['n_refs']} different reference images; "
                     "features are only comparable within one reference.")
    return m, "\n".join(lines)

def load_model():
    return json.load(open(MODEL_PATH)) if os.path.exists(MODEL_PATH) else None

def predict(m, feat):
    x = feat[m["feature"]]
    v = float(np.polyval(m["coef"], (x - m["mu"]) / m["sd"]))
    span = max(m["x_max"] - m["x_min"], 1e-9)
    flags = []
    if x < m["x_min"] - 0.1 * span or x > m["x_max"] + 0.1 * span:
        flags.append("outside calibrated range")
    if v > 1.1 * m["y_max"]:
        flags.append("above highest calibration level")
    lod = m.get("lod", float("nan"))
    return {"value": max(v, 0.0), "flags": flags, "below_lod": lod == lod and v < lod}

def plot_fit(m, rows, w=520, h=360):
    img = np.full((h, w, 3), (23, 29, 37), dtype=np.uint8)
    t = m["target"]
    pts = [(r[m["feature"]], r[t]) for r in rows
           if not math.isnan(r[m["feature"]]) and not math.isnan(r[t])]
    if not pts:
        return img
    xs, ys = zip(*pts)
    x0, x1 = min(xs + (m["x_min"],)), max(xs + (m["x_max"],))
    y0, y1 = min(0, min(ys)), (max(ys) * 1.1 or 1)
    l, r_, tp, b = 55, 15, 30, 42
    px = lambda x: int(l + (x - x0) / max(x1 - x0, 1e-9) * (w - l - r_))
    py = lambda y: int(h - b - (y - y0) / max(y1 - y0, 1e-9) * (h - tp - b))
    cv2.rectangle(img, (l, tp), (w - r_, h - b), (60, 70, 80), 1)
    g = np.linspace(x0, x1, 60)
    cv2.polylines(img, [np.array([[px(a), py(c)] for a, c in
                                  zip(g, np.polyval(m["coef"], (g - m["mu"]) / m["sd"]))])],
                  False, (75, 184, 242), 2, cv2.LINE_AA)
    for a, c in pts:
        cv2.circle(img, (px(a), py(c)), 4, (198, 214, 62), -1, cv2.LINE_AA)
    put = lambda s, p: cv2.putText(img, s, p, cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                                   (200, 210, 220), 1, cv2.LINE_AA)
    put(f"x: {m['feature']}", (l, h - 12))
    put(f"y: {t} {m.get('unit', '')}", (5, 18))
    put(f"R2={m['r2']:.3f}", (w - 100, 18))
    return img
