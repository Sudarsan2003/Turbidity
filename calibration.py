import math

import numpy as np

import data_manager
import regression_engine as RE

LOD_K = 3.0
LOQ_K = 10.0


def _loo_inverse_rmse(sig, tgt):
    """LOO RMSE when predicting the target from the signal (the way it is used)."""
    n = len(sig)
    A = np.column_stack([sig, np.ones(n)])
    G = np.linalg.pinv(A.T @ A)
    beta = G @ A.T @ tgt
    h = np.clip(np.einsum("ij,jk,ik->i", A, G, A), 0, 0.999)
    return float(np.sqrt(np.mean(((tgt - A @ beta) / (1 - h)) ** 2)))


def _fit_one(sig, tgt):
    slope, icpt = np.polyfit(tgt, sig, 1)
    res = sig - (slope * tgt + icpt)
    sst = float(np.sum((sig - sig.mean()) ** 2))
    r2 = float(1 - np.sum(res ** 2) / sst) if sst > 1e-18 else float("nan")
    blank = sig[tgt == 0]
    sigma = float(np.std(blank, ddof=1)) if len(blank) >= 3 else float("nan")
    lod = LOD_K * sigma / abs(slope) if sigma == sigma and abs(slope) > 1e-12 else float("nan")
    loq = LOQ_K * sigma / abs(slope) if sigma == sigma and abs(slope) > 1e-12 else float("nan")
    return {"slope": float(slope), "intercept": float(icpt), "r2": r2,
            "sigma_blank": sigma, "lod": float(lod), "loq": float(loq),
            "n_blank": int(len(blank)),
            "loo_rmse": _loo_inverse_rmse(sig, tgt) if abs(slope) > 1e-12 else float("inf")}


def calibration_curve(records, target="known_NTU", feature=None):
    """Fit the calibration line. feature=None picks the ratio feature with the lowest
    leave-one-out error. Returns a dict (including the points) or raises ValueError."""
    X, y, names = RE.prepare_data(records, target)
    n = 0 if X is None else len(X)
    if n < 4 or len(np.unique(y)) < 3:
        raise ValueError(f"Need >= 4 saved rows with '{target}' and >= 3 different levels "
                         f"(found {n} usable rows).")
    cols = [names.index(feature)] if feature else range(len(names))
    best = None
    for j in cols:
        sig = X[:, j]
        if np.ptp(sig) < 1e-9:
            continue
        f = _fit_one(sig, y)
        if best is None or f["loo_rmse"] < best[1]["loo_rmse"]:
            best = (j, f)
    if best is None:
        raise ValueError("No usable feature: the images do not differ between levels.")
    j, f = best
    f.update({"feature": names[j], "target": target, "n": int(n),
              "x": y.tolist(), "y": X[:, j].tolist(),
              "target_max": float(y.max())})
    return f


def render_calibration_plot(cal, width=620, height=320, unit=""):
    import cv2
    img = np.full((height, width, 3), (23, 29, 37), dtype=np.uint8)
    if not cal:
        cv2.putText(img, "No calibration yet", (width // 3, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 150, 160), 1)
        return img
    xs, ys = np.array(cal["x"]), np.array(cal["y"])
    l, r, t, b = 62, 18, 34, 58
    x0, x1 = 0.0, max(xs.max(), 1e-9) * 1.05
    ymin, ymax = min(ys.min(), 0.0), ys.max()
    pad = (ymax - ymin) * 0.08 or 1.0
    y0, y1 = ymin - pad * (ymin < 0), ymax + pad
    px = lambda x: int(l + (x - x0) / (x1 - x0) * (width - l - r))
    py = lambda y: int(height - b - (y - y0) / max(y1 - y0, 1e-12) * (height - t - b))
    grey, txt = (140, 150, 160), (200, 210, 220)
    cv2.rectangle(img, (l, t), (width - r, height - b), (60, 70, 80), 1)
    for k in range(5):
        gx = x0 + (x1 - x0) * k / 4
        gy = y0 + (y1 - y0) * k / 4
        cv2.putText(img, f"{gx:.3g}", (px(gx) - 12, height - b + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, grey, 1)
        cv2.putText(img, f"{gy:.3g}", (4, py(gy) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, grey, 1)
        cv2.line(img, (l, py(gy)), (width - r, py(gy)), (35, 42, 50), 1)
    # fitted line
    p1 = (px(x0), py(cal["slope"] * x0 + cal["intercept"]))
    p2 = (px(x1), py(cal["slope"] * x1 + cal["intercept"]))
    cv2.line(img, p1, p2, (75, 184, 242), 2, cv2.LINE_AA)
    # LOD marker
    if cal["lod"] == cal["lod"] and x0 <= cal["lod"] <= x1:
        lx = px(cal["lod"])
        for yy in range(t, height - b, 8):
            cv2.line(img, (lx, yy), (lx, min(yy + 4, height - b)), (242, 184, 75), 1)
        cv2.putText(img, "LOD", (lx + 3, t + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (242, 184, 75), 1)
    for a, c in zip(xs, ys):
        cv2.circle(img, (px(a), py(c)), 4, (198, 214, 62), -1, cv2.LINE_AA)
    u = f" ({unit})" if unit else ""
    lod_s = f"{cal['lod']:.3g}{' ' + unit if unit else ''}" if cal["lod"] == cal["lod"] else "n/a (need >=3 blanks)"
    cv2.putText(img, f"Calibration: {cal['feature']}", (l, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(img, f"R2={cal['r2']:.3f}   LOD={lod_s}   LOO err={cal['loo_rmse']:.3g}   n={cal['n']}",
                (l, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (62, 214, 198), 1, cv2.LINE_AA)
    cv2.putText(img, f"{cal['target']}{u}", (width // 2 - 40, height - b + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt, 1)
    cv2.putText(img, "signal", (4, t - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt, 1)
    return img


def summary_text(cal, unit=""):
    lod = f"{cal['lod']:.3g} {unit}" if cal["lod"] == cal["lod"] else "n/a (needs >= 3 blank rows with target 0)"
    loq = f"{cal['loq']:.3g} {unit}" if cal["loq"] == cal["loq"] else "n/a"
    return (f"feature {cal['feature']}\n"
            f"signal = {cal['slope']:.4g} * {cal['target']} + {cal['intercept']:.4g}\n"
            f"R2 = {cal['r2']:.4f}   n = {cal['n']}   LOO error = {cal['loo_rmse']:.3g} {unit}\n"
            f"blank sigma = {cal['sigma_blank']:.3g} ({cal['n_blank']} blanks)\n"
            f"LOD (3 sigma) = {lod}   LOQ (10 sigma) = {loq}")