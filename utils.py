"""
utils.py
General helpers shared across the project: file IO, synthetic blur/noise
for software-only testing (no physical rig needed), and lightweight chart
drawing done with OpenCV primitives so no matplotlib dependency is required
on an embedded target (Raspberry Pi / Jetson).
"""

import os
import time
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# File / path helpers
# ---------------------------------------------------------------------------

def ensure_dir(path):
    """Create a directory if it doesn't exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path


def timestamp():
    """Filesystem-safe timestamp string, e.g. 20260728_143210."""
    return time.strftime("%Y%m%d_%H%M%S")


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def save_image(path, img):
    ensure_dir(os.path.dirname(path) or ".")
    cv2.imwrite(path, img)
    return path


# ---------------------------------------------------------------------------
# Synthetic blur / noise -- lets the whole pipeline be validated in software
# before the physical LED/tube/imager rig exists.
# ---------------------------------------------------------------------------

def simulate_blur(img, radius_px, noise_pct=0.0):
    """
    Apply a Gaussian blur (as a stand-in for fluid forward-scattering) and
    optional additive noise to an image. radius_px controls blur strength;
    noise_pct is 0-100.
    """
    out = img.copy()
    r = int(round(radius_px))
    if r > 0:
        ksize = 2 * r + 1
        out = cv2.GaussianBlur(out, (ksize, ksize), sigmaX=r / 2.0)
    if noise_pct > 0:
        amt = (noise_pct / 100.0) * 40.0
        noise = np.random.normal(0, amt, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Lightweight chart rendering (OpenCV only -- no matplotlib)
# ---------------------------------------------------------------------------

def draw_line_chart(xs, ys, width=600, height=240, xlabel="", ylabel="",
                     color=(198, 214, 62), title=""):
    """
    Render a simple line chart to a BGR numpy image. Used for the blur-radius
    sensitivity sweep so results stay inspectable without a heavy plotting
    dependency.
    """
    pad_l, pad_r, pad_t, pad_b = 55, 20, 30, 40
    img = np.full((height, width, 3), (17, 20, 24), dtype=np.uint8)

    if title:
        cv2.putText(img, title, (pad_l, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (217, 225, 232), 1, cv2.LINE_AA)

    if len(xs) < 2:
        return img

    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if y1 == y0:
        y1 = y0 + 1
    if x1 == x0:
        x1 = x0 + 1

    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def to_px(x, y):
        px = pad_l + int((x - x0) / (x1 - x0) * plot_w)
        py = pad_t + plot_h - int((y - y0) / (y1 - y0) * plot_h)
        return px, py

    # gridlines
    for i in range(5):
        gy = pad_t + int(plot_h * i / 4)
        cv2.line(img, (pad_l, gy), (width - pad_r, gy), (35, 42, 50), 1)

    # axis labels
    cv2.putText(img, xlabel, (pad_l, height - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (125, 138, 151), 1, cv2.LINE_AA)
    cv2.putText(img, ylabel, (5, pad_t - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (125, 138, 151), 1, cv2.LINE_AA)

    pts = [to_px(x, y) for x, y in zip(xs, ys)]
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 3, color, -1, cv2.LINE_AA)

    return img


def draw_bar_chart(labels, values, width=600, height=260, color=(198, 214, 62),
                    title=""):
    """Horizontal bar chart -- used for the pattern-sensitivity comparison."""
    img = np.full((height, width, 3), (17, 20, 24), dtype=np.uint8)
    if title:
        cv2.putText(img, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (217, 225, 232), 1, cv2.LINE_AA)

    n = len(labels)
    if n == 0:
        return img
    max_v = max(values) if max(values) > 0 else 1
    top, bottom = 34, height - 10
    bar_h = (bottom - top) / n * 0.6
    gap = (bottom - top) / n

    label_w = 110
    bar_x0 = label_w
    bar_x1 = width - 60

    for i, (lbl, v) in enumerate(zip(labels, values)):
        y = top + i * gap
        cv2.putText(img, lbl[:14], (5, int(y + bar_h)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (217, 225, 232), 1, cv2.LINE_AA)
        bw = int((bar_x1 - bar_x0) * (v / max_v))
        cv2.rectangle(img, (bar_x0, int(y)), (bar_x0 + bw, int(y + bar_h)),
                       color, -1)
        cv2.putText(img, f"{v*100:.0f}%", (bar_x0 + bw + 6, int(y + bar_h)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (125, 138, 151), 1, cv2.LINE_AA)

    return img