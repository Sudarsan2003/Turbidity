\
\
\
\
\
\

import numpy as np
import cv2

from blur import to_gray, variance_of_laplacian, tenengrad, fft_high_freq_energy

_METRIC_FNS = {
    "laplacian": variance_of_laplacian,
    "tenengrad": tenengrad,
    "fft": fft_high_freq_energy,
}

def grid_from_kernel(img_h, img_w, kernel_m, kernel_n):
    """Spec Fig 2: imager of p x q pixels, kernel of m x n pixels
    -> blur matrix of (p/m) x (q/n).  kernel_m = kernel height (rows),
    kernel_n = kernel width (cols), both in pixels."""
    kernel_m, kernel_n = max(2, int(kernel_m)), max(2, int(kernel_n))
    return max(1, img_h // kernel_m), max(1, img_w // kernel_n)

def compute_blur_matrix(img, grid_cols=12, grid_rows=None, method="laplacian", kernel=None):
\
\
\
\
\
\
\

    gray = to_gray(img)
    h, w = gray.shape
    if kernel is not None:
        grid_rows, grid_cols = grid_from_kernel(h, w, kernel[0], kernel[1])
    if grid_rows is None:
        grid_rows = max(2, round(grid_cols * h / w))
    else:
        grid_rows = max(2, int(grid_rows))
    grid_cols = max(2, int(grid_cols))

    cell_w = w / grid_cols
    cell_h = h / grid_rows

    matrix = np.zeros((grid_rows, grid_cols), dtype=np.float64)
    if method not in _METRIC_FNS:
        raise ValueError(f"Unknown method: {method}. Choose from {list(_METRIC_FNS)}")
    fn = _METRIC_FNS[method]

    for r in range(grid_rows):
        y0, y1 = int(r * cell_h), int((r + 1) * cell_h)
        for c in range(grid_cols):
            x0, x1 = int(c * cell_w), int((c + 1) * cell_w)
            cell = gray[y0:y1, x0:x1]
            matrix[r, c] = fn(cell) if cell.size > 0 else 0.0

    return matrix

def compute_all_blur_matrices(img, grid_cols=12, grid_rows=None, kernel=None):
\
\
\

    return {
        m: compute_blur_matrix(img, grid_cols=grid_cols, grid_rows=grid_rows, method=m, kernel=kernel)
        for m in ("laplacian", "tenengrad", "fft")
    }

def matrix_stats(mat):
\
\
\

    if mat is None or mat.size == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "median": 0.0}
    return {
        "mean": float(np.mean(mat)),
        "min": float(np.min(mat)),
        "max": float(np.max(mat)),
        "std": float(np.std(mat)),
        "median": float(np.median(mat)),
    }

def compute_scattering_angle_matrix(blur_mat, ref_mat=None, mode="ratio"):
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

    if ref_mat is not None and mode != "within":
        if blur_mat.shape != ref_mat.shape:

            ref_mat_resized = cv2.resize(ref_mat, (blur_mat.shape[1], blur_mat.shape[0]),
                                         interpolation=cv2.INTER_LINEAR)
        else:
            ref_mat_resized = ref_mat
        ratio = np.clip(1.0 - blur_mat / np.maximum(ref_mat_resized, 1e-9), 0.0, 1.0)
    else:
        ratio = to_blur_ratio(blur_mat)

    angle_deg = np.degrees(np.arctan(ratio))
    return angle_deg

def draw_grid_overlay(img, grid_cols=12, grid_rows=None, color=(0, 255, 255), thickness=1):
\
\

    out = img.copy()
    h, w = out.shape[:2]
    if grid_rows is None:
        grid_rows = max(2, round(grid_cols * h / w))
    cell_w = w / grid_cols
    cell_h = h / grid_rows

    for c in range(1, grid_cols):
        x = int(c * cell_w)
        cv2.line(out, (x, 0), (x, h), color, thickness)
    for r in range(1, grid_rows):
        y = int(r * cell_h)
        cv2.line(out, (0, y), (w, y), color, thickness)

    return out

def to_blur_ratio(matrix):
\
\
\
\
\

    vmin, vmax = matrix.min(), matrix.max()
    rng = (vmax - vmin) or 1.0
    sharpness_norm = (matrix - vmin) / rng
    return 1.0 - sharpness_norm

def matrix_to_heatmap(matrix, vmin=None, vmax=None, cmap=cv2.COLORMAP_JET,
                      out_w=480, out_h=360, annotate=False):
\
\

    rows, cols = matrix.shape
    if vmin is None:
        vmin = float(matrix.min())
    if vmax is None:
        vmax = float(matrix.max())
    span = (vmax - vmin) if (vmax - vmin) > 1e-9 else 1.0

    norm = np.clip((matrix - vmin) / span, 0.0, 1.0)
    scaled = (norm * 255).astype(np.uint8)
    small_color = cv2.applyColorMap(scaled, cmap)
    heat = cv2.resize(small_color, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    cell_w = out_w / cols
    cell_h = out_h / rows
    for c in range(cols + 1):
        x = int(c * cell_w)
        cv2.line(heat, (x, 0), (x, out_h), (30, 30, 30), 1)
    for r in range(rows + 1):
        y = int(r * cell_h)
        cv2.line(heat, (0, y), (out_w, y), (30, 30, 30), 1)

    if annotate and rows * cols <= 140:
        for r in range(rows):
            for c in range(cols):
                val = matrix[r, c]
                txt = f"{val:.1f}" if abs(val) < 1000 else f"{val:.0f}"
                x = int(c * cell_w + cell_w * 0.15)
                y = int(r * cell_h + cell_h * 0.6)
                cv2.putText(heat, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.28, (0, 0, 0), 1, cv2.LINE_AA)
    return heat

def save_matrix_csv(matrix, filepath, header=None):

    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savetxt(filepath, matrix, delimiter=",", fmt="%.6f", header=header or "")

def load_matrix_csv(filepath):

    return np.loadtxt(filepath, delimiter=",", comments="#")

if __name__ == "__main__":
    import sys
    from utils import load_image, save_image
    path = sys.argv[1] if len(sys.argv) > 1 else "assets/checkerboard.png"
    img = load_image(path)
    matrix = compute_blur_matrix(img, grid_cols=12, method="laplacian")
    ratio = to_blur_ratio(matrix)
    heat = matrix_to_heatmap(ratio, annotate=True)
    save_image("results/blur_matrix_demo.png", heat)
    print("Blur matrix shape:", matrix.shape)
    print("Mean blur ratio:", ratio.mean())
    print("Saved results/blur_matrix_demo.png")