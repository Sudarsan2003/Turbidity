"""
blur_matrix.py
Implements Fig 2 from the slides: divide the imager frame into an m x n grid
of kernels, compute a per-kernel sharpness value, and produce the blur
matrix (and a normalized blur-ratio matrix used for the heatmap and for
scattering-angle conversion).
"""

import numpy as np
import cv2

from blur import to_gray, variance_of_laplacian, tenengrad, fft_high_freq_energy

_METRIC_FNS = {
    "laplacian": variance_of_laplacian,
    "tenengrad": tenengrad,
    "fft": fft_high_freq_energy,
}


def compute_blur_matrix(img, grid_cols=12, method="laplacian"):
    """
    Split the image into grid_cols columns (rows chosen to keep cells
    roughly square, matching the imager's p x q aspect ratio) and compute
    a per-cell sharpness score.

    Returns: matrix (2D numpy array, rows x cols) of raw sharpness values.
    Higher value = sharper cell.
    """
    gray = to_gray(img)
    h, w = gray.shape
    grid_rows = max(2, round(grid_cols * h / w))
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


def to_blur_ratio(matrix):
    """
    Normalize a raw sharpness matrix to a 0..1 "blur ratio" where
    0 = sharpest cell in this frame, 1 = blurriest cell in this frame.
    This is a within-frame normalization (relative), not an absolute
    calibrated blur value.
    """
    vmin, vmax = matrix.min(), matrix.max()
    rng = (vmax - vmin) or 1.0
    sharpness_norm = (matrix - vmin) / rng
    return 1.0 - sharpness_norm


def matrix_to_heatmap(ratio_matrix, out_w=480, out_h=360, annotate=False):
    """
    Render a 0..1 matrix (0=sharp/blue, 1=blurry/red) as a BGR heatmap
    image, upscaled to (out_w, out_h) with cell gridlines.
    """
    rows, cols = ratio_matrix.shape
    # Map 0..1 -> OpenCV COLORMAP_JET-like blue->red via built-in colormap.
    scaled = np.clip(ratio_matrix * 255, 0, 255).astype(np.uint8)
    small_color = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    heat = cv2.resize(small_color, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    if annotate and rows * cols <= 140:
        cell_w, cell_h = out_w / cols, out_h / rows
        for r in range(rows):
            for c in range(cols):
                val = ratio_matrix[r, c]
                txt = f"{val:.2f}"
                x = int(c * cell_w + cell_w * 0.15)
                y = int(r * cell_h + cell_h * 0.6)
                cv2.putText(heat, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.32, (0, 0, 0), 1, cv2.LINE_AA)
    return heat


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