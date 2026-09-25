"""
scattering.py
Converts the blur matrix into scattering angles, following the slide's
formula: Scattering Angle = tan^-1(Blur Value).

The blur values fed in here are the normalized 0..1 blur-ratio matrix from
blur_matrix.py, so angles come out in the 0-45 degree range. This is a
software-only placeholder mapping: it demonstrates the pipeline end-to-end,
but converting normalized blur into a physically accurate scattering angle
requires calibrating against the real optical rig (Fig 1a) with known
particle standards.
"""

import numpy as np


def blur_ratio_to_angle(ratio_matrix):
    """Elementwise tan^-1(blur_ratio) in degrees."""
    return np.degrees(np.arctan(ratio_matrix))


def summarize_angles(angle_matrix):
    return {
        "mean_deg": float(angle_matrix.mean()),
        "min_deg": float(angle_matrix.min()),
        "max_deg": float(angle_matrix.max()),
        "std_deg": float(angle_matrix.std()),
    }


if __name__ == "__main__":
    from blur_matrix import compute_blur_matrix, to_blur_ratio
    from utils import load_image
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "assets/checkerboard.png"
    img = load_image(path)
    matrix = compute_blur_matrix(img, grid_cols=12)
    ratio = to_blur_ratio(matrix)
    angles = blur_ratio_to_angle(ratio)
    stats = summarize_angles(angles)
    print("Scattering angle summary (deg):", stats)