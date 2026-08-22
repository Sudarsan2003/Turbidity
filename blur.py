"""
blur.py
Whole-image blur/sharpness estimation. Three metrics are implemented,
matching the "implement an algorithm to estimate image blur" objective on
the slide (the FFT metric additionally covers the "DSP techniques
(FFT/Matrix operations)" required-skills line):

  - variance of Laplacian  (fast, standard, works well on textured patterns)
  - Tenengrad (Sobel gradient energy)  (more robust to low-contrast regions)
  - FFT high-frequency energy ratio  (frequency-domain sharpness measure)

All three return a *sharpness* score where higher = sharper. Blur matrices
and the scattering-angle conversion (blur_matrix.py / scattering.py) build
on these.
"""

import numpy as np
import cv2

METHODS = ["laplacian", "tenengrad", "fft"]


def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return img.astype(np.float32)


def variance_of_laplacian(gray):
    """Classic blur metric: higher variance -> sharper image."""
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(lap.var())


def tenengrad(gray):
    """Mean squared gradient magnitude via Sobel -- higher = sharper."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag_sq = gx * gx + gy * gy
    return float(mag_sq.mean())


def fft_high_freq_energy(gray, cutoff_frac=0.25):
    """
    Frequency-domain sharpness measure (DSP/FFT technique called out in the
    slide's "Required Skills"). Blurring is a low-pass filtering operation,
    so a sharp image retains proportionally more energy at high spatial
    frequencies than a blurred one. This computes:

        score = (energy outside a central low-frequency disk) / (total energy)

    via 2D FFT magnitude, expressed as a percentage. Higher = sharper.

    cutoff_frac: radius of the "low frequency" disk as a fraction of the
    half-diagonal of frequency space (0..1). Frequencies inside that disk
    are excluded from the numerator.
    """
    h, w = gray.shape
    # Hann window reduces edge/boundary spectral leakage before the FFT.
    win = np.outer(np.hanning(h), np.hanning(w))
    windowed = gray * win

    F = np.fft.fft2(windowed)
    F_shifted = np.fft.fftshift(F)
    magnitude_sq = np.abs(F_shifted) ** 2

    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_dist = np.sqrt(cy ** 2 + cx ** 2)
    low_freq_mask = dist <= (cutoff_frac * max_dist)

    total_energy = magnitude_sq.sum()
    if total_energy <= 0:
        return 0.0
    high_freq_energy = magnitude_sq[~low_freq_mask].sum()
    return float(100.0 * high_freq_energy / total_energy)


def blur_score(img_or_gray, method="laplacian"):
    """
    Convenience entry point. Accepts either a BGR image or a precomputed
    grayscale float32 array.
    """
    gray = img_or_gray if img_or_gray.dtype == np.float32 and img_or_gray.ndim == 2 \
        else to_gray(img_or_gray)
    if method == "laplacian":
        return variance_of_laplacian(gray)
    elif method == "tenengrad":
        return tenengrad(gray)
    elif method == "fft":
        return fft_high_freq_energy(gray)
    raise ValueError(f"Unknown method: {method}. Choose from {METHODS}")


if __name__ == "__main__":
    import sys
    from utils import load_image
    path = sys.argv[1] if len(sys.argv) > 1 else "assets/checkerboard.png"
    img = load_image(path)
    print(f"{path}")
    print(f"  variance of Laplacian : {blur_score(img, 'laplacian'):.2f}")
    print(f"  Tenengrad             : {blur_score(img, 'tenengrad'):.2f}")
    print(f"  FFT high-freq energy  : {blur_score(img, 'fft'):.2f}%")