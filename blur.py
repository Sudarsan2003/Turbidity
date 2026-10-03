import numpy as np
import cv2

METHODS = ["laplacian", "tenengrad", "fft"]

def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return img.astype(np.float32)

def variance_of_laplacian(gray):

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(lap.var())

def tenengrad(gray):

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag_sq = gx * gx + gy * gy
    return float(mag_sq.mean())

def fft_high_freq_energy(gray, cutoff_frac=0.25):
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

    h, w = gray.shape

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
\
\
\

    gray = img_or_gray if img_or_gray.dtype == np.float32 and img_or_gray.ndim == 2 \
        else to_gray(img_or_gray)
    if method == "laplacian":
        return variance_of_laplacian(gray)
    elif method == "tenengrad":
        return tenengrad(gray)
    elif method == "fft":
        return fft_high_freq_energy(gray)
    raise ValueError(f"Unknown method: {method}. Choose from {METHODS}")

def compute_all_blur_scores(img_or_gray):

    gray = img_or_gray if img_or_gray.dtype == np.float32 and img_or_gray.ndim == 2 \
        else to_gray(img_or_gray)
    return {
        "laplacian": variance_of_laplacian(gray),
        "tenengrad": tenengrad(gray),
        "fft": fft_high_freq_energy(gray),
    }

def compute_blur_map(img_or_gray, method="laplacian"):
\
\

    gray = img_or_gray if img_or_gray.dtype == np.float32 and img_or_gray.ndim == 2 \
        else to_gray(img_or_gray)
    if method == "tenengrad":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return norm.astype(np.uint8)
    elif method == "fft":
        h, w = gray.shape
        win = np.outer(np.hanning(h), np.hanning(w))
        F = np.fft.fftshift(np.fft.fft2(gray * win))
        spec = np.log1p(np.abs(F))
        norm = cv2.normalize(spec, None, 0, 255, cv2.NORM_MINMAX)
        return norm.astype(np.uint8)
    else:
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        norm = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)
        return norm.astype(np.uint8)

if __name__ == "__main__":
    import sys
    from utils import load_image
    path = sys.argv[1] if len(sys.argv) > 1 else "assets/checkerboard.png"
    img = load_image(path)
    print(f"{path}")
    print(f"  variance of Laplacian : {blur_score(img, 'laplacian'):.2f}")
    print(f"  Tenengrad             : {blur_score(img, 'tenengrad'):.2f}")
    print(f"  FFT high-freq energy  : {blur_score(img, 'fft'):.2f}%")
