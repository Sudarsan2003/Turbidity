\
\
\

import cv2
import numpy as np

def to_gray_u8(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return norm.astype(np.uint8)
    return img

def align_images(ref_img, cap_img, max_shift=50):
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

    h_ref, w_ref = ref_img.shape[:2]

    if cap_img.shape[:2] != (h_ref, w_ref):
        cap_img = cv2.resize(cap_img, (w_ref, h_ref), interpolation=cv2.INTER_AREA)

    ref_g = to_gray_u8(ref_img)
    cap_g = to_gray_u8(cap_img)

    ref_f32 = np.float32(ref_g)
    cap_f32 = np.float32(cap_g)

    h, w = ref_g.shape
    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(ref_f32, cap_f32, hann)
    dx, dy = shift

    if abs(dx) > max_shift or abs(dy) > max_shift:
        dx, dy = 0.0, 0.0

    warp_mat = np.float32([[1, 0, -dx], [0, 1, -dy]])
    aligned_bgr = cv2.warpAffine(cap_img, warp_mat, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    aligned_g = to_gray_u8(aligned_bgr)

    diff_g = cv2.absdiff(ref_g, aligned_g)
    diff_color = cv2.applyColorMap(diff_g, cv2.COLORMAP_MAGMA)

    overlay_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    overlay_bgr[:, :, 0] = ref_g
    overlay_bgr[:, :, 1] = ref_g
    overlay_bgr[:, :, 2] = aligned_g

    diff_mean = float(diff_g.mean())
    diff_std = float(diff_g.std())

    return {
        "aligned_bgr": aligned_bgr,
        "aligned_gray": aligned_g,
        "ref_gray": ref_g,
        "diff_gray": diff_g,
        "diff_color": diff_color,
        "overlay_bgr": overlay_bgr,
        "dx": float(dx),
        "dy": float(dy),
        "diff_mean": diff_mean,
        "diff_std": diff_std,
    }
