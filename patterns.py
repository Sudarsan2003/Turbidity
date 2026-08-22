"""
patterns.py
Generates the projected image patterns referenced on the project slide
(Fig 1b: "Image pattern template slide with pre-designed pattern").
Each function returns a BGR uint8 numpy array. A dispatcher + a __main__
block regenerate the reference PNGs under assets/.
"""

import numpy as np
import cv2

PATTERN_TYPES = ["checker", "vstripe", "hstripe", "dots", "circles", "random"]

PATTERN_NAMES = {
    "checker": "Checkerboard",
    "vstripe": "Vertical Stripes",
    "hstripe": "Horizontal Stripes",
    "dots": "Dot Grid",
    "circles": "Concentric Circles",
    "random": "Random Binary",
}


def _blank(w, h):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def generate_checkerboard(w, h, feat=24):
    img = _blank(w, h)
    for y in range(0, h, feat):
        for x in range(0, w, feat):
            if ((x // feat) + (y // feat)) % 2 == 0:
                cv2.rectangle(img, (x, y), (x + feat, y + feat), (0, 0, 0), -1)
    return img


def generate_vstripes(w, h, feat=24):
    img = _blank(w, h)
    for x in range(0, w, feat * 2):
        cv2.rectangle(img, (x, 0), (x + feat, h), (0, 0, 0), -1)
    return img


def generate_hstripes(w, h, feat=24):
    img = _blank(w, h)
    for y in range(0, h, feat * 2):
        cv2.rectangle(img, (0, y), (w, y + feat), (0, 0, 0), -1)
    return img


def generate_dots(w, h, feat=24):
    img = _blank(w, h)
    r = max(2, int(feat * 0.28))
    for y in range(feat, h, feat):
        for x in range(feat, w, feat):
            cv2.circle(img, (x, y), r, (0, 0, 0), -1)
    return img


def generate_circles(w, h, feat=24):
    img = _blank(w, h)
    cx, cy = w // 2, h // 2
    thickness = max(1, int(feat * 0.4))
    r = feat
    while r < max(w, h):
        cv2.circle(img, (cx, cy), r, (0, 0, 0), thickness)
        r += int(feat * 1.4)
    return img


def generate_random(w, h, feat=24, seed=None):
    rng = np.random.default_rng(seed)
    block = max(2, feat // 3)
    small_h = h // block + 1
    small_w = w // block + 1
    small = (rng.random((small_h, small_w)) < 0.5).astype(np.uint8) * 255
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    return img


_DISPATCH = {
    "checker": generate_checkerboard,
    "vstripe": generate_vstripes,
    "hstripe": generate_hstripes,
    "dots": generate_dots,
    "circles": generate_circles,
    "random": generate_random,
}


def generate_pattern(pattern_type, w=480, h=360, feat=24):
    if pattern_type not in _DISPATCH:
        raise ValueError(f"Unknown pattern type: {pattern_type}")
    return _DISPATCH[pattern_type](w, h, feat)


if __name__ == "__main__":
    # Regenerate the reference pattern assets mentioned in the project layout.
    import os
    out_dir = os.path.join(os.path.dirname(__file__), "assets")
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "checkerboard.png"),
                generate_checkerboard(480, 360, 24))
    cv2.imwrite(os.path.join(out_dir, "stripes.png"),
                generate_vstripes(480, 360, 24))
    cv2.imwrite(os.path.join(out_dir, "dots.png"),
                generate_dots(480, 360, 24))
    print(f"Wrote checkerboard.png, stripes.png, dots.png to {out_dir}")