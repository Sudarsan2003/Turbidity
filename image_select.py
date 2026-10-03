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
\
\

import argparse
import os
import sys

import cv2
import numpy as np

import rigcore

EXTS = [("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")]
SIZE = (480, 360)

def _dialog(multi, title, initialdir, parent):
    from tkinter import filedialog
    root = None
    if parent is None:
        import tkinter as tk
        root = parent = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
    try:
        fn = filedialog.askopenfilenames if multi else filedialog.askopenfilename
        res = fn(parent=parent, title=title, filetypes=EXTS,
                 initialdir=initialdir or (rigcore.CAP_DIR if os.path.isdir(rigcore.CAP_DIR)
                                           else rigcore.BASE))
    finally:
        if root is not None:
            root.destroy()
    return list(res) if multi else (res or None)

def pick_image(parent=None, title="Select an image", initialdir=None):
\

    return _dialog(False, title, initialdir, parent)

def pick_images(parent=None, title="Select image(s)", initialdir=None):
    return _dialog(True, title, initialdir, parent)

def load_bgr(path):
\

    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    resized = (img.shape[1], img.shape[0]) != SIZE
    if resized:
        img = cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA)
    return img, resized

def reference_from_file(path, pattern="custom", feat=0, grid_cols=rigcore.GRID_COLS):
    img, _ = load_bgr(path)
    return rigcore.make_reference(img, pattern, feat, grid_cols)

def save_as_reference(path, pattern, feat):

    img, resized = load_bgr(path)
    rigcore.save_reference_image(img, pattern, feat)
    return rigcore.ref_path(pattern, feat), resized

def analyze(img, ref, model=None):

    f = rigcore.frame_features(img, ref)
    return f, (model.predict(f) if model is not None else None)

def analyze_file(path, ref, model=None):
    img, resized = load_bgr(path)
    f, res = analyze(img, ref, model)
    return {"path": path, "image": img, "resized": resized, "features": f, "result": res}

def _load_model():
    try:
        from model import TurbidityModel
        return TurbidityModel.load()
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*")
    ap.add_argument("--ref", help="clear-water image file")
    ap.add_argument("--ref-dialog", action="store_true", help="pick the clear-water image")
    ap.add_argument("--pattern")
    ap.add_argument("--feat", type=int)
    ap.add_argument("--save-ref", action="store_true",
                    help="store the selected image as the reference for --pattern/--feat")
    a = ap.parse_args()

    if a.save_ref:
        if not a.pattern or a.feat is None:
            sys.exit("--save-ref needs --pattern and --feat")
        p = a.images[0] if a.images else pick_image(title="Select the CLEAR-WATER image")
        if not p:
            return
        out, rs = save_as_reference(p, a.pattern, a.feat)
        print(f"stored {p} -> {out}" + ("  (was resized to 480x360)" if rs else ""))
        return

    model = _load_model()
    if a.ref_dialog and not a.ref:
        a.ref = pick_image(title="Select the CLEAR-WATER reference image")
        if not a.ref:
            return
    if a.ref:
        ref = reference_from_file(a.ref)
        print(f"reference: {a.ref}")
    else:
        pat = a.pattern or (model.pattern if model else None)
        ft = a.feat if a.feat is not None else (model.feat if model else None)
        if pat is None:
            sys.exit("no model.json - give --ref FILE or --pattern/--feat")
        ref = rigcore.load_reference(pat, ft)
        if ref is None:
            sys.exit(f"no stored reference for {pat} f{ft} - use --ref FILE")
        print(f"reference: stored {pat} f{ft}")

    paths = a.images or pick_images()
    if not paths:
        return
    for p in paths:
        try:
            r = analyze_file(p, ref, model)
        except Exception as e:
            print(f"{os.path.basename(p)}: ERROR {e}")
            continue
        f, res = r["features"], r["result"]
        line = (f"{os.path.basename(p)}: lap_ratio={f['laplacian_ratio_mean']:.3f} "
                f"fft_ratio={f['fft_ratio_mean']:.3f} int_ratio={f['int_ratio']:.3f}")
        if res is not None:
            lod = model.d["lod_ntu"]
            ntu = f"<{lod:.2f}" if res["below_lod"] else f"{res['ntu']:.2f}"
            line += f"  ->  NTU {ntu}"
            if res["size_um"] is not None:
                line += f"  size~{res['size_um']:.2f}um"
            if res["flags"]:
                line += "  !! " + "; ".join(res["flags"])
        if r["resized"]:
            line += "  (resized to 480x360)"
        print(line)
    if model is None:
        print("(no results/model.json yet - showing features only)")

if __name__ == "__main__":
    main()
