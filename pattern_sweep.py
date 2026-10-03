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
import time

import cv2

import rigcore
from hardware import add_hw_args, open_hardware, flush

DEF_PATTERNS = "checker,vstripe,hstripe,dots,circles,random"
DEF_FEATS = "12,24,48"

def capture_references(cam, proj, patterns, feats, n_avg, log=print):
    for p in patterns:
        for f in feats:
            proj.show(p, f)
            flush(cam)
            img, sat = rigcore.capture_average(cam, n_avg)
            rigcore.save_reference_image(img, p, f)
            log(f"reference {p:8s} f{f:<3d} mean={img.mean():6.1f} clipped={sat*100:.2f}%")
            if sat > 0.02:
                log("   WARNING: >2% clipped pixels. Lower exposure/gain and redo the references.")

def run_sample(cam, proj, patterns, feats, meta, reps, n_avg, log=print):
    stamp = rigcore.timestamp()
    for p in patterns:
        for f in feats:
            ref = rigcore.load_reference(p, f)
            if ref is None:
                log(f"skip {p} f{f}: no clear-water reference (run the 'reference' step)")
                continue
            proj.show(p, f)
            flush(cam)
            for rep in range(reps):
                img, sat = rigcore.capture_average(cam, n_avg)
                ft = rigcore.frame_features(img, ref)
                name = ""
                if rep == 0:
                    name = f"sweep_{stamp}_{meta['label']}_{p}_f{f}.png"
                    ok, buf = cv2.imencode(".png", img)
                    if ok:
                        buf.tofile(f"{rigcore.CAP_DIR}/{name}")
                row = dict(meta)
                row.update({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "pattern": p, "feat": f, "grid_cols": ref["grid_cols"],
                            "rep": rep, "sat_frac": float(sat), "image": name})
                row.update(ft)
                rigcore.append_sweep_row(row)
            log(f"{meta['label']:>8s} {p:8s} f{f:<3d} x{reps}  "
                f"lap_ratio={ft['laplacian_ratio_mean']:.3f}  int_ratio={ft['int_ratio']:.3f}")

def _lists(a):
    pats = [s.strip() for s in a.patterns.split(",") if s.strip()]
    feats = [int(s) for s in a.feats.split(",") if s.strip()]
    return pats, feats

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference", help="capture clear-water references")
    s = sub.add_parser("sample", help="capture one standard/sample")
    for p in (r, s):
        add_hw_args(p)
        p.add_argument("--patterns", default=DEF_PATTERNS)
        p.add_argument("--feats", default=DEF_FEATS)
    s.add_argument("--fluid", default="formazin")
    s.add_argument("--label", required=True)
    s.add_argument("--ntu", type=float, required=True, help="known NTU (0 for clear water)")
    s.add_argument("--particle", type=float, default=None, help="particle size in um")
    s.add_argument("--blank", action="store_true",
                   help="freshly refilled CLEAR water used to measure the noise floor")
    s.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()

    pats, feats = _lists(a)
    cam, proj = open_hardware(a)
    try:
        if a.cmd == "reference":
            capture_references(cam, proj, pats, feats, a.n_avg)
        else:
            meta = {"fluid": a.fluid, "label": a.label, "ntu_ref": float(a.ntu),
                    "particle_um": float(a.particle) if a.particle else "",
                    "is_blank": int(a.blank or a.ntu == 0)}
            run_sample(cam, proj, pats, feats, meta, a.reps, a.n_avg)
        print("done.")
    finally:
        proj.close()
        cam.release()

if __name__ == "__main__":
    main()
