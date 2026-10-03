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
import csv
import os
import time

import rigcore
from hardware import add_hw_args, open_hardware, flush
from model import TurbidityModel

LOG = os.path.join(rigcore.RES, "runtime_log.csv")

def measure(cam, model, ref, n_avg):
    img, sat = rigcore.capture_average(cam, n_avg)
    f = rigcore.frame_features(img, ref)
    r = model.predict(f)
    if sat > 0.02:
        r["flags"].append(f"{sat*100:.1f}% clipped pixels")
    r["sat"] = sat
    return r

def fmt(r, lod):
    ntu = f"<{lod:.2f}" if r["below_lod"] else f"{r['ntu']:.2f}"
    size = "" if r["size_um"] is None else f"  size~{r['size_um']:.2f} um"
    flag = "" if not r["flags"] else "  !! " + "; ".join(r["flags"])
    return f"NTU {ntu}{size}{flag}"

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_hw_args(ap)
    ap.add_argument("--model", default=None)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--check-ref", action="store_true")
    a = ap.parse_args()

    model = TurbidityModel.load(a.model) if a.model else TurbidityModel.load()
    ref = rigcore.load_reference(model.pattern, model.feat, model.grid_cols)
    if ref is None:
        raise SystemExit("clear-water reference for the model's pattern is missing")
    lod = model.d["lod_ntu"]

    cam, proj = open_hardware(a)
    new = not os.path.exists(LOG)
    try:
        proj.show(model.pattern, model.feat)
        flush(cam)
        if a.check_ref:
            rs = [measure(cam, model, ref, a.n_avg) for _ in range(5)]
            vals = [r["ntu_raw"] for r in rs]
            mean = sum(vals) / len(vals)
            ok = abs(mean) <= max(lod, 1e-9)
            print(f"clear water reads {mean:.3f} NTU (LOD {lod:.3f}) -> "
                  f"{'OK' if ok else 'DRIFT: retake references / check lamp, cell, exposure'}")
            return
        with open(LOG, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["time", "ntu", "ntu_raw", "size_um", "below_lod", "flags"])
            while True:
                r = measure(cam, model, ref, a.n_avg)
                print(time.strftime("%H:%M:%S"), fmt(r, lod), flush=True)
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), f"{r['ntu']:.3f}",
                            f"{r['ntu_raw']:.3f}",
                            "" if r["size_um"] is None else f"{r['size_um']:.3f}",
                            int(r["below_lod"]), "|".join(r["flags"])])
                fh.flush()
                if a.once:
                    break
                time.sleep(a.interval)
    except KeyboardInterrupt:
        pass
    finally:
        proj.close()
        cam.release()

if __name__ == "__main__":
    main()
