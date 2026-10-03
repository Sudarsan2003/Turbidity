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
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

import rigcore

RANK_CSV = os.path.join(rigcore.RES, "pattern_ranking.csv")
BEST_JSON = os.path.join(rigcore.RES, "best_pattern.json")

def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")

def load_rows(path=rigcore.SWEEP_CSV):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            r["ntu_ref"] = _f(r.get("ntu_ref"))
            p = _f(r.get("particle_um"))
            r["particle_um"] = None if math.isnan(p) else p
            r["feat"] = int(_f(r.get("feat")))
            r["sat_frac"] = 0.0 if math.isnan(_f(r.get("sat_frac"))) else _f(r["sat_frac"])
            for n in rigcore.FEATURE_NAMES:
                r[n] = _f(r.get(n))
            rows.append(r)
    return rows

def _metrics(x, y, r2_thr=0.98):
    levels = np.unique(x)
    if len(levels) < 4 or levels.min() != 0:
        return None
    grp = [y[x == l] for l in levels]
    dof = sum(len(g) - 1 for g in grp if len(g) > 1)
    if dof < 1:
        return None
    sigma = math.sqrt(sum((len(g) - 1) * np.var(g, ddof=1) for g in grp if len(g) > 1) / dof)
    sigma = max(sigma, 1e-9)
    slope, icpt = np.polyfit(x, y, 1)
    if abs(slope) < 1e-12:
        return None
    means = np.array([g.mean() for g in grp])
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    kmax = 3
    for k in range(len(levels), 2, -1):
        if np.corrcoef(levels[:k], means[:k])[0, 1] ** 2 >= r2_thr:
            kmax = k
            break
    range_max = float(levels[kmax - 1])
    cvs = [g.std(ddof=1) / max(abs(g.mean()), 1e-9)
           for l, g in zip(levels, grp) if l > 0 and len(g) > 1]
    sens = abs(slope) / sigma
    return {"slope": float(slope), "sigma": float(sigma), "sensitivity": float(sens),
            "r2": r2, "range_max": range_max, "lod_ntu": float(3.3 * sigma / abs(slope)),
            "rep_cv": float(np.median(cvs)) if cvs else float("nan"),
            "score": float(sens * r2 * range_max / max(levels[-1], 1e-9)),
            "n_levels": int(len(levels)), "n_points": int(len(x))}

def rank(rows=None, max_sat=0.02):

    rows = load_rows() if rows is None else rows
    rows = [r for r in rows if r["sat_frac"] <= max_sat and not math.isnan(r["ntu_ref"])]
    blanks = defaultdict(list)
    samples = defaultdict(list)
    for r in rows:
        if r["ntu_ref"] == 0:
            blanks[r["fluid"]].append(r)
        else:
            samples[(r["fluid"], r["particle_um"])].append(r)
    out = []
    for (fluid, part), srows in samples.items():
        data = srows + blanks.get(fluid, [])
        cfgs = sorted({(r["pattern"], r["feat"]) for r in data})
        for pat, feat in cfgs:
            d = [r for r in data if r["pattern"] == pat and r["feat"] == feat]
            x = np.array([r["ntu_ref"] for r in d])
            best = None
            for name in rigcore.FEATURE_NAMES:
                y = np.array([r[name] for r in d])
                ok = ~np.isnan(y)
                if ok.sum() < 8:
                    continue
                m = _metrics(x[ok], y[ok])
                if m and (best is None or m["score"] > best["score"]):
                    best = dict(m, feature=name)
            if best:
                best.update({"fluid": fluid, "particle_um": part,
                             "pattern": pat, "feat": feat})
                out.append(best)
    out.sort(key=lambda d: d["score"], reverse=True)
    return out

def overall(results):

    agg = defaultdict(list)
    for r in results:
        agg[(r["fluid"], r["pattern"], r["feat"])].append(r["score"])
    rows = [{"fluid": k[0], "pattern": k[1], "feat": k[2],
             "mean_score": float(np.mean(v)), "n_groups": len(v)} for k, v in agg.items()]
    rows.sort(key=lambda d: d["mean_score"], reverse=True)
    return rows

def format_table(results, top=10):
    if not results:
        return "No rankable data yet (need >=4 NTU levels incl. 0, >=2 reps each)."
    lines = [f"{'#':>2} {'fluid':9s} {'part_um':>7} {'pattern':9s} {'feat':>4} {'feature':22s}"
             f"{'sens':>8} {'R2':>6} {'range':>7} {'LOD':>7} {'CV%':>6} {'score':>8}"]
    for i, r in enumerate(results[:top], 1):
        pu = "-" if r["particle_um"] is None else f"{r['particle_um']:g}"
        lines.append(f"{i:>2} {r['fluid'][:9]:9s} {pu:>7} {r['pattern']:9s} {r['feat']:>4} "
                     f"{r['feature']:22s}{r['sensitivity']:8.1f} {r['r2']:6.3f} "
                     f"{r['range_max']:7.1f} {r['lod_ntu']:7.2f} {r['rep_cv']*100:6.2f} "
                     f"{r['score']:8.1f}")
    return "\n".join(lines)

def save(results):
    if not results:
        return
    keys = ["fluid", "particle_um", "pattern", "feat", "feature", "slope", "sigma",
            "sensitivity", "r2", "range_max", "lod_ntu", "rep_cv", "score",
            "n_levels", "n_points"]
    with open(RANK_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank"] + keys)
        for i, r in enumerate(results, 1):
            w.writerow([i] + [r.get(k, "") for k in keys])
    with open(BEST_JSON, "w") as fh:
        json.dump({"best": results[0], "overall": overall(results)[:5]}, fh, indent=2)

def run(top=10):
    res = rank()
    save(res)
    txt = format_table(res, top)
    ov = overall(res)
    if ov:
        txt += "\n\nOverall (mean score over particle sizes):\n" + "\n".join(
            f"  {i}. {o['pattern']} f{o['feat']}  ({o['fluid']})  {o['mean_score']:.1f}"
            for i, o in enumerate(ov[:5], 1))
    return txt, res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    print(run(ap.parse_args().top)[0])
