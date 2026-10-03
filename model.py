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
import json
import math
import os

import numpy as np

import rank_patterns
import rigcore

MODEL_PATH = os.path.join(rigcore.RES, "model.json")

def _loo(A, y, pen):
    G = np.linalg.pinv(A.T @ A + np.diag(pen))
    beta = G @ A.T @ y
    h = np.clip(np.einsum("ij,jk,ik->i", A, G, A), 0, 0.999)
    loo = (y - A @ beta) / (1 - h)
    return float(np.sqrt(np.mean(loo ** 2))), beta

def _fit_target(X, y, names, max_feats=3, lam=1e-2):
    n, p = X.shape
    mu, sd = X.mean(0), X.std(0)
    keep = [j for j in range(p) if sd[j] > 1e-9]
    Xs = (X - mu) / np.where(sd > 1e-9, sd, 1.0)
    chosen, best = [], math.inf
    while len(chosen) < min(max_feats, n - 3):
        cand = None
        for j in keep:
            if j in chosen:
                continue
            idx = chosen + [j]
            A = np.hstack([Xs[:, idx], np.ones((n, 1))])
            e, _ = _loo(A, y, [lam] * len(idx) + [0])
            if cand is None or e < cand[0]:
                cand = (e, j)
        if cand is None or cand[0] >= best * 0.97:
            break
        best = cand[0]
        chosen.append(cand[1])
    if not chosen:
        raise RuntimeError("no usable feature (all constant?)")
    A = np.hstack([Xs[:, chosen], np.ones((n, 1))])
    e, beta = _loo(A, y, [lam] * len(chosen) + [0])
    return {"features": [names[j] for j in chosen],
            "mu": [float(mu[j]) for j in chosen], "sd": [float(sd[j]) for j in chosen],
            "coef": [float(b) for b in beta[:-1]], "intercept": float(beta[-1]),
            "loo_rmse": e}

def _apply(t, f):
    z = [(f[n] - m) / s for n, m, s in zip(t["features"], t["mu"], t["sd"])]
    return float(np.dot(z, t["coef"]) + t["intercept"])

class TurbidityModel:
    def __init__(self, d):
        self.d = d

    pattern = property(lambda s: s.d["pattern"])
    feat = property(lambda s: s.d["feat"])
    grid_cols = property(lambda s: s.d.get("grid_cols", rigcore.GRID_COLS))

    @classmethod
    def fit(cls, rows, pattern, feat, fluid=None, particle=None, max_feats=3):
        d = [r for r in rows if r["pattern"] == pattern and r["feat"] == feat
             and r["sat_frac"] <= 0.02 and not math.isnan(r["ntu_ref"])
             and (fluid is None or r["fluid"] == fluid)]
        if particle is not None:
            d = [r for r in d if r["ntu_ref"] == 0 or r["particle_um"] == particle]
        names = [n for n in rigcore.FEATURE_NAMES
                 if not any(math.isnan(r[n]) for r in d)]
        if len(d) < 8 or not names:
            raise RuntimeError(f"not enough sweep data for {pattern} f{feat} ({len(d)} rows)")
        X = np.array([[r[n] for n in names] for r in d])
        ntu = np.array([r["ntu_ref"] for r in d])
        tn = _fit_target(X, ntu, names, max_feats)

        ts = None
        if particle is None:
            ds = [i for i, r in enumerate(d) if r["particle_um"] and r["ntu_ref"] > 0]
            sizes = {d[i]["particle_um"] for i in ds}
            if len(sizes) >= 2 and len(ds) >= 8:
                ts = _fit_target(X[ds], np.log([d[i]["particle_um"] for i in ds]),
                                 names, max_feats)

        pred = np.array([_apply(tn, dict(zip(names, x))) for x in X])
        blank = pred[ntu == 0]
        lod = 3 * float(np.std(blank, ddof=1)) if len(blank) >= 3 else float("nan")
        model = {"pattern": pattern, "feat": int(feat), "fluid": fluid,
                 "grid_cols": int(d[0]["grid_cols"] or rigcore.GRID_COLS)
                 if str(d[0].get("grid_cols", "")).strip() else rigcore.GRID_COLS,
                 "ntu": tn, "size": ts, "lod_ntu": lod,
                 "ntu_max": float(ntu.max()), "n_rows": len(d),
                 "sizes_um": sorted({r["particle_um"] for r in d if r["particle_um"]}),
                 "train_range": {n: [float(X[:, i].min()), float(X[:, i].max())]
                                 for i, n in enumerate(names)}}
        m = cls(model)
        m.levels = sorted(set(ntu))
        m.table = [(l, float(pred[ntu == l].mean()), float(pred[ntu == l].std()))
                   for l in m.levels]
        return m

    def predict(self, f):
        d, flags = self.d, []
        ntu_raw = _apply(d["ntu"], f)
        for n in set(d["ntu"]["features"] + (d["size"]["features"] if d["size"] else [])):
            lo, hi = d["train_range"][n]
            span = max(hi - lo, 1e-9)
            if f[n] < lo - 0.1 * span or f[n] > hi + 0.1 * span:
                flags.append(f"{n} outside calibration")
        if ntu_raw > 1.1 * d["ntu_max"]:
            flags.append("above calibrated NTU range")
        lod = d["lod_ntu"]
        res = {"ntu_raw": ntu_raw, "ntu": max(ntu_raw, 0.0),
               "below_lod": (not math.isnan(lod)) and ntu_raw < lod,
               "size_um": None, "flags": flags}
        if d["size"] and ntu_raw > (0 if math.isnan(lod) else lod):
            res["size_um"] = float(math.exp(_apply(d["size"], f)))
        return res

    def save(self, path=MODEL_PATH):
        with open(path, "w") as fh:
            json.dump(self.d, fh, indent=2)
        return path

    @classmethod
    def load(cls, path=MODEL_PATH):
        with open(path) as fh:
            return cls(json.load(fh))

    def summary(self):
        d = self.d
        s = [f"pattern {d['pattern']} f{d['feat']}  fluid={d['fluid']}  rows={d['n_rows']}",
             f"NTU features : {', '.join(d['ntu']['features'])}",
             f"NTU LOO RMSE : {d['ntu']['loo_rmse']:.3f} NTU",
             f"detection limit (3 sigma on clear water): {d['lod_ntu']:.3f} NTU",
             f"calibrated up to {d['ntu_max']:g} NTU"]
        if d["size"]:
            e = d["size"]["loo_rmse"]
            s.append(f"size features: {', '.join(d['size']['features'])}  "
                     f"LOO error ~ x{math.exp(e):.2f} (multiplicative)")
        else:
            s.append("size model  : not fitted (needs >= 2 particle sizes)")
        s.append("NTU level -> mean model reading (in-sample):")
        s += [f"   {l:8g} -> {m:8.2f} +/- {sd:.2f}" for l, m, sd in self.table]
        return "\n".join(s)

def fit_from_sweep(pattern=None, feat=None, fluid=None, particle=None):
    rows = rank_patterns.load_rows()
    if not rows:
        raise RuntimeError("results/sweep.csv is empty - run pattern_sweep.py first")
    if pattern is None or feat is None:
        res = rank_patterns.rank(rows)
        if not res:
            raise RuntimeError("cannot pick a pattern yet - need more sweep data")
        pattern, feat = res[0]["pattern"], res[0]["feat"]
        fluid = fluid or res[0]["fluid"]
    m = TurbidityModel.fit(rows, pattern, int(feat), fluid, particle)
    m.save()
    return m

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern")
    ap.add_argument("--feat", type=int)
    ap.add_argument("--fluid")
    ap.add_argument("--particle", type=float)
    a = ap.parse_args()
    mdl = fit_from_sweep(a.pattern, a.feat, a.fluid, a.particle)
    print(mdl.summary())
    print(f"\nsaved {MODEL_PATH}")
