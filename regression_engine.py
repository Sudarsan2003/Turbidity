import json
import math
import os

import numpy as np

import data_manager

MODEL_DIR = os.path.join(data_manager.DATA_DIR, "models")
NTU_MODEL_PATH = os.path.join(MODEL_DIR, "ntu_model.json")
SIZE_MODEL_PATH = os.path.join(MODEL_DIR, "size_model.json")

MIN_ROWS = 6          # refuse to train / predict with fewer rows than this
MIN_LEVELS = 3        # distinct target values needed
MAX_FEATS = 3         # forward selection stops here (few rows -> few features)
RANGE_MARGIN = 0.10   # allowed extrapolation beyond the training range, as a fraction of span
LAMBDA = 1e-2
MODEL_VERSION = 2

_FIXED_PATHS = {"known_NTU": NTU_MODEL_PATH, "particle_size": SIZE_MODEL_PATH}


def model_path(target):
    """One file per target. known_NTU / particle_size keep their old file names."""
    if target in _FIXED_PATHS:
        return _FIXED_PATHS[target]
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in target)
    return os.path.join(MODEL_DIR, f"{safe}_model.json")


# ----------------------------------------------------------------------------
# Random forest (small, dependency free), now serialisable
# ----------------------------------------------------------------------------
class SimpleDecisionTreeRegressor:

    def __init__(self, max_depth=4, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None

    def fit(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split or np.var(y) < 1e-7:
            self.value = float(np.mean(y))
            return

        best_feat, best_thresh = None, None
        best_loss = float("inf")
        for f in range(n_features):
            vals = np.unique(X[:, f])
            if len(vals) < 2:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0
            if len(thresholds) > 10:
                thresholds = np.quantile(vals, np.linspace(0.1, 0.9, 10))
            for t in thresholds:
                left_mask = X[:, f] <= t
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                y_l, y_r = y[left_mask], y[right_mask]
                loss = float(np.sum((y_l - y_l.mean()) ** 2) + np.sum((y_r - y_r.mean()) ** 2))
                if loss < best_loss:
                    best_loss, best_feat, best_thresh = loss, f, t

        if best_feat is None:
            self.value = float(np.mean(y))
            return

        self.feature_idx = int(best_feat)
        self.threshold = float(best_thresh)
        left_mask = X[:, best_feat] <= best_thresh
        self.left = SimpleDecisionTreeRegressor(self.max_depth, self.min_samples_split)
        self.left.fit(X[left_mask], y[left_mask], depth + 1)
        self.right = SimpleDecisionTreeRegressor(self.max_depth, self.min_samples_split)
        self.right.fit(X[~left_mask], y[~left_mask], depth + 1)

    def predict_one(self, x):
        if self.value is not None:
            return self.value
        if x[self.feature_idx] <= self.threshold:
            return self.left.predict_one(x)
        return self.right.predict_one(x)

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def to_dict(self):
        if self.value is not None:
            return {"value": self.value}
        return {"f": self.feature_idx, "t": self.threshold,
                "l": self.left.to_dict(), "r": self.right.to_dict()}

    @classmethod
    def from_dict(cls, d):
        t = cls()
        if "value" in d:
            t.value = float(d["value"])
        else:
            t.feature_idx, t.threshold = int(d["f"]), float(d["t"])
            t.left, t.right = cls.from_dict(d["l"]), cls.from_dict(d["r"])
        return t


class SimpleRandomForestRegressor:

    def __init__(self, n_estimators=15, max_depth=4, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        n_samples = len(X)
        self.trees = []
        rng = np.random.default_rng(42)
        for _ in range(self.n_estimators):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            tree = SimpleDecisionTreeRegressor(self.max_depth, self.min_samples_split)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

    def predict(self, X):
        return np.mean(np.array([t.predict(X) for t in self.trees]), axis=0)

    def to_list(self):
        return [t.to_dict() for t in self.trees]

    @classmethod
    def from_list(cls, trees):
        rf = cls(n_estimators=len(trees))
        rf.trees = [SimpleDecisionTreeRegressor.from_dict(t) for t in trees]
        return rf


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def prepare_data(records, target_name):
    """Rows with a valid target AND all ratio features finite.
    Rows with missing features (e.g. saved before the ratio columns existed) are
    skipped; they are NOT filled with zeros."""
    X_rows, y_rows, ids = [], [], []
    for r in records:
        try:
            y_val = float(str(r.get(target_name, "")).strip())
        except (ValueError, TypeError):
            continue
        if not math.isfinite(y_val):
            continue
        feats = []
        ok = True
        for col in data_manager.REGRESSION_FEATURE_COLUMNS:
            try:
                fv = float(str(r.get(col, "")).strip())
            except (ValueError, TypeError):
                ok = False
                break
            if not math.isfinite(fv):
                ok = False
                break
            feats.append(fv)
        if ok:
            X_rows.append(feats)
            y_rows.append(y_val)
            ids.append(r.get("experiment_id", ""))
    if not X_rows:
        return None, None, []
    return (np.array(X_rows, dtype=np.float64), np.array(y_rows, dtype=np.float64),
            list(data_manager.REGRESSION_FEATURE_COLUMNS))


# ----------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------
def _ridge_loo(A, y, pen):
    """Closed-form leave-one-out RMSE of a ridge fit (used for feature selection)."""
    G = np.linalg.pinv(A.T @ A + np.diag(pen))
    beta = G @ A.T @ y
    h = np.clip(np.einsum("ij,jk,ik->i", A, G, A), 0, 0.999)
    loo = (y - A @ beta) / (1 - h)
    return float(np.sqrt(np.mean(loo ** 2))), beta


def _select_features(Xs, y, max_feats):
    """Greedy forward selection by inner LOO error."""
    n, p = Xs.shape
    keep = [j for j in range(p) if np.ptp(Xs[:, j]) > 1e-9]
    chosen, best = [], math.inf
    while len(chosen) < min(max_feats, max(n - 3, 1)):
        cand = None
        for j in keep:
            if j in chosen:
                continue
            idx = chosen + [j]
            A = np.hstack([Xs[:, idx], np.ones((n, 1))])
            e, _ = _ridge_loo(A, y, [LAMBDA] * len(idx) + [0])
            if cand is None or e < cand[0]:
                cand = (e, j)
        if cand is None or (chosen and cand[0] >= best * 0.97):
            break
        best = cand[0]
        chosen.append(cand[1])
    return chosen


def _design(Z, model_type):
    n = len(Z)
    if model_type == "polynomial":
        return np.hstack([Z, Z ** 2, np.ones((n, 1))])
    return np.hstack([Z, np.ones((n, 1))])


def _fit(X, y, model_type, max_feats=MAX_FEATS):
    """Fit one model on (X, y). Returns a dict with everything needed to predict."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd
    idx = _select_features(Xs, y, max_feats)
    if not idx:
        raise ValueError("No usable feature: all features are constant across the saved rows.")
    Z = Xs[:, idx]
    fit = {"idx": idx, "mu": mu[idx].tolist(), "sd": sd[idx].tolist()}
    if model_type == "random_forest":
        rf = SimpleRandomForestRegressor(n_estimators=20, max_depth=4)
        rf.fit(Z, y)
        fit["trees"] = rf.to_list()
    else:
        A = _design(Z, model_type)
        k = A.shape[1]
        pen = np.full(k, LAMBDA)
        pen[-1] = 0.0
        G = np.linalg.pinv(A.T @ A + np.diag(pen))
        fit["coef"] = (G @ A.T @ y).tolist()
    return fit


def _predict_selected(fit, model_type, x_sel_raw):
    """Predict from RAW values of the selected features. x_sel_raw: (m, k)."""
    Z = (np.atleast_2d(x_sel_raw) - np.array(fit["mu"])) / np.array(fit["sd"])
    if "trees" in fit:
        return SimpleRandomForestRegressor.from_list(fit["trees"]).predict(Z)
    return _design(Z, model_type) @ np.array(fit["coef"])


def _predict_full(fit, model_type, X):
    return _predict_selected(fit, model_type, X[:, fit["idx"]])


def _metrics(y, pred):
    res = y - pred
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"rmse": float(np.sqrt(np.mean(res ** 2))),
            "mae": float(np.mean(np.abs(res))),
            "r2": float(1.0 - np.sum(res ** 2) / sst) if sst > 1e-12 else float("nan")}


def loo_predictions(X, y, model_type, max_feats=MAX_FEATS):
    """Leave-one-out predictions. Feature selection and fitting are redone without
    the held-out row, so nothing about that row leaks into its prediction."""
    n = len(y)
    out = np.empty(n)
    for i in range(n):
        tr = np.arange(n) != i
        fit = _fit(X[tr], y[tr], model_type, max_feats)
        out[i] = _predict_full(fit, model_type, X[i:i + 1])[0]
    return out


def train_regression_model(records, target="known_NTU", model_type="ridge", deg=1):
    """Train and save a model for `target`. Returned metrics (r2, rmse, mae) are
    LEAVE-ONE-OUT; the in-sample versions are train_r2 / train_rmse / train_mae."""
    if model_type == "polynomial" and deg != 2:
        model_type = "ridge"
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y, names = prepare_data(records, target)
    n = 0 if X is None else len(X)
    if n < MIN_ROWS:
        raise ValueError(
            f"Need at least {MIN_ROWS} saved experiments with a '{target}' value and "
            f"ratio features to train (found {n}). Rows saved before the ratio columns "
            f"existed cannot be used - recapture them.")
    n_levels = len(np.unique(y))
    if n_levels < MIN_LEVELS:
        raise ValueError(f"Need at least {MIN_LEVELS} different '{target}' values "
                         f"(found {n_levels}).")

    fit = _fit(X, y, model_type)
    pred_train = _predict_full(fit, model_type, X)
    pred_loo = loo_predictions(X, y, model_type)
    loo, tr = _metrics(y, pred_loo), _metrics(y, pred_train)

    sel = [names[j] for j in fit["idx"]]
    span = float(np.ptp(y))
    meta = {
        "version": MODEL_VERSION,
        "target": target,
        "model_type": model_type,
        "n_samples": int(n),
        "n_levels": int(n_levels),
        "feature_names": names,
        "selected": sel,
        "fit": fit,
        "train_range": {names[j]: [float(X[:, j].min()), float(X[:, j].max())]
                        for j in fit["idx"]},
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "validation": "leave-one-out (feature selection repeated in each fold)",
        # headline numbers = out-of-sample
        "rmse": loo["rmse"], "mae": loo["mae"], "r2": loo["r2"],
        "rmse_pct_range": loo["rmse"] / span * 100.0 if span > 0 else float("nan"),
        # in-sample, for comparison only
        "train_rmse": tr["rmse"], "train_mae": tr["mae"], "train_r2": tr["r2"],
        "actual": y.tolist(),
        "predicted": pred_loo.tolist(),          # LOO predictions (plotted)
        "predicted_train": pred_train.tolist(),
    }
    with open(model_path(target), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_model(target="known_NTU"):
    path = model_path(target)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# ----------------------------------------------------------------------------
# Prediction (with guards)
# ----------------------------------------------------------------------------
def predict_target(model, feat_dict):
    """Returns {"value": float|None, "status": str, ...}.
    value is None whenever the number would not be trustworthy; status says why."""
    if model is None:
        return {"value": None, "status": "No model trained"}
    if model.get("version", 1) < MODEL_VERSION or "fit" not in model:
        return {"value": None,
                "status": "Model file is from an older version - retrain it"}
    if model.get("n_samples", 0) < MIN_ROWS:
        return {"value": None,
                "status": f"Model trained on only {model.get('n_samples', 0)} rows "
                          f"(need {MIN_ROWS})"}
    if not feat_dict:
        return {"value": None, "status": "No features (load a reference and a sample)"}

    sel = model["selected"]
    x, problems = [], []
    for name in sel:
        v = feat_dict.get(name)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = float("nan")
        if not math.isfinite(v):
            problems.append(f"missing {name}")
        x.append(v)
    if problems:
        return {"value": None, "status": "Invalid features: " + ", ".join(problems)}

    outside = []
    for name, v in zip(sel, x):
        lo, hi = model["train_range"][name]
        span = max(hi - lo, 1e-9)
        if v < lo - RANGE_MARGIN * span or v > hi + RANGE_MARGIN * span:
            outside.append(name)
    if outside:
        return {"value": None, "outside": outside,
                "status": "Features outside calibrated range: " + ", ".join(outside)}

    raw = float(_predict_selected(model["fit"], model["model_type"], np.array([x]))[0])
    flags = []
    y_span = max(model["y_max"] - model["y_min"], 1e-9)
    if raw > model["y_max"] + RANGE_MARGIN * y_span:
        flags.append("Above highest calibration level")
    if raw < model["y_min"] - RANGE_MARGIN * y_span:
        flags.append("Below lowest calibration level")
    return {
        "value": max(0.0, raw),
        "raw": raw,
        "target": model["target"],
        "r2": model.get("r2", float("nan")),
        "rmse": model.get("rmse", float("nan")),   # leave-one-out
        "flags": flags,
        "status": "OK",
    }


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def render_regression_plot(model, width=440, height=280):
    """Actual vs leave-one-out predicted."""
    import cv2
    img = np.full((height, width, 3), (24, 28, 34), dtype=np.uint8)
    if model is None or "actual" not in model or "predicted" not in model:
        cv2.putText(img, "No Model Trained Yet", (width // 4, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 150, 160), 1)
        return img

    y_act = np.array(model["actual"])
    y_pred = np.array(model["predicted"])
    if len(y_act) == 0:
        return img

    lo = min(y_act.min(), y_pred.min(), 0.0)
    hi = max(y_act.max(), y_pred.max(), 1e-6)
    span = max(hi - lo, 1e-6)

    ml, mr, mt, mb = 50, 20, 35, 45
    pw, ph = width - ml - mr, height - mt - mb
    cv2.rectangle(img, (ml, mt), (width - mr, height - mb), (50, 60, 72), 1)
    cv2.line(img, (ml, height - mb), (width - mr, mt), (80, 100, 120), 1, cv2.LINE_AA)

    def px(a, p):
        return (int(ml + (a - lo) / span * pw), int(height - mb - (p - lo) / span * ph))

    for a, p in zip(y_act, y_pred):
        sx, sy = px(a, p)
        cv2.circle(img, (sx, sy), 5, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (sx, sy), 5, (255, 255, 255), 1, cv2.LINE_AA)

    grey = (140, 150, 160)
    cv2.putText(img, f"{lo:.3g}", (ml - 4, height - mb + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.33, grey, 1)
    cv2.putText(img, f"{hi:.3g}", (width - mr - 24, height - mb + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.33, grey, 1)
    cv2.putText(img, f"{hi:.3g}", (4, mt + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.33, grey, 1)
    cv2.putText(img, f"{lo:.3g}", (4, height - mb), cv2.FONT_HERSHEY_SIMPLEX, 0.33, grey, 1)

    t_name = model.get("target", "Target")
    cv2.putText(img, f"{t_name}: actual vs leave-one-out prediction", (ml, mt - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(img, f"LOO R2={model.get('r2', float('nan')):.3f}  RMSE={model.get('rmse', 0.0):.3g}",
                (ml, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (62, 214, 198), 1, cv2.LINE_AA)
    cv2.putText(img, f"n={model.get('n_samples', len(y_act))}", (width - 60, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, grey, 1, cv2.LINE_AA)
    return img