\
\
\
\
\
\
\
\
\

import json
import math
import os
import cv2
import numpy as np

import data_manager

MODEL_DIR = os.path.join(data_manager.DATA_DIR, "models")
NTU_MODEL_PATH = os.path.join(MODEL_DIR, "ntu_model.json")
SIZE_MODEL_PATH = os.path.join(MODEL_DIR, "size_model.json")

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
                    best_loss = loss
                    best_feat = f
                    best_thresh = t

        if best_feat is None:
            self.value = float(np.mean(y))
            return

        self.feature_idx = best_feat
        self.threshold = best_thresh
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
            tree = SimpleDecisionTreeRegressor(max_depth=self.max_depth,
                                               min_samples_split=self.min_samples_split)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

    def predict(self, X):
        preds = np.array([t.predict(X) for t in self.trees])
        return np.mean(preds, axis=0)

def prepare_data(records, target_name):
\
\
\

    rows = []
    for r in records:
        t_val = r.get(target_name, "").strip()
        try:
            y_val = float(t_val)
            if np.isnan(y_val):
                continue
        except (ValueError, TypeError):
            continue

        feat_vals = []
        valid_features = True
        for col in data_manager.REGRESSION_FEATURE_COLUMNS:
            f_str = r.get(col, "").strip()
            try:
                fv = float(f_str)
                if np.isnan(fv):
                    fv = 0.0
                feat_vals.append(fv)
            except (ValueError, TypeError):
                valid_features = False
                break

        if valid_features:
            rows.append((feat_vals, y_val))

    if not rows:
        return None, None, []

    X = np.array([r[0] for r in rows], dtype=np.float64)
    y = np.array([r[1] for r in rows], dtype=np.float64)
    return X, y, data_manager.REGRESSION_FEATURE_COLUMNS

def train_regression_model(records, target="known_NTU", model_type="ridge", deg=1):
\
\
\
\
\
\
\

    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y, feat_names = prepare_data(records, target)
    if X is None or len(X) < 3:
        raise ValueError(f"Need at least 3 experiments with ground-truth '{target}' to train. Found {0 if X is None else len(X)}.")

    n, p = X.shape
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    X_norm = (X - mu) / sd

    model_meta = {
        "target": target,
        "model_type": model_type,
        "n_samples": n,
        "feature_names": feat_names,
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "deg": deg,
    }

    if model_type == "polynomial" and deg == 2:

        X_poly = np.hstack([X_norm, X_norm ** 2])
        A = np.hstack([X_poly, np.ones((n, 1))])
        lam = 1e-2
        G = np.linalg.pinv(A.T @ A + lam * np.eye(A.shape[1]))
        beta = G @ A.T @ y
        y_pred = A @ beta
        model_meta["coef"] = beta.tolist()
    elif model_type == "random_forest":
        rf = SimpleRandomForestRegressor(n_estimators=20, max_depth=4)
        rf.fit(X_norm, y)
        y_pred = rf.predict(X_norm)

        model_meta["rf_model"] = True
    else:
        A = np.hstack([X_norm, np.ones((n, 1))])
        lam = 1e-2
        G = np.linalg.pinv(A.T @ A + lam * np.eye(A.shape[1]))
        beta = G @ A.T @ y
        y_pred = A @ beta
        model_meta["coef"] = beta.tolist()

    res = y - y_pred
    rmse = float(np.sqrt(np.mean(res ** 2)))
    mae = float(np.mean(np.abs(res)))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum(res ** 2) / max(sst, 1e-9)) if sst > 1e-9 else 1.0

    model_meta["rmse"] = rmse
    model_meta["mae"] = mae
    model_meta["r2"] = r2
    model_meta["actual"] = y.tolist()
    model_meta["predicted"] = y_pred.tolist()

    save_path = NTU_MODEL_PATH if target == "known_NTU" else SIZE_MODEL_PATH
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2)

    return model_meta

def load_model(target="known_NTU"):

    save_path = NTU_MODEL_PATH if target == "known_NTU" else SIZE_MODEL_PATH
    if os.path.exists(save_path):
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def predict_target(model, feat_dict):
\
\

    if model is None:
        return {"value": None, "status": "No model loaded"}

    feat_names = model["feature_names"]
    mu = np.array(model["mu"])
    sd = np.array(model["sd"])

    x_raw = []
    for f in feat_names:
        v = feat_dict.get(f, 0.0)
        try:
            x_raw.append(float(v))
        except (ValueError, TypeError):
            x_raw.append(0.0)

    x = np.array(x_raw)
    x_norm = (x - mu) / sd

    deg = model.get("deg", 1)
    coef = np.array(model.get("coef", []))

    if len(coef) == 0:
        return {"value": None, "status": "Invalid model coefficients"}

    if model.get("model_type") == "polynomial" and deg == 2:
        x_poly = np.concatenate([x_norm, x_norm ** 2, [1.0]])
        pred = float(np.dot(x_poly, coef))
    else:
        x_lin = np.concatenate([x_norm, [1.0]])
        pred = float(np.dot(x_lin, coef))

    pred = max(0.0, pred)
    flags = []
    if pred < model.get("y_min", 0.0) or pred > model.get("y_max", 1e9):
        flags.append("Outside calibration range")

    return {
        "value": pred,
        "target": model["target"],
        "r2": model.get("r2", 0.0),
        "rmse": model.get("rmse", 0.0),
        "flags": flags,
        "status": "OK",
    }

def render_regression_plot(model, width=440, height=280):
\
\
\

    img = np.full((height, width, 3), (24, 28, 34), dtype=np.uint8)
    if model is None or "actual" not in model or "predicted" not in model:
        cv2.putText(img, "No Model Trained Yet", (width // 4, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 150, 160), 1)
        return img

    y_act = np.array(model["actual"])
    y_pred = np.array(model["predicted"])

    if len(y_act) == 0:
        return img

    min_val = min(y_act.min(), y_pred.min(), 0.0)
    max_val = max(y_act.max(), y_pred.max(), 1.0)
    val_range = max(max_val - min_val, 1e-6)

    margin_l, margin_r = 50, 20
    margin_t, margin_b = 35, 45
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    cv2.rectangle(img, (margin_l, margin_t), (width - margin_r, height - margin_b),
                  (50, 60, 72), 1)

    p1 = (margin_l, height - margin_b)
    p2 = (width - margin_r, margin_t)
    cv2.line(img, p1, p2, (80, 100, 120), 1, cv2.LINE_AA)

    def to_screen(act, pred):
        x = int(margin_l + (act - min_val) / val_range * plot_w)
        y = int(height - margin_b - (pred - min_val) / val_range * plot_h)
        return x, y

    for a, p in zip(y_act, y_pred):
        sx, sy = to_screen(a, p)
        cv2.circle(img, (sx, sy), 5, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (sx, sy), 5, (255, 255, 255), 1, cv2.LINE_AA)

    t_name = model.get("target", "Target")
    r2_str = f"R2 = {model.get('r2', 0.0):.3f}"
    rmse_str = f"RMSE = {model.get('rmse', 0.0):.2f}"

    cv2.putText(img, f"{t_name} Actual vs Predicted", (margin_l, margin_t - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(img, f"{r2_str}  |  {rmse_str}", (width - 170, margin_t - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (62, 214, 198), 1, cv2.LINE_AA)
    cv2.putText(img, "Actual Ground Truth ->", (margin_l + plot_w // 4, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 150, 160), 1, cv2.LINE_AA)

    return img
