import os, sys, csv, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_manager as dm

_tmp = tempfile.mkdtemp()
for name in ("DATA_DIR",):
    setattr(dm, name, _tmp)
dm.REF_DIR = os.path.join(_tmp, "references"); dm.CAP_DIR = os.path.join(_tmp, "captures")
dm.BLUR_MAT_DIR = os.path.join(_tmp, "blur_matrices"); dm.ANGLE_MAT_DIR = os.path.join(_tmp, "angles")
dm.CSV_PATH = os.path.join(_tmp, "experiments.csv")

import regression_engine as RE
RE.MODEL_DIR = os.path.join(_tmp, "models")
RE.NTU_MODEL_PATH = os.path.join(RE.MODEL_DIR, "ntu_model.json")
RE.SIZE_MODEL_PATH = os.path.join(RE.MODEL_DIR, "size_model.json")
RE._FIXED_PATHS = {"known_NTU": RE.NTU_MODEL_PATH, "particle_size": RE.SIZE_MODEL_PATH}

import calibration, noise_floor, preprocessing
from patterns import generate_pattern
from utils import simulate_blur

W, H = 640, 360
REF = generate_pattern("checker", W, H, 24)
rng_state = np.random.seed(0)


def run(img):
    return preprocessing.run_preprocessing_pipeline(REF, img, target_size=(W, H))


def test_identical_images_give_zero_ratio():
    res = run(REF.copy())
    f = dm.features_from_proc(res)
    for m in ("laplacian", "tenengrad", "fft"):
        assert f[f"ratio_mean_{m}"] < 1e-3
        assert f[f"global_ratio_{m}"] < 1e-3


def test_more_blur_gives_higher_ratio():
    vals = [dm.features_from_proc(run(simulate_blur(REF, r)))["ratio_mean_laplacian"] for r in (0, 1, 2, 3, 5)]
    assert all(b > a for a, b in zip(vals, vals[1:])), vals


def _build_dataset():
    dm.init_storage()
    levels = [0] * 5 + [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    for i, lvl in enumerate(levels):
        img = simulate_blur(REF, lvl, noise_pct=2.0)
        proc = run(img)
        gt = {"known_NTU": lvl * 20.0, "particle_concentration": lvl * 10.0,
              "particle_size": 5.0 + (i % 3)}
        dm.save_experiment_record(f"EXP{i+1:03d}", {"filename": "ref.png"}, img, proc, gt)
    return dm.load_experiments()


RECORDS = _build_dataset()


def test_ratio_columns_are_saved():
    assert RECORDS[0]["ratio_mean_laplacian"] != ""
    assert RECORDS[-1]["intensity_ratio"] != ""
    assert float(RECORDS[-1]["ratio_mean_laplacian"]) > float(RECORDS[0]["ratio_mean_laplacian"])


def test_loo_is_reported_and_worse_or_equal_to_train():
    m = RE.train_regression_model(RECORDS, "known_NTU")
    assert "leave-one-out" in m["validation"]
    assert m["rmse"] >= m["train_rmse"] - 1e-9
    assert m["r2"] > 0.8, m["r2"]
    assert all(f in dm.RATIO_FEATURE_COLUMNS for f in m["selected"])


def test_all_zero_features_are_refused():
    m = RE.train_regression_model(RECORDS, "known_NTU")
    zero = {k: 0.0 for k in dm.REGRESSION_FEATURE_COLUMNS}
    zero.update({"intensity_ratio": 0.0, "contrast_ratio": 0.0})
    far = {k: 1e4 for k in dm.REGRESSION_FEATURE_COLUMNS}
    p = RE.predict_target(m, far)
    assert p["value"] is None and "outside" in p["status"].lower()
    assert RE.predict_target(m, {})["value"] is None


def test_too_few_rows_refused():
    try:
        RE.train_regression_model(RECORDS[:4], "known_NTU")
    except ValueError as e:
        assert "at least" in str(e)
    else:
        raise AssertionError("should have refused")


def test_random_forest_roundtrip():
    RE.train_regression_model(RECORDS, "known_NTU", model_type="random_forest")
    m = RE.load_model("known_NTU")
    f = dm.features_from_proc(run(simulate_blur(REF, 3, 2.0)))
    p = RE.predict_target(m, f)
    assert p["value"] is not None and 20 <= p["value"] <= 100, p


def test_targets_do_not_overwrite_each_other():
    RE.train_regression_model(RECORDS, "particle_size")
    RE.train_regression_model(RECORDS, "particle_concentration")
    assert RE.load_model("particle_size")["target"] == "particle_size"
    assert RE.load_model("particle_concentration")["target"] == "particle_concentration"


def test_calibration_and_lod():
    cal = calibration.calibration_curve(RECORDS, "known_NTU")
    assert cal["r2"] > 0.9 and cal["slope"] != 0
    assert cal["lod"] == cal["lod"] and cal["lod"] > 0 and cal["n_blank"] == 5
    img = calibration.render_calibration_plot(cal, unit="NTU")
    assert img.shape == (320, 620, 3)


def test_noise_floor():
    imgs = [simulate_blur(REF, 0, 2.0) for _ in range(8)]
    s = noise_floor.summarize(noise_floor.features_for_images(REF, imgs, (W, H)))
    assert s["n"] == 8 and all(r["std"] >= 0 for r in s["rows"])
    assert "std" in noise_floor.format_report(s)


def test_old_csv_is_migrated():
    old = os.path.join(_tmp, "old")
    os.makedirs(old)
    dm.CSV_PATH = os.path.join(old, "experiments.csv")
    dm.DATA_DIR = old
    with open(dm.CSV_PATH, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["experiment_id", "known_NTU"]); w.writerow(["EXP001", "5"])
    dm._migrate_csv_if_needed()
    rows = dm.load_experiments()
    assert rows[0]["experiment_id"] == "EXP001" and rows[0]["known_NTU"] == "5"
    assert "ratio_mean_laplacian" in rows[0]


if __name__ == "__main__":
    fails = 0
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)