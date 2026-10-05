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

import csv
import datetime
import os
import re
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# All data lives in ONE folder. Default: "data" next to this file. To keep a single dataset no matter
# which copy of the code you run, set the environment variable TURBIDITY_DATA_DIR to a fixed path.
DATA_DIR = os.environ.get("TURBIDITY_DATA_DIR") or os.path.join(BASE_DIR, "data")
REF_DIR = os.path.join(DATA_DIR, "references")
CAP_DIR = os.path.join(DATA_DIR, "captures")
BLUR_MAT_DIR = os.path.join(DATA_DIR, "blur_matrices")
ANGLE_MAT_DIR = os.path.join(DATA_DIR, "scattering_angles")
CSV_PATH = os.path.join(DATA_DIR, "experiments.csv")
REGRESSION_CSV_PATH = os.path.join(DATA_DIR, "regression_dataset.csv")

_METHODS = ("laplacian", "tenengrad", "fft")

# Ratio features (sample vs clear-water reference). These are lighting-independent
# and are what the regression uses. Raw scores are still saved for reference only.
RATIO_FEATURE_COLUMNS = (
    [f"{k}_{m}" for m in _METHODS
     for k in ("global_ratio", "ratio_mean", "ratio_std", "ratio_p90", "angle_mean")]
    + ["intensity_ratio", "contrast_ratio"]
)
QC_COLUMNS = ["clipped_pct"]

EXPERIMENT_COLUMNS = [
    "experiment_id",
    "timestamp",
    "reference_image_name",
    "captured_image_name",
    "fluid_type",
    "particle_material",
    "particle_size",
    "particle_size_unit",
    "particle_concentration",
    "concentration_unit",
    "known_NTU",
    "sample_id",
    "algorithm",
    "image_width",
    "image_height",
    "kernel_rows",
    "kernel_columns",
    "laplacian_variance",
    "tenengrad",
    "fft_high_frequency_energy",
    "mean_blur",
    "min_blur",
    "max_blur",
    "median_blur",
    "std_blur",
    "mean_scattering_angle",
    "min_scattering_angle",
    "max_scattering_angle",
    "median_scattering_angle",
    "std_scattering_angle",
    "diff_mean",
    "diff_std",
    "shift_dx",
    "shift_dy",
    "ref_laplacian_variance",
    "ref_tenengrad",
    "ref_fft_energy",
] + RATIO_FEATURE_COLUMNS + QC_COLUMNS + [
    "blur_matrix_file",
    "scattering_angles_file",
    "status",
]

REGRESSION_FEATURE_COLUMNS = list(RATIO_FEATURE_COLUMNS)

REGRESSION_TARGET_COLUMNS = [
    "known_NTU",
    "particle_size",
    "particle_concentration",
]

def _write_png(path, img):
    """Lossless PNG write that raises on failure (cv2.imwrite fails silently)."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise IOError(f"Could not encode PNG: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf.tofile(path)

def _migrate_csv_if_needed():
    """If experiments.csv was written with an older column set, rewrite it with the
    current columns (new columns left blank) and keep a backup of the original."""
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), [])
    if header == EXPERIMENT_COLUMNS:
        return
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(DATA_DIR, f"experiments_backup_{stamp}.csv")
    os.replace(CSV_PATH, backup)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EXPERIMENT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

def init_storage():

    for d in (DATA_DIR, REF_DIR, CAP_DIR, BLUR_MAT_DIR, ANGLE_MAT_DIR):
        os.makedirs(d, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(EXPERIMENT_COLUMNS)
    else:
        _migrate_csv_if_needed()

def features_from_proc(proc_res):
    """Ratio features for one processed frame, keyed by REGRESSION_FEATURE_COLUMNS.
    Single source of truth for the CSV, the regression training and live prediction.
    Returns None if the pipeline result has no ratio metrics."""
    if not proc_res or "metrics" not in proc_res:
        return None
    m = proc_res["metrics"]
    out = {}
    for meth in _METHODS:
        d = m.get(meth)
        if d is None:
            return None
        for k in ("global_ratio", "ratio_mean", "ratio_std", "ratio_p90", "angle_mean"):
            out[f"{k}_{meth}"] = float(d[k])
    out["intensity_ratio"] = float(proc_res.get("intensity_ratio", float("nan")))
    out["contrast_ratio"] = float(proc_res.get("contrast_ratio", float("nan")))
    return out

def get_next_experiment_id():

    init_storage()
    existing_ids = []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                eid = r.get("experiment_id", "")
                m = re.match(r"EXP(\d+)", eid, re.IGNORECASE)
                if m:
                    existing_ids.append(int(m.group(1)))

    if os.path.exists(CAP_DIR):
        for fname in os.listdir(CAP_DIR):
            m = re.match(r"EXP(\d+)\.png", fname, re.IGNORECASE)
            if m:
                existing_ids.append(int(m.group(1)))

    next_num = max(existing_ids, default=0) + 1
    return f"EXP{next_num:03d}"

def save_reference_image(img, original_filename="reference.png"):
\
\
\

    init_storage()
    base, ext = os.path.splitext(os.path.basename(original_filename))
    if not ext:
        ext = ".png"
    safe_base = re.sub(r"[^\w\-.]+", "_", base).strip("._") or "reference"
    target_name = f"{safe_base}{ext}"
    target_path = os.path.join(REF_DIR, target_name)

    counter = 1
    while os.path.exists(target_path):

        existing = cv2.imread(target_path)
        if existing is not None and existing.shape == img.shape and np.array_equal(existing, img):
            break
        target_name = f"{safe_base}_{counter:03d}{ext}"
        target_path = os.path.join(REF_DIR, target_name)
        counter += 1

    if not os.path.exists(target_path):
        _write_png(target_path, img)

    h, w = img.shape[:2]
    fmt = ext.replace(".", "").upper()
    return {
        "filename": target_name,
        "path": target_path,
        "width": w,
        "height": h,
        "format": fmt,
    }

def save_experiment_record(exp_id, ref_meta, cap_img, proc_res, gt):
\
\
\
\
\
\

    init_storage()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cap_fname = f"{exp_id}.png"
    cap_path = os.path.join(CAP_DIR, cap_fname)
    _write_png(cap_path, cap_img)

    blur_fname = f"{exp_id}_blur.csv"
    blur_path = os.path.join(BLUR_MAT_DIR, blur_fname)
    np.savetxt(blur_path, proc_res["blur_matrix"], delimiter=",", fmt="%.6f")

    angle_fname = f"{exp_id}_angles.csv"
    angle_path = os.path.join(ANGLE_MAT_DIR, angle_fname)
    np.savetxt(angle_path, proc_res["angle_matrix"], delimiter=",", fmt="%.4f")

    h, w = cap_img.shape[:2]
    all_scores = proc_res.get("all_blur_scores", {})
    ref_scores = proc_res.get("ref_blur_scores", {})
    b_stats = proc_res.get("blur_stats", {})
    a_stats = proc_res.get("angle_stats", {})

    def clean_num(v):
        if v is None:
            return ""
        s = str(v).strip()
        if not s:
            return ""
        try:
            val = float(s)
            return f"{val:.6g}"
        except ValueError:
            return ""

    row = {
        "experiment_id": exp_id,
        "timestamp": timestamp,
        "reference_image_name": ref_meta.get("filename", "") if ref_meta else "",
        "captured_image_name": cap_fname,
        "fluid_type": gt.get("fluid_type", "Water"),
        "particle_material": gt.get("particle_material", "Silica"),
        "particle_size": clean_num(gt.get("particle_size")),
        "particle_size_unit": gt.get("particle_size_unit", "µm"),
        "particle_concentration": clean_num(gt.get("particle_concentration")),
        "concentration_unit": gt.get("concentration_unit", "mg/L"),
        "known_NTU": clean_num(gt.get("known_NTU")),
        "sample_id": gt.get("sample_id", ""),
        "algorithm": gt.get("algorithm", "Variance of Laplacian"),
        "image_width": w,
        "image_height": h,
        "kernel_rows": proc_res.get("grid_rows", 12),
        "kernel_columns": proc_res.get("grid_cols", 12),
        "laplacian_variance": f"{all_scores.get('laplacian', 0.0):.4f}",
        "tenengrad": f"{all_scores.get('tenengrad', 0.0):.4f}",
        "fft_high_frequency_energy": f"{all_scores.get('fft', 0.0):.4f}",
        "mean_blur": f"{b_stats.get('mean', 0.0):.4f}",
        "min_blur": f"{b_stats.get('min', 0.0):.4f}",
        "max_blur": f"{b_stats.get('max', 0.0):.4f}",
        "median_blur": f"{b_stats.get('median', 0.0):.4f}",
        "std_blur": f"{b_stats.get('std', 0.0):.4f}",
        "mean_scattering_angle": f"{a_stats.get('mean', 0.0):.4f}",
        "min_scattering_angle": f"{a_stats.get('min', 0.0):.4f}",
        "max_scattering_angle": f"{a_stats.get('max', 0.0):.4f}",
        "median_scattering_angle": f"{a_stats.get('median', 0.0):.4f}",
        "std_scattering_angle": f"{a_stats.get('std', 0.0):.4f}",
        "diff_mean": f"{proc_res.get('diff_mean', 0.0):.4f}",
        "diff_std": f"{proc_res.get('diff_std', 0.0):.4f}",
        "shift_dx": f"{proc_res.get('shift_dx', 0.0):.2f}",
        "shift_dy": f"{proc_res.get('shift_dy', 0.0):.2f}",
        "ref_laplacian_variance": f"{ref_scores.get('laplacian', 0.0):.4f}",
        "ref_tenengrad": f"{ref_scores.get('tenengrad', 0.0):.4f}",
        "ref_fft_energy": f"{ref_scores.get('fft', 0.0):.4f}",
        "blur_matrix_file": blur_fname,
        "scattering_angles_file": angle_fname,
        "status": "Saved",
    }
    ratio_feats = features_from_proc(proc_res) or {}
    for col in RATIO_FEATURE_COLUMNS:
        v = ratio_feats.get(col)
        row[col] = "" if v is None or not np.isfinite(v) else f"{v:.6g}"
    row["clipped_pct"] = f"{proc_res.get('clipped_pct', 0.0):.4f}"

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_COLUMNS)
        writer.writerow(row)

    return row

def load_experiments():

    init_storage()
    if not os.path.exists(CSV_PATH):
        return []
    records = []
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            records.append(r)
    return records

def load_experiment_detail(exp_id):
\
\
\

    records = load_experiments()
    match = next((r for r in records if r["experiment_id"] == exp_id), None)
    if not match:
        return None

    cap_path = os.path.join(CAP_DIR, match["captured_image_name"])
    cap_img = cv2.imread(cap_path) if os.path.exists(cap_path) else None

    ref_name = match.get("reference_image_name", "")
    ref_path = os.path.join(REF_DIR, ref_name)
    ref_img = cv2.imread(ref_path) if ref_name and os.path.exists(ref_path) else None

    b_path = os.path.join(BLUR_MAT_DIR, match.get("blur_matrix_file", ""))
    blur_mat = np.loadtxt(b_path, delimiter=",") if os.path.exists(b_path) else None

    a_path = os.path.join(ANGLE_MAT_DIR, match.get("scattering_angles_file", ""))
    angle_mat = np.loadtxt(a_path, delimiter=",") if os.path.exists(a_path) else None

    return {
        "record": match,
        "captured_image": cap_img,
        "reference_image": ref_img,
        "blur_matrix": blur_mat,
        "angle_matrix": angle_mat,
    }

def get_dataset_stats():
\
\
\
\
\
\
\

    records = load_experiments()
    total = len(records)
    has_ntu = 0
    has_size = 0
    complete = 0

    for r in records:
        ntu_val = r.get("known_NTU", "").strip()
        size_val = r.get("particle_size", "").strip()
        ntu_ok = False
        size_ok = False
        try:
            if ntu_val and not np.isnan(float(ntu_val)):
                ntu_ok = True
                has_ntu += 1
        except ValueError:
            pass
        try:
            if size_val and not np.isnan(float(size_val)):
                size_ok = True
                has_size += 1
        except ValueError:
            pass
        if ntu_ok and size_ok:
            complete += 1

    incomplete = total - complete
    return {
        "total": total,
        "has_ntu": has_ntu,
        "has_size": has_size,
        "complete": complete,
        "incomplete": incomplete,
    }

def export_regression_dataset(dest_path=None):
\
\
\

    records = load_experiments()
    dest = dest_path or REGRESSION_CSV_PATH

    headers = ["experiment_id"] + REGRESSION_FEATURE_COLUMNS + REGRESSION_TARGET_COLUMNS
    exported_rows = []

    for r in records:
        row_dict = {"experiment_id": r["experiment_id"]}

        valid = True
        for col in REGRESSION_FEATURE_COLUMNS:
            val_str = r.get(col, "").strip()
            try:
                row_dict[col] = float(val_str)
            except (ValueError, TypeError):
                row_dict[col] = float("nan")

        for col in REGRESSION_TARGET_COLUMNS:
            val_str = r.get(col, "").strip()
            try:
                row_dict[col] = float(val_str)
            except (ValueError, TypeError):
                row_dict[col] = float("nan")

        has_any_target = any(not np.isnan(row_dict[t]) for t in REGRESSION_TARGET_COLUMNS)
        if has_any_target:
            exported_rows.append(row_dict)

    with open(dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in exported_rows:
            writer.writerow(row)

    return dest, len(exported_rows)

def _row_key(r):
    return (r.get("timestamp", ""), r.get("laplacian_variance", ""), r.get("known_NTU", ""),
            r.get("particle_concentration", ""))


def merge_experiment_csvs(paths):
    """Append the rows of other experiments.csv files into the main dataset
    (data/experiments.csv), and copy their capture / blur-matrix / angle-matrix files.

    * a row already present (same timestamp, laplacian_variance, NTU, concentration) is skipped
    * a row whose experiment_id is already taken gets a new id; its files are renamed to match
    * the main CSV itself is ignored if listed
    Returns {"added": n, "skipped_duplicates": n, "renumbered": n, "files": [...]}."""
    import shutil
    init_storage()
    main_abs = os.path.abspath(CSV_PATH)
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        existing = list(csv.DictReader(f))
    seen = {_row_key(r) for r in existing}
    used = {r.get("experiment_id", "") for r in existing}

    def next_free():
        nums = [int(m.group(1)) for i in used for m in [re.match(r"EXP(\d+)", i or "", re.I)] if m]
        return f"EXP{max(nums, default=0) + 1:03d}"

    added = dup = renum = 0
    done_files = []
    for path in paths:
        if os.path.abspath(path) == main_abs or not os.path.exists(path):
            continue
        src = os.path.dirname(os.path.abspath(path))
        with open(path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        new_rows = []
        for r in rows:
            if _row_key(r) in seen:
                dup += 1
                continue
            old_id = r.get("experiment_id", "")
            new_id = old_id
            if not old_id or old_id in used:
                new_id = next_free()
                renum += 1
            used.add(new_id)
            seen.add(_row_key(r))

            def copy(sub_src, sub_dst, old_name, new_name):
                if not old_name:
                    return old_name
                s_p = os.path.join(src, sub_src, old_name)
                d_p = os.path.join(sub_dst, new_name)
                if os.path.exists(s_p) and not os.path.exists(d_p):
                    shutil.copy2(s_p, d_p)
                return new_name

            old_cap = r.get("captured_image_name", "")
            old_blu = r.get("blur_matrix_file", "")
            old_ang = r.get("scattering_angles_file", "")
            if new_id != old_id:
                r["experiment_id"] = new_id
                r["captured_image_name"] = f"{new_id}.png"
                r["blur_matrix_file"] = f"{new_id}_blur.csv"
                r["scattering_angles_file"] = f"{new_id}_angles.csv"
            copy("captures", CAP_DIR, old_cap, r.get("captured_image_name", ""))
            copy("blur_matrices", BLUR_MAT_DIR, old_blu, r.get("blur_matrix_file", ""))
            copy("scattering_angles", ANGLE_MAT_DIR, old_ang, r.get("scattering_angles_file", ""))
            ref = r.get("reference_image_name", "")
            if ref:
                copy("references", REF_DIR, ref, ref)
            new_rows.append(r)
        if new_rows:
            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=EXPERIMENT_COLUMNS, extrasaction="ignore")
                for r in new_rows:
                    w.writerow(r)
            added += len(new_rows)
            done_files.append(path)
    return {"added": added, "skipped_duplicates": dup, "renumbered": renum, "files": done_files}