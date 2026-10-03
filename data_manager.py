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
DATA_DIR = os.path.join(BASE_DIR, "data")
REF_DIR = os.path.join(DATA_DIR, "references")
CAP_DIR = os.path.join(DATA_DIR, "captures")
BLUR_MAT_DIR = os.path.join(DATA_DIR, "blur_matrices")
ANGLE_MAT_DIR = os.path.join(DATA_DIR, "scattering_angles")
CSV_PATH = os.path.join(DATA_DIR, "experiments.csv")
REGRESSION_CSV_PATH = os.path.join(DATA_DIR, "regression_dataset.csv")

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
    "blur_matrix_file",
    "scattering_angles_file",
    "status",
]

REGRESSION_FEATURE_COLUMNS = [
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
    "ref_laplacian_variance",
    "ref_tenengrad",
    "ref_fft_energy",
]

REGRESSION_TARGET_COLUMNS = [
    "known_NTU",
    "particle_size",
    "particle_concentration",
]

def init_storage():

    for d in (DATA_DIR, REF_DIR, CAP_DIR, BLUR_MAT_DIR, ANGLE_MAT_DIR):
        os.makedirs(d, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(EXPERIMENT_COLUMNS)

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
        cv2.imwrite(target_path, img)

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
    cv2.imwrite(cap_path, cap_img)

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
        "kernel_columns": proc_res.get("grid_cols", 16),
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
