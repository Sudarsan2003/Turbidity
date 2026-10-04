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

import cv2
import numpy as np

import blur
import blur_matrix
import alignment

STAGE_NAMES = [
    "1. Reference Image",
    "2. Captured Image",
    "3. Resized Image",
    "4. Grayscale Image",
    "5. Normalized Image",
    "6. Aligned Image",
    "7. Difference Image",
    "8. Kernel/Grid Division",
    "9. Blur Calculation",
    "10. Blur Matrix",
    "11. Scattering Angle Matrix",
]

def run_preprocessing_pipeline(ref_img, cap_img, grid_cols=12, grid_rows=None,
                              method="laplacian", target_size=(480, 360), kernel=None):
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

    w_t, h_t = target_size

    if ref_img is None:

        ref_stage = np.full((h_t, w_t, 3), 128, dtype=np.uint8)
        cv2.putText(ref_stage, "No Reference Loaded", (w_t // 6, h_t // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    else:
        ref_stage = ref_img.copy()

    cap_stage = cap_img.copy()

    resized_cap = cv2.resize(cap_stage, (w_t, h_t), interpolation=cv2.INTER_AREA)
    resized_ref = cv2.resize(ref_stage, (w_t, h_t), interpolation=cv2.INTER_AREA)

    gray_cap = blur.to_gray(resized_cap).astype(np.uint8)
    gray_cap_bgr = cv2.cvtColor(gray_cap, cv2.COLOR_GRAY2BGR)

    norm_u8 = cv2.normalize(gray_cap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    norm_bgr = cv2.cvtColor(norm_u8, cv2.COLOR_GRAY2BGR)

    align_res = alignment.align_images(resized_ref, resized_cap)
    aligned_bgr = align_res["aligned_bgr"]
    diff_color = align_res["diff_color"]
    overlay_bgr = align_res["overlay_bgr"]

    if kernel is not None:  # (m, n) kernel in pixels -> (p/m) x (q/n) blur matrix
        grid_rows, grid_cols = blur_matrix.grid_from_kernel(h_t, w_t, kernel[0], kernel[1])
    if grid_rows is None:
        grid_rows = max(2, round(grid_cols * h_t / w_t))
    else:
        grid_rows = max(2, int(grid_rows))
    grid_cols = max(2, int(grid_cols))

    grid_div_bgr = blur_matrix.draw_grid_overlay(aligned_bgr, grid_cols=grid_cols,
                                                 grid_rows=grid_rows,
                                                 color=(0, 230, 255), thickness=1)

    effective_method = "laplacian" if method == "all" else method
    blur_calc_gray = blur.compute_blur_map(align_res["aligned_gray"], method=effective_method)
    blur_calc_bgr = cv2.applyColorMap(blur_calc_gray, cv2.COLORMAP_VIRIDIS)

    all_blur_matrices = blur_matrix.compute_all_blur_matrices(aligned_bgr,
                                                              grid_cols=grid_cols,
                                                              grid_rows=grid_rows)
    all_ref_matrices = blur_matrix.compute_all_blur_matrices(resized_ref,
                                                             grid_cols=grid_cols,
                                                             grid_rows=grid_rows)

    all_angle_matrices = {}
    for m in ("laplacian", "tenengrad", "fft"):
        all_angle_matrices[m] = blur_matrix.compute_scattering_angle_matrix(
            all_blur_matrices[m], all_ref_matrices[m], mode="ratio"
        )

    active_m = "laplacian" if method == "all" else method
    primary_blur_matrix = all_blur_matrices[active_m]
    primary_angle_matrix = all_angle_matrices[active_m]

    blur_stats = blur_matrix.matrix_stats(primary_blur_matrix)
    angle_stats = blur_matrix.matrix_stats(primary_angle_matrix)

    blur_mat_heatmap = blur_matrix.matrix_to_heatmap(primary_blur_matrix, out_w=w_t, out_h=h_t)

    angle_mat_heatmap = blur_matrix.matrix_to_heatmap(primary_angle_matrix, vmin=0.0, vmax=45.0,
                                                      cmap=cv2.COLORMAP_TURBO, out_w=w_t, out_h=h_t)

    all_scores = blur.compute_all_blur_scores(aligned_bgr)
    ref_scores = blur.compute_all_blur_scores(resized_ref)

    stages = [
        {"name": STAGE_NAMES[0], "image": resized_ref, "desc": "Baseline clear projected pattern"},
        {"name": STAGE_NAMES[1], "image": cap_stage, "desc": "Raw captured frame from camera/sample"},
        {"name": STAGE_NAMES[2], "image": resized_cap, "desc": f"Resized to standard {w_t}x{h_t} px"},
        {"name": STAGE_NAMES[3], "image": gray_cap_bgr, "desc": "Grayscale single-channel luminance"},
        {"name": STAGE_NAMES[4], "image": norm_bgr, "desc": "Min-Max contrast normalized luminance"},
        {"name": STAGE_NAMES[5], "image": aligned_bgr, "desc": f"Phase-correlated registration (shift: {align_res['dx']:.1f}, {align_res['dy']:.1f}px)"},
        {"name": STAGE_NAMES[6], "image": diff_color, "desc": f"Absolute difference |Sample - Ref| (mean diff: {align_res['diff_mean']:.2f})"},
        {"name": STAGE_NAMES[7], "image": grid_div_bgr, "desc": f"Kernel {h_t // grid_rows}x{w_t // grid_cols} px -> blur matrix {grid_rows} x {grid_cols} ({grid_rows * grid_cols} cells)"},
        {"name": STAGE_NAMES[8], "image": blur_calc_bgr, "desc": f"Local edge/gradient response ({active_m})"},
        {"name": STAGE_NAMES[9], "image": blur_mat_heatmap, "desc": f"Blur Matrix (mean: {blur_stats['mean']:.2f}, std: {blur_stats['std']:.2f})"},
        {"name": STAGE_NAMES[10], "image": angle_mat_heatmap, "desc": f"Scattering Angle Matrix (mean: {angle_stats['mean']:.2f}°, max: {angle_stats['max']:.2f}°)"},
    ]

    return {
        "stages": stages,
        "aligned_bgr": aligned_bgr,
        "ref_bgr": resized_ref,
        "diff_gray": align_res["diff_gray"],
        "diff_color": diff_color,
        "overlay_bgr": overlay_bgr,
        "blur_matrix": primary_blur_matrix,
        "angle_matrix": primary_angle_matrix,
        "blur_stats": blur_stats,
        "angle_stats": angle_stats,
        "all_blur_scores": all_scores,
        "ref_blur_scores": ref_scores,
        "all_blur_matrices": all_blur_matrices,
        "all_angle_matrices": all_angle_matrices,
        "shift_dx": align_res["dx"],
        "shift_dy": align_res["dy"],
        "diff_mean": align_res["diff_mean"],
        "diff_std": align_res["diff_std"],
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "kernel_px": (h_t // grid_rows, w_t // grid_cols),
    }