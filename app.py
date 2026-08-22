"""
app.py
Desktop GUI for the turbidity/image-processing pipeline:

  pattern -> (simulated blur | live camera capture) -> blur matrix (Fig 2)
  -> scattering angle -> placeholder turbidity/particle-size estimate

Runs today on a laptop with a webcam. The same modules (camera.py in
particular) drop onto a Raspberry Pi with picamera2 installed without any
changes to this file.
"""

import csv
import os
import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

import patterns
import blur
import blur_matrix
import scattering
import calibration
import particle_id
from camera import Camera
from utils import ensure_dir, simulate_blur, save_image, timestamp, \
    draw_line_chart, draw_bar_chart

REF_W, REF_H = 480, 360
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURED_DIR = os.path.join(BASE_DIR, "captured")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ensure_dir(CAPTURED_DIR)
ensure_dir(RESULTS_DIR)

BG = "#0a0d10"
PANEL = "#12161b"
LINE = "#232a32"
TEXT = "#d9e1e8"
DIM = "#7d8a97"
SHARP = "#3ED6C6"
AMBER = "#F2B84B"


class TurbidityApp:
    def __init__(self, root):
        self.root = root
        root.title("Turbidity Image Processing — Test Rig")
        root.configure(bg=BG)

        self.pattern_type = tk.StringVar(value="checker")
        self.feat_size = tk.IntVar(value=24)
        self.mode = tk.StringVar(value="sim")  # sim | cam
        self.blur_radius = tk.IntVar(value=0)
        self.noise_pct = tk.IntVar(value=0)
        self.grid_cols = tk.IntVar(value=12)
        self.metric = tk.StringVar(value="laplacian")
        self.cal_slope = tk.DoubleVar(value=1.0)
        self.cal_k = tk.DoubleVar(value=2.5)

        self.reference_img = None     # sharp generated pattern (BGR)
        self.current_input = None     # image actually analyzed (BGR)
        self.last_matrix = None
        self.last_ratio = None
        self.last_angle = None
        self.cam = None
        self.cam_active = False
        self.particle_library = None  # built lazily on first analysis
        self._particle_lib_key = None  # (pattern, grid_cols, metric) the library was built for

        self._build_ui()
        self._on_generate()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=PANEL)
        style.configure("TLabel", background=PANEL, foreground=DIM,
                         font=("Consolas", 9))
        style.configure("Head.TLabel", background=PANEL, foreground=DIM,
                         font=("Consolas", 9, "bold"))
        style.configure("TButton", font=("Segoe UI", 9))
        style.configure("Horizontal.TScale", background=PANEL)

        root_frame = ttk.Frame(self.root, padding=10)
        root_frame.pack(fill="both", expand=True)

        left = ttk.Frame(root_frame, padding=10, relief="flat")
        left.grid(row=0, column=0, sticky="n", padx=(0, 10))
        right = ttk.Frame(root_frame, padding=0)
        right.grid(row=0, column=1, sticky="nsew")

        self._build_controls(left)
        self._build_display(right)

    def _section(self, parent, title):
        lbl = ttk.Label(parent, text=title, style="Head.TLabel")
        lbl.pack(anchor="w", pady=(12, 4))
        return lbl

    def _build_controls(self, parent):
        ttk.Label(parent, text="TURBIDITYVISION", style="Head.TLabel",
                  font=("Consolas", 12, "bold")).pack(anchor="w")
        ttk.Label(parent, text="software test rig", style="TLabel").pack(anchor="w")

        self._section(parent, "1 · PATTERN")
        opt = ttk.OptionMenu(parent, self.pattern_type, self.pattern_type.get(),
                              *patterns.PATTERN_TYPES,
                              command=lambda _: self._on_generate())
        opt.pack(fill="x")

        ttk.Label(parent, text="Feature size (px)").pack(anchor="w", pady=(6, 0))
        ttk.Scale(parent, from_=6, to=60, variable=self.feat_size,
                  orient="horizontal", command=lambda _: None).pack(fill="x")

        ttk.Button(parent, text="Generate Reference Pattern",
                   command=self._on_generate).pack(fill="x", pady=(6, 0))

        self._section(parent, "2 · INPUT SOURCE")
        mode_row = ttk.Frame(parent)
        mode_row.pack(fill="x")
        ttk.Button(mode_row, text="Simulated Blur",
                   command=lambda: self._set_mode("sim")).pack(side="left", expand=True, fill="x")
        ttk.Button(mode_row, text="Live Camera",
                   command=lambda: self._set_mode("cam")).pack(side="left", expand=True, fill="x")

        ttk.Label(parent, text="Blur radius (px)").pack(anchor="w", pady=(6, 0))
        ttk.Scale(parent, from_=0, to=14, variable=self.blur_radius,
                  orient="horizontal").pack(fill="x")
        ttk.Label(parent, text="Noise (%)").pack(anchor="w", pady=(4, 0))
        ttk.Scale(parent, from_=0, to=40, variable=self.noise_pct,
                  orient="horizontal").pack(fill="x")
        ttk.Button(parent, text="Apply Blur → Analyze",
                   command=self._run_sim_analysis).pack(fill="x", pady=(6, 2))

        ttk.Button(parent, text="Start Camera",
                   command=self._start_camera).pack(fill="x", pady=(4, 2))
        ttk.Button(parent, text="Capture Frame → Analyze",
                   command=self._capture_and_analyze).pack(fill="x")

        self._section(parent, "3 · KERNEL GRID")
        ttk.Label(parent, text="Grid columns").pack(anchor="w")
        ttk.Scale(parent, from_=4, to=24, variable=self.grid_cols,
                  orient="horizontal").pack(fill="x")
        metric_row = ttk.Frame(parent)
        metric_row.pack(fill="x", pady=(4, 0))
        ttk.Radiobutton(metric_row, text="Laplacian", variable=self.metric,
                        value="laplacian").pack(side="left")
        ttk.Radiobutton(metric_row, text="Tenengrad", variable=self.metric,
                        value="tenengrad").pack(side="left")
        ttk.Radiobutton(metric_row, text="FFT", variable=self.metric,
                        value="fft").pack(side="left")

        self._section(parent, "4 · CALIBRATION (placeholder)")
        ttk.Label(parent, text="Turbidity slope (NTU / blur%)").pack(anchor="w")
        ttk.Scale(parent, from_=0.2, to=3.0, variable=self.cal_slope,
                  orient="horizontal").pack(fill="x")
        ttk.Label(parent, text="Particle-size k (µm/px)").pack(anchor="w")
        ttk.Scale(parent, from_=0.5, to=6.0, variable=self.cal_k,
                  orient="horizontal").pack(fill="x")

        self._section(parent, "5 · VALIDATION")
        ttk.Button(parent, text="Run Sensitivity Sweep",
                   command=self._run_sweep).pack(fill="x", pady=(0, 2))
        ttk.Button(parent, text="Compare All Patterns",
                   command=self._run_compare).pack(fill="x")

        self._section(parent, "EXPORT")
        ttk.Button(parent, text="Export Blur Matrix (CSV)",
                   command=self._export_csv).pack(fill="x")
        ttk.Button(parent, text="Save Snapshot (results/)",
                   command=self._save_snapshot).pack(fill="x", pady=(2, 0))

        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(parent, textvariable=self.status_var, wraplength=260,
                  foreground=AMBER).pack(anchor="w", pady=(10, 0))

    def _build_display(self, parent):
        cams = ttk.Frame(parent)
        cams.pack(fill="x")

        titles = ["Input Frame", "Blur Matrix", "Scattering Angle"]
        labels = []
        for i, title in enumerate(titles):
            col = ttk.Frame(cams, padding=4)
            col.grid(row=0, column=i, padx=4)
            ttk.Label(col, text=title, style="Head.TLabel").pack(anchor="w")
            lbl = tk.Label(col, bg="black")
            lbl.pack()
            labels.append(lbl)

        self.canvas_in, self.canvas_blur, self.canvas_angle = labels

        metrics = ttk.Frame(parent, padding=(0, 12))
        metrics.pack(fill="x")
        self.metric_vars = {}
        for i, key in enumerate(["Mean Blur Ratio", "Mean Angle °",
                                  "Est. Turbidity NTU", "Est. Particle µm",
                                  "Compute ms", "Particle Type", "Concentration"]):
            box = ttk.Frame(metrics, padding=8)
            box.grid(row=0, column=i, padx=4)
            v = tk.StringVar(value="—")
            ttk.Label(box, textvariable=v, font=("Consolas", 16),
                      foreground=SHARP, background=PANEL).pack()
            ttk.Label(box, text=key, style="TLabel").pack()
            self.metric_vars[key] = v

    # ------------------------------------------------------------------
    # Pattern / mode handling
    # ------------------------------------------------------------------
    def _on_generate(self):
        img = patterns.generate_pattern(self.pattern_type.get(), REF_W, REF_H,
                                         self.feat_size.get())
        self.reference_img = img
        self._show(self.canvas_in, img)
        self.status_var.set("Reference pattern generated.")

    def _set_mode(self, mode):
        self.mode.set(mode)
        self.status_var.set(f"Mode: {'Simulated Blur' if mode=='sim' else 'Live Camera'}")

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def _start_camera(self):
        if self.cam is None:
            try:
                self.cam = Camera(width=REF_W, height=REF_H)
            except Exception as e:
                messagebox.showerror("Camera error", str(e))
                return
        self.mode.set("cam")
        self.cam_active = True
        self.status_var.set(f"Camera live ({self.cam.backend}). Point at pattern, then capture.")
        self._preview_loop()

    def _preview_loop(self):
        if not self.cam_active or self.cam is None:
            return
        try:
            frame = self.cam.read_frame()
            self.current_input = frame
            self._show(self.canvas_in, frame)
        except Exception as e:
            self.status_var.set(f"Camera read error: {e}")
            self.cam_active = False
            return
        self.root.after(80, self._preview_loop)

    def _capture_and_analyze(self):
        if self.cam is None or self.current_input is None:
            self.status_var.set("Start the camera first.")
            return
        frame = self.current_input.copy()
        path = os.path.join(CAPTURED_DIR, f"frame_{timestamp()}.png")
        save_image(path, frame)
        self._analyze(frame)
        self.status_var.set(f"Captured frame analyzed and saved to {os.path.basename(path)}.")

    # ------------------------------------------------------------------
    # Analysis pipeline
    # ------------------------------------------------------------------
    def _run_sim_analysis(self):
        if self.reference_img is None:
            self.status_var.set("Generate a reference pattern first.")
            return
        img = simulate_blur(self.reference_img, self.blur_radius.get(),
                             self.noise_pct.get())
        self.current_input = img
        self._show(self.canvas_in, img)
        self._analyze(img)
        self.status_var.set("Simulated-blur analysis complete.")

    def _analyze(self, img):
        t0 = time.time()
        matrix = blur_matrix.compute_blur_matrix(
            img, grid_cols=self.grid_cols.get(), method=self.metric.get())
        ratio = blur_matrix.to_blur_ratio(matrix)
        angle = scattering.blur_ratio_to_angle(ratio)

        lib_key = (self.pattern_type.get(), self.grid_cols.get(), self.metric.get())
        if self.particle_library is None or self._particle_lib_key != lib_key:
            self.status_var.set("Building placeholder particle-signature library...")
            self.root.update_idletasks()
            self.particle_library = particle_id.build_reference_library(
                pattern_type=self.pattern_type.get(), grid_cols=self.grid_cols.get(),
                method=self.metric.get())
            self._particle_lib_key = lib_key
        pid_result = particle_id.identify(
            img, self.particle_library, grid_cols=self.grid_cols.get(),
            method=self.metric.get())
        t1 = time.time()

        heat_blur = blur_matrix.matrix_to_heatmap(ratio, REF_W, REF_H, annotate=True)
        heat_angle = blur_matrix.matrix_to_heatmap(angle / 45.0, REF_W, REF_H, annotate=True)
        self._show(self.canvas_blur, heat_blur)
        self._show(self.canvas_angle, heat_angle)

        mean_blur = float(ratio.mean())
        mean_angle = float(angle.mean())
        cal = calibration.CalibrationModel(slope=self.cal_slope.get(),
                                            intercept=0.0, particle_k=self.cal_k.get())
        radius_for_particle = self.blur_radius.get() if self.mode.get() == "sim" \
            else mean_blur * 14
        ntu = cal.blur_to_ntu(mean_blur)
        particle = cal.radius_to_particle_size(radius_for_particle)

        self.metric_vars["Mean Blur Ratio"].set(f"{mean_blur*100:.1f}%")
        self.metric_vars["Mean Angle °"].set(f"{mean_angle:.1f}")
        self.metric_vars["Est. Turbidity NTU"].set(f"{ntu:.1f}")
        self.metric_vars["Est. Particle µm"].set(f"{particle:.1f}")
        self.metric_vars["Compute ms"].set(f"{(t1-t0)*1000:.1f}")
        self.metric_vars["Particle Type"].set(pid_result["particle_type"])
        self.metric_vars["Concentration"].set(pid_result["concentration_band"])

        self.last_matrix, self.last_ratio, self.last_angle = matrix, ratio, angle

    # ------------------------------------------------------------------
    # Validation: sweep + pattern comparison
    # ------------------------------------------------------------------
    def _run_sweep(self):
        if self.reference_img is None:
            self.status_var.set("Generate a reference pattern first.")
            return
        radii, scores = [], []
        method = self.metric.get()
        for r in range(0, 13):
            test_img = simulate_blur(self.reference_img, r, 0)
            gray = blur.to_gray(test_img)
            score = blur.blur_score(gray, method)
            radii.append(r)
            scores.append(score)
        chart = draw_line_chart(radii, scores, xlabel="blur radius (px)",
                                 ylabel=f"sharpness ({method})",
                                 title="Sensitivity Sweep")
        out_path = os.path.join(RESULTS_DIR, f"sweep_{timestamp()}.png")
        save_image(out_path, chart)
        self._show_popup("Sensitivity Sweep", chart)
        self.status_var.set(f"Sweep saved to results/{os.path.basename(out_path)}. "
                             f"Sharpness should fall monotonically.")

    def _run_compare(self):
        feat = self.feat_size.get()
        results = []
        for t in patterns.PATTERN_TYPES:
            sharp_img = patterns.generate_pattern(t, REF_W, REF_H, feat)
            sharp_score = blur.blur_score(sharp_img, "laplacian")
            blurred_img = simulate_blur(sharp_img, 6, 0)
            blur_score = blur.blur_score(blurred_img, "laplacian")
            sensitivity = (sharp_score - blur_score) / sharp_score if sharp_score > 0 else 0
            results.append((patterns.PATTERN_NAMES[t], sensitivity))
        results.sort(key=lambda x: -x[1])
        labels = [r[0] for r in results]
        values = [r[1] for r in results]
        chart = draw_bar_chart(labels, values, title="Pattern Sensitivity @ 6px blur")
        out_path = os.path.join(RESULTS_DIR, f"pattern_comparison_{timestamp()}.png")
        save_image(out_path, chart)
        self._show_popup("Pattern Comparison", chart)
        self.status_var.set(f"Comparison saved to results/{os.path.basename(out_path)}.")
        # restore current pattern as reference
        self._on_generate()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _export_csv(self):
        if self.last_matrix is None:
            self.status_var.set("Run an analysis first.")
            return
        path = os.path.join(RESULTS_DIR, f"blur_matrix_{timestamp()}.csv")
        rows, cols = self.last_matrix.shape
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row", "col", "sharpness_raw", "blur_ratio", "scattering_angle_deg"])
            for r in range(rows):
                for c in range(cols):
                    w.writerow([r, c, f"{self.last_matrix[r,c]:.4f}",
                                f"{self.last_ratio[r,c]:.4f}",
                                f"{self.last_angle[r,c]:.2f}"])
        self.status_var.set(f"Exported {os.path.basename(path)}")

    def _save_snapshot(self):
        if self.current_input is None:
            self.status_var.set("Nothing to save yet.")
            return
        ts = timestamp()
        save_image(os.path.join(RESULTS_DIR, f"input_{ts}.png"), self.current_input)
        if self.last_ratio is not None:
            heat = blur_matrix.matrix_to_heatmap(self.last_ratio, REF_W, REF_H, annotate=True)
            save_image(os.path.join(RESULTS_DIR, f"blur_heatmap_{ts}.png"), heat)
        self.status_var.set(f"Snapshot saved to results/ ({ts}).")

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
    def _show(self, label_widget, bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        im.thumbnail((360, 270))
        tkimg = ImageTk.PhotoImage(im)
        label_widget.imgtk = tkimg  # keep reference
        label_widget.configure(image=tkimg)

    def _show_popup(self, title, bgr_img):
        top = tk.Toplevel(self.root)
        top.title(title)
        top.configure(bg=BG)
        lbl = tk.Label(top, bg="black")
        lbl.pack(padx=8, pady=8)
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        tkimg = ImageTk.PhotoImage(im)
        lbl.imgtk = tkimg
        lbl.configure(image=tkimg)

    def on_close(self):
        self.cam_active = False
        if self.cam is not None:
            self.cam.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = TurbidityApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()