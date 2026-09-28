import csv
import os
import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np

import blur_matrix
import scattering
import calibration
from camera import Camera
from utils import ensure_dir, save_image, timestamp
from ui_widgets import (BG, PANEL, CARD, FIELD, LINE, TEXT, DIM, SHARP, AMBER,
                        BLUE, VIOLET, RED, FONT_UI, FONT_MONO,
                        MatrixView, StageCard, fmt_compact, fmt_ratio, fmt_angle)

CAM_W, CAM_H = 480, 360
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURED_DIR = os.path.join(BASE_DIR, "captured")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ensure_dir(CAPTURED_DIR)
ensure_dir(RESULTS_DIR)

METRIC_INFO = {
    "laplacian": ("Laplacian Values", "var(∇²I) per kernel · higher = sharper", ""),
    "tenengrad": ("Tenengrad Values", "mean(Gx²+Gy²) per kernel · higher = sharper", ""),
    "fft":       ("FFT HF-Energy Values", "high-freq / total energy (%) · higher = sharper", "%"),
}
METRIC_LONG = {
    "laplacian": "Variance of Laplacian",
    "tenengrad": "Tenengrad (Sobel gradient energy)",
    "fft": "FFT high-frequency energy",
}
LIVE_INTERVAL_S = 0.35


class TurbidityApp:
    def __init__(self, root):
        self.root = root
        root.title("TurbidityVision — Live Camera")
        root.configure(bg=BG)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{min(1560, sw - 40)}x{min(940, sh - 80)}+20+20")
        root.minsize(1100, 680)

        self.grid_cols = tk.IntVar(value=12)
        self.metric = tk.StringVar(value="laplacian")
        self.cal_slope = tk.DoubleVar(value=1.0)
        self.cal_k = tk.DoubleVar(value=2.5)
        self.animate = tk.BooleanVar(value=True)
        self.live = tk.BooleanVar(value=False)

        self.current_input = None
        self.last_input = None
        self.last_matrix = None
        self.last_ratio = None
        self.last_angle = None
        self.cam = None
        self.cam_active = False
        self._last_live = 0.0
        self._stage_jobs = []

        self._build_ui()
        self.root.after(300, lambda: self._start_camera(silent=True))

    def _setup_style(self):
        st = ttk.Style()
        st.theme_use("clam")
        r = self.root
        r.option_add("*TCombobox*Listbox.background", FIELD)
        r.option_add("*TCombobox*Listbox.foreground", TEXT)
        r.option_add("*TCombobox*Listbox.selectBackground", SHARP)
        r.option_add("*TCombobox*Listbox.selectForeground", "#04110f")

        st.configure("TFrame", background=PANEL)
        st.configure("TLabel", background=PANEL, foreground=DIM, font=(FONT_UI, 9))
        st.configure("Val.TLabel", background=PANEL, foreground=SHARP,
                     font=(FONT_MONO, 9, "bold"))
        st.configure("Sec.TLabel", background=PANEL, foreground=TEXT,
                     font=(FONT_UI, 10, "bold"))
        st.configure("Num.TLabel", background=PANEL, foreground=SHARP,
                     font=(FONT_MONO, 9, "bold"))

        st.configure("TButton", background=FIELD, foreground=TEXT,
                     bordercolor=LINE, lightcolor=FIELD, darkcolor=FIELD,
                     focuscolor=FIELD, padding=(10, 6), font=(FONT_UI, 9))
        st.map("TButton", background=[("active", "#2a3541"), ("pressed", "#33414f")])

        st.configure("Accent.TButton", background=SHARP, foreground="#04110f",
                     bordercolor=SHARP, lightcolor=SHARP, darkcolor=SHARP,
                     focuscolor=SHARP, padding=(10, 8), font=(FONT_UI, 9, "bold"))
        st.map("Accent.TButton", background=[("active", "#66e8db"), ("pressed", "#2cb5a6")])

        st.configure("Seg.Toolbutton", background=FIELD, foreground=DIM,
                     bordercolor=LINE, lightcolor=FIELD, darkcolor=FIELD,
                     padding=(6, 5), anchor="center", font=(FONT_UI, 9))
        st.map("Seg.Toolbutton",
               background=[("selected", SHARP), ("active", "#2a3541")],
               foreground=[("selected", "#04110f"), ("active", TEXT)],
               lightcolor=[("selected", SHARP)], darkcolor=[("selected", SHARP)])

        st.configure("TCheckbutton", background=PANEL, foreground=TEXT,
                     font=(FONT_UI, 9), indicatorcolor=FIELD,
                     indicatorbackground=FIELD, focuscolor=PANEL)
        st.map("TCheckbutton", background=[("active", PANEL)],
               indicatorcolor=[("selected", SHARP)])

        st.configure("Teal.Horizontal.TScale", background=SHARP, troughcolor=LINE,
                     bordercolor=PANEL, lightcolor=SHARP, darkcolor=SHARP)
        st.configure("Vertical.TScrollbar", background=LINE, troughcolor=PANEL,
                     bordercolor=PANEL, arrowcolor=DIM, lightcolor=LINE, darkcolor=LINE)

    def _build_ui(self):
        self._setup_style()

        header = tk.Frame(self.root, bg=PANEL, height=54)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="◉", bg=PANEL, fg=SHARP,
                 font=(FONT_UI, 18)).pack(side="left", padx=(16, 6))
        tk.Label(header, text="TURBIDITYVISION", bg=PANEL, fg=TEXT,
                 font=(FONT_MONO, 13, "bold")).pack(side="left")
        tk.Label(header, text="  image-blur → scattering-angle test rig", bg=PANEL,
                 fg=DIM, font=(FONT_UI, 9)).pack(side="left", pady=(4, 0))
        self.mode_pill = tk.Label(header, text="● CAMERA OFF", bg=FIELD, fg=DIM,
                                  font=(FONT_MONO, 9, "bold"), padx=12, pady=5)
        self.mode_pill.pack(side="right", padx=16)
        tk.Frame(self.root, bg=LINE, height=1).pack(fill="x")

        status = tk.Frame(self.root, bg=PANEL, height=28)
        status.pack(side="bottom", fill="x")
        status.pack_propagate(False)
        tk.Frame(self.root, bg=LINE, height=1).pack(side="bottom", fill="x")
        self.status_var = tk.StringVar(value="Idle.")
        tk.Label(status, text="●", bg=PANEL, fg=SHARP,
                 font=(FONT_UI, 8)).pack(side="left", padx=(12, 4))
        tk.Label(status, textvariable=self.status_var, bg=PANEL, fg=AMBER,
                 font=(FONT_UI, 9)).pack(side="left")
        self.hover_var = tk.StringVar(value="Hover a kernel in any matrix to inspect it")
        tk.Label(status, textvariable=self.hover_var, bg=PANEL, fg=TEXT,
                 font=(FONT_MONO, 9)).pack(side="right", padx=12)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left_wrap = tk.Frame(body, bg=PANEL, width=310)
        left_wrap.grid(row=0, column=0, sticky="ns")
        left_wrap.pack_propagate(False)
        inp = tk.Frame(left_wrap, bg=PANEL, height=250)
        inp.pack(fill="x", padx=10, pady=(10, 0))
        inp.pack_propagate(False)
        self.card_in = StageCard(inp, "▣", "Camera Frame", "live input", BLUE, kind="image")
        self.card_in.pack(fill="both", expand=True)
        self.card_in.set_footer("camera not started")
        scroll_host = tk.Frame(left_wrap, bg=PANEL)
        scroll_host.pack(fill="both", expand=True)
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=12, pady=10)

        self._build_controls(self._scrollable(scroll_host))
        self._build_display(right)

    def _scrollable(self, wrap):
        canvas = tk.Canvas(wrap, bg=PANEL, highlightthickness=0, width=290)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, padding=(14, 6, 10, 14))
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))

        def _wheel(e):
            if e.num == 4:
                canvas.yview_scroll(-2, "units")
            elif e.num == 5:
                canvas.yview_scroll(2, "units")
            else:
                canvas.yview_scroll(-1 if e.delta > 0 else 1, "units")

        def _bind(_e):
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                canvas.bind_all(ev, _wheel)

        def _unbind(_e):
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                canvas.unbind_all(ev)

        wrap.bind("<Enter>", _bind)
        wrap.bind("<Leave>", _unbind)
        return inner

    def _section(self, parent, num, title):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(16, 6))
        ttk.Label(row, text=num, style="Num.TLabel").pack(side="left")
        ttk.Label(row, text=f"  {title}", style="Sec.TLabel").pack(side="left")
        tk.Frame(parent, bg=LINE, height=1).pack(fill="x", pady=(0, 4))

    def _slider(self, parent, text, var, lo, hi, fmt="{:.0f}", integer=True,
                on_release=None):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(6, 0))
        ttk.Label(row, text=text).pack(side="left")
        val = ttk.Label(row, text=fmt.format(var.get()), style="Val.TLabel")
        val.pack(side="right")

        def _cmd(v):
            x = float(v)
            if integer:
                xi = int(round(x))
                if var.get() != xi:
                    var.set(xi)
                x = xi
            val.config(text=fmt.format(x))

        sc = ttk.Scale(parent, from_=lo, to=hi, variable=var, orient="horizontal",
                       command=_cmd, style="Teal.Horizontal.TScale")
        sc.pack(fill="x", pady=(2, 0))
        if on_release:
            sc.bind("<ButtonRelease-1>", lambda e: on_release())
        return sc

    def _build_controls(self, p):
        self._section(p, "01", "CAMERA")
        self.cam_btn = ttk.Button(p, text="Start Camera", command=self._toggle_camera)
        self.cam_btn.pack(fill="x")
        ttk.Button(p, text="▶  Capture → Analyze", style="Accent.TButton",
                   command=self._capture_and_analyze).pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(p, text="Live analysis (every frame)",
                        variable=self.live).pack(anchor="w", pady=(10, 0))
        ttk.Checkbutton(p, text="Animate stages on capture",
                        variable=self.animate).pack(anchor="w", pady=(2, 0))

        self._section(p, "02", "ALGORITHM & KERNEL GRID")
        ttk.Label(p, text="Blur metric").pack(anchor="w")
        seg2 = ttk.Frame(p)
        seg2.pack(fill="x", pady=(2, 0))
        for txt, val in (("Laplacian", "laplacian"), ("Tenengrad", "tenengrad"), ("FFT", "fft")):
            ttk.Radiobutton(seg2, text=txt, variable=self.metric, value=val,
                            style="Seg.Toolbutton",
                            command=self._on_metric_change
                            ).pack(side="left", expand=True, fill="x")
        self._slider(p, "Grid columns", self.grid_cols, 4, 24,
                     on_release=self._reanalyze)

        self._section(p, "03", "CALIBRATION  (placeholder)")
        self._slider(p, "Turbidity slope (NTU / blur%)", self.cal_slope, 0.2, 3.0,
                     fmt="{:.2f}", integer=False, on_release=self._reanalyze)
        self._slider(p, "Particle-size k (µm/px)", self.cal_k, 0.5, 6.0,
                     fmt="{:.2f}", integer=False, on_release=self._reanalyze)

        self._section(p, "··", "EXPORT")
        ttk.Button(p, text="Export Matrices (CSV)",
                   command=self._export_csv).pack(fill="x")
        ttk.Button(p, text="Save Snapshot (results/)",
                   command=self._save_snapshot).pack(fill="x", pady=(3, 0))

    def _build_display(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        pipe = tk.Frame(parent, bg=BG)
        pipe.grid(row=0, column=0, sticky="nsew")
        pipe.rowconfigure(0, weight=1)
        for i in range(3):
            pipe.columnconfigure(i, weight=1, uniform="stage")

        self.card_raw = StageCard(pipe, "1", *METRIC_INFO["laplacian"][:2], AMBER,
                                  cmap=cv2.COLORMAP_VIRIDIS, fmt=fmt_compact,
                                  hover_cb=self._on_hover,
                                  dbl_cb=lambda: self._open_popup("raw"))
        self.card_ratio = StageCard(pipe, "2", "Blur Matrix",
                                    "1−(v−min)/(max−min) · 0 sharp, 1 blurry",
                                    SHARP, cmap=cv2.COLORMAP_JET, fmt=fmt_ratio,
                                    hover_cb=self._on_hover,
                                    dbl_cb=lambda: self._open_popup("ratio"))
        self.card_angle = StageCard(pipe, "3", "Scattering Angle Matrix",
                                    "θ = atan(blur ratio) · degrees",
                                    VIOLET, cmap=cv2.COLORMAP_INFERNO, fmt=fmt_angle,
                                    hover_cb=self._on_hover,
                                    dbl_cb=lambda: self._open_popup("angle"))
        for i, c in enumerate((self.card_raw, self.card_ratio, self.card_angle)):
            c.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 5, 0 if i == 2 else 5))
            if i < 2:
                tk.Label(pipe, text="›", bg=BG, fg=DIM, font=(FONT_UI, 16)).place(
                    in_=c, relx=1.0, rely=0.5, anchor="center", x=6)
        self.card_raw.view.clear("capture a frame to analyze")
        self.card_ratio.view.clear("waiting for step 1")
        self.card_angle.view.clear("waiting for step 2")

        tiles = tk.Frame(parent, bg=BG)
        tiles.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.metric_vars = {}
        keys = [("Mean Blur Ratio", SHARP), ("Mean Angle °", VIOLET),
                ("Est. Turbidity NTU", AMBER), ("Est. Particle µm", BLUE),
                ("Compute ms", DIM)]
        for i, (key, accent) in enumerate(keys):
            tiles.columnconfigure(i, weight=1, uniform="tile")
            box = tk.Frame(tiles, bg=CARD, highlightbackground=LINE, highlightthickness=1)
            box.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 4, 0 if i == 4 else 4))
            tk.Frame(box, bg=accent, height=3).pack(fill="x")
            v = tk.StringVar(value="—")
            tk.Label(box, textvariable=v, bg=CARD, fg=accent,
                     font=(FONT_MONO, 15, "bold")).pack(pady=(8, 0))
            tk.Label(box, text=key, bg=CARD, fg=DIM,
                     font=(FONT_UI, 8)).pack(pady=(0, 8))
            self.metric_vars[key] = v

        logbox = tk.Frame(parent, bg=CARD, highlightbackground=LINE, highlightthickness=1)
        logbox.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        tk.Label(logbox, text="CALCULATION BREAKDOWN", bg=CARD, fg=DIM,
                 font=(FONT_MONO, 8, "bold")).pack(anchor="w", padx=12, pady=(8, 0))
        self.log = tk.Text(logbox, height=10, bg=CARD, fg=TEXT, bd=0,
                           highlightthickness=0, font=(FONT_MONO, 9), wrap="none",
                           padx=12, pady=6, state="disabled", cursor="arrow")
        self.log.pack(fill="x")
        self.log.tag_configure("step", foreground=AMBER, font=(FONT_MONO, 9, "bold"))
        self.log.tag_configure("val", foreground=SHARP)
        self.log.tag_configure("dim", foreground=DIM)
        self.log.tag_configure("warn", foreground=RED)
        self.log.tag_configure("res", foreground=VIOLET, font=(FONT_MONO, 9, "bold"))
        self._log_write([("Start the camera, then press Capture → Analyze (or enable live "
                          "analysis) to see how the raw metric values become the blur "
                          "matrix and then scattering angles.\n", "dim")])

    def _on_metric_change(self):
        name = METRIC_INFO[self.metric.get()]
        self.card_raw.set_title(name[0], name[1])
        self._reanalyze()

    def _reanalyze(self):
        if self.last_input is not None:
            self._analyze(self.last_input, staged=False)

    def _toggle_camera(self):
        if self.cam_active:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self, silent=False):
        if self.cam is None:
            try:
                self.cam = Camera(width=CAM_W, height=CAM_H)
            except Exception as e:
                self.cam = None
                self.status_var.set(f"Camera unavailable: {e}")
                if not silent:
                    messagebox.showerror("Camera error", str(e))
                return
        self.cam_active = True
        self.cam_btn.config(text="Stop Camera")
        self.mode_pill.config(text="● LIVE CAMERA", fg=SHARP)
        self.status_var.set(f"Camera live ({self.cam.backend}). Press Capture → Analyze.")
        self._preview_loop()

    def _stop_camera(self):
        self.cam_active = False
        if self.cam is not None:
            self.cam.release()
            self.cam = None
        self.cam_btn.config(text="Start Camera")
        self.mode_pill.config(text="● CAMERA OFF", fg=DIM)
        self.card_in.set_footer("camera stopped")
        self.status_var.set("Camera stopped.")

    def _preview_loop(self):
        if not self.cam_active or self.cam is None:
            return
        try:
            frame = self.cam.read_frame()
            self.current_input = frame
            self.card_in.view.set_image(frame)
            self.card_in.set_footer(f"{CAM_W}×{CAM_H} · {self.cam.backend}")
            if self.live.get():
                now = time.time()
                if now - self._last_live >= LIVE_INTERVAL_S:
                    self._last_live = now
                    self._analyze(frame.copy(), staged=False)
        except Exception as e:
            self.status_var.set(f"Camera read error: {e}")
            self.cam_active = False
            self.cam_btn.config(text="Start Camera")
            self.mode_pill.config(text="● CAMERA OFF", fg=DIM)
            return
        self.root.after(80, self._preview_loop)

    def _capture_and_analyze(self):
        if self.cam is None or self.current_input is None:
            self.status_var.set("Start the camera first.")
            return
        frame = self.current_input.copy()
        path = os.path.join(CAPTURED_DIR, f"frame_{timestamp()}.png")
        save_image(path, frame)
        self._analyze(frame, staged=True)
        self.status_var.set(f"Captured frame analyzed and saved to {os.path.basename(path)}.")

    def _analyze(self, img, staged=True):
        self._cancel_stage_jobs()
        t0 = time.time()
        method = self.metric.get()
        matrix = blur_matrix.compute_blur_matrix(
            img, grid_cols=self.grid_cols.get(), method=method)
        ratio = blur_matrix.to_blur_ratio(matrix)
        angle = scattering.blur_ratio_to_angle(ratio)
        dt_ms = (time.time() - t0) * 1000

        self.last_input = img
        self.last_matrix, self.last_ratio, self.last_angle = matrix, ratio, angle

        mean_blur = float(ratio.mean())
        mean_angle = float(angle.mean())
        cal = calibration.CalibrationModel(slope=self.cal_slope.get(),
                                            intercept=0.0, particle_k=self.cal_k.get())
        ntu = cal.blur_to_ntu(mean_blur)
        particle = cal.radius_to_particle_size(mean_blur * 14)

        self.metric_vars["Mean Blur Ratio"].set(f"{mean_blur*100:.1f}%")
        self.metric_vars["Mean Angle °"].set(f"{mean_angle:.1f}")
        self.metric_vars["Est. Turbidity NTU"].set(f"{ntu:.1f}")
        self.metric_vars["Est. Particle µm"].set(f"{particle:.1f}")
        self.metric_vars["Compute ms"].set(f"{dt_ms:.1f}")

        self._write_breakdown(img, matrix, ratio, angle, dt_ms, ntu, particle)
        self._reveal_stages(staged and self.animate.get())

    def _cancel_stage_jobs(self):
        for j in self._stage_jobs:
            try:
                self.root.after_cancel(j)
            except Exception:
                pass
        self._stage_jobs = []

    def _show_raw(self):
        m = self.last_matrix
        info = METRIC_INFO[self.metric.get()]
        self.card_raw.set_title(info[0], info[1])
        self.card_raw.view.set_matrix(m)
        u = info[2]
        self.card_raw.colorbar.set_range(f"{fmt_compact(m.min())}{u}", f"{fmt_compact(m.max())}{u}")
        self.card_raw.set_footer(f"step 1/3 · {m.shape[0]}×{m.shape[1]} kernels · "
                                 f"mean {fmt_compact(m.mean())}{u}")
        self.status_var.set("Step 1/3 — raw metric values computed per kernel.")

    def _show_ratio(self):
        r = self.last_ratio
        self.card_ratio.view.set_matrix(r, 0.0, 1.0)
        self.card_ratio.colorbar.set_range("0 sharp", "1 blurry")
        self.card_ratio.set_footer(f"step 2/3 · normalised · mean {r.mean():.3f}")
        self.status_var.set("Step 2/3 — raw values normalised into the blur matrix.")

    def _show_angle(self):
        a = self.last_angle
        self.card_angle.view.set_matrix(a, 0.0, 45.0)
        self.card_angle.colorbar.set_range("0°", "45°")
        self.card_angle.set_footer(f"step 3/3 · atan(ratio) · mean {a.mean():.2f}°")
        self.status_var.set("Step 3/3 — scattering-angle matrix ready.")

    def _reveal_stages(self, animate):
        self._show_raw()
        if not animate:
            self._show_ratio()
            self._show_angle()
            self.status_var.set("Analysis complete.")
            return
        self.card_ratio.view.clear("computing…")
        self.card_ratio.colorbar.set_range("", "")
        self.card_ratio.set_footer("")
        self.card_angle.view.clear("waiting…")
        self.card_angle.colorbar.set_range("", "")
        self.card_angle.set_footer("")
        d = 550
        self._stage_jobs = [
            self.root.after(d, self._show_ratio),
            self.root.after(d, lambda: self.card_angle.view.clear("computing…")),
            self.root.after(2 * d, self._show_angle),
            self.root.after(2 * d + 50, lambda: self.status_var.set("Analysis complete.")),
        ]

    def _log_write(self, chunks):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        for text, tag in chunks:
            self.log.insert("end", text, tag or ())
        self.log.configure(state="disabled")

    def _write_breakdown(self, img, m, ratio, angle, dt_ms, ntu, particle):
        method = self.metric.get()
        title, formula, u = METRIC_INFO[method]
        rows, cols = m.shape
        rmin = np.unravel_index(m.argmin(), m.shape)
        rmax = np.unravel_index(m.argmax(), m.shape)
        h, w = img.shape[:2]
        c = []
        c += [("INPUT   ", "step"),
              (f"{w}×{h} px · grid {rows}×{cols} = {rows*cols} kernels · "
               f"method: {METRIC_LONG[method]}\n", None)]
        c += [("STEP 1  ", "step"), (f"raw {METRIC_LONG[method]} per kernel  ", None),
              (f"[{formula}]\n", "dim")]
        c += [("        ", None),
              (f"min {m.min():.3f}{u}  max {m.max():.3f}{u}  mean {m.mean():.3f}{u}\n", "val")]
        c += [("        ", None),
              (f"sharpest kernel r{rmax[0]} c{rmax[1]} · blurriest kernel r{rmin[0]} c{rmin[1]}\n", "dim")]
        c += [("STEP 2  ", "step"),
              ("blur ratio = 1 − (v − min) / (max − min)   ", None),
              ("[within-frame normalisation]\n", "dim")]
        c += [("        ", None),
              (f"mean {ratio.mean():.4f}  std {ratio.std():.4f}  "
               f"(0 = sharpest, 1 = blurriest)\n", "val")]
        if m.max() == m.min():
            c += [("        ⚠ all kernels identical — ratio is not meaningful for a "
                   "uniform frame\n", "warn")]
        c += [("STEP 3  ", "step"), ("scattering angle θ = atan(blur ratio)\n", None)]
        c += [("        ", None),
              (f"mean {angle.mean():.3f}°  min {angle.min():.3f}°  max {angle.max():.3f}°  "
               f"std {angle.std():.3f}°\n", "val")]
        c += [("RESULT  ", "res"),
              (f"turbidity {ntu:.1f} NTU · particle {particle:.1f} µm · "
               f"{dt_ms:.1f} ms  (placeholder calibration)\n", "res")]
        self._log_write(c)

    def _on_hover(self, cell):
        for card in (self.card_raw, self.card_ratio, self.card_angle):
            card.view.highlight(cell)
        if cell is None or self.last_matrix is None:
            self.hover_var.set("Hover a kernel in any matrix to inspect it")
            return
        r, c = cell
        if r >= self.last_matrix.shape[0] or c >= self.last_matrix.shape[1]:
            return
        self.hover_var.set(
            f"kernel r{r} c{c}  │  raw {self.last_matrix[r, c]:.3f}  │  "
            f"ratio {self.last_ratio[r, c]:.4f}  │  θ {self.last_angle[r, c]:.3f}°")

    def _open_popup(self, which):
        if self.last_matrix is None:
            return
        cfg = {
            "raw": (METRIC_INFO[self.metric.get()][0], self.last_matrix, None, None,
                    cv2.COLORMAP_VIRIDIS, lambda v: f"{v:.2f}"),
            "ratio": ("Blur Matrix", self.last_ratio, 0.0, 1.0,
                      cv2.COLORMAP_JET, lambda v: f"{v:.3f}"),
            "angle": ("Scattering Angle Matrix (°)", self.last_angle, 0.0, 45.0,
                      cv2.COLORMAP_INFERNO, lambda v: f"{v:.2f}"),
        }[which]
        title, mat, lo, hi, cmap, fmt = cfg
        top = tk.Toplevel(self.root)
        top.title(title)
        top.configure(bg=BG)
        top.geometry("1000x760")
        tk.Label(top, text=title, bg=BG, fg=TEXT,
                 font=(FONT_UI, 12, "bold")).pack(anchor="w", padx=16, pady=(12, 4))
        mv = MatrixView(top, cmap=cmap, fmt=fmt, interactive=False)
        mv.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        top.update_idletasks()
        mv.set_matrix(mat, lo, hi)

    def _export_csv(self):
        if self.last_matrix is None:
            self.status_var.set("Run an analysis first.")
            return
        path = os.path.join(RESULTS_DIR, f"blur_matrix_{timestamp()}.csv")
        rows, cols = self.last_matrix.shape
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row", "col", f"raw_{self.metric.get()}", "blur_ratio",
                        "scattering_angle_deg"])
            for r in range(rows):
                for c in range(cols):
                    w.writerow([r, c, f"{self.last_matrix[r,c]:.4f}",
                                f"{self.last_ratio[r,c]:.4f}",
                                f"{self.last_angle[r,c]:.2f}"])
        self.status_var.set(f"Exported {os.path.basename(path)}")

    def _save_snapshot(self):
        if self.last_input is None:
            self.status_var.set("Run an analysis first.")
            return
        ts = timestamp()
        save_image(os.path.join(RESULTS_DIR, f"input_{ts}.png"), self.last_input)
        m = self.last_matrix
        raw_norm = (m - m.min()) / ((m.max() - m.min()) or 1.0)
        save_image(os.path.join(RESULTS_DIR, f"raw_{self.metric.get()}_{ts}.png"),
                   blur_matrix.matrix_to_heatmap(raw_norm, CAM_W, CAM_H, annotate=False))
        save_image(os.path.join(RESULTS_DIR, f"blur_heatmap_{ts}.png"),
                   blur_matrix.matrix_to_heatmap(self.last_ratio, CAM_W, CAM_H, annotate=True))
        save_image(os.path.join(RESULTS_DIR, f"angle_heatmap_{ts}.png"),
                   blur_matrix.matrix_to_heatmap(self.last_angle / 45.0, CAM_W, CAM_H, annotate=True))
        self.status_var.set(f"Snapshot saved to results/ ({ts}).")

    def on_close(self):
        self._cancel_stage_jobs()
        self.cam_active = False
        if self.cam is not None:
            self.cam.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = TurbidityApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()