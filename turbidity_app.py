import datetime
import os
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

import alignment
import blur
import blur_matrix
import data_manager
import preprocessing
import regression_engine
import ui_widgets as W
from camera import Camera

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
STD_W, STD_H = 480, 360   # standard analysis size used everywhere

class TurbidityApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Turbidity & Particle Size Analysis System -- Honeywell Reference Project")
        self.configure(bg=W.BG)
        self.geometry("1440x920")
        self.minsize(1280, 800)

        data_manager.init_storage()

        self.cam = None
        self.is_camera_running = False
        self.ref_img = None
        self.ref_meta = None
        self.captured_img = None
        self.proc_res = None
        self.active_exp_id = data_manager.get_next_experiment_id()

        self.v_cam_idx = tk.StringVar(value="0")
        self.v_navg = tk.StringVar(value="5")
        self.v_exp_id = tk.StringVar(value=self.active_exp_id)
        self.v_algo = tk.StringVar(value="All Algorithms")
        self.v_grid_rows = tk.StringVar(value="12")
        self.v_grid_cols = tk.StringVar(value="16")
        self.v_mode = tk.StringVar(value="ratio")

        self.v_fluid = tk.StringVar(value="Water")
        self.v_material = tk.StringVar(value="Silica")
        self.v_size = tk.StringVar(value="10.0")
        self.v_size_unit = tk.StringVar(value="µm")
        self.v_conc = tk.StringVar(value="50.0")
        self.v_conc_unit = tk.StringVar(value="mg/L")
        self.v_ntu = tk.StringVar(value="")
        self.v_sample_id = tk.StringVar(value="Sample_001")

        self.v_status = tk.StringVar(value="Ready. Load reference image and connect camera.")
        self.v_ref_info = tk.StringVar(value="Reference: None loaded")
        self.v_pred_ntu = tk.StringVar(value="--")
        self.v_pred_size = tk.StringVar(value="--")
        self.v_blur_ratio = tk.StringVar(value="--")

        self._build_ui()

        self._try_load_default_reference()

        self.after(100, self._camera_tick)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):

        self._build_top_header()

        body = tk.Frame(self, bg=W.BG)
        body.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        self._build_sidebar(body)

        self._build_main_display(body)

    def _build_top_header(self):
        hdr = tk.Frame(self, bg=W.PANEL, height=52, padx=14, pady=6)
        hdr.pack(fill="x", padx=8, pady=(8, 4))
        hdr.pack_propagate(False)

        title_box = tk.Frame(hdr, bg=W.PANEL)
        title_box.pack(side="left", fill="y")
        tk.Label(title_box, text="TURBIDITY & PARTICLE ANALYZER", bg=W.PANEL, fg=W.SHARP,
                 font=(W.FONT_UI, 12, "bold")).pack(anchor="w")
        tk.Label(title_box, textvariable=self.v_status, bg=W.PANEL, fg=W.DIM,
                 font=(W.FONT_MONO, 8)).pack(anchor="w")

        pred_box = tk.Frame(hdr, bg=W.PANEL)
        pred_box.pack(side="right", fill="y")

        t0 = tk.Frame(pred_box, bg=W.CARD, padx=12, pady=2, highlightbackground=W.LINE, highlightthickness=1)
        t0.pack(side="left", padx=6)
        tk.Label(t0, text="BLUR RATIO", bg=W.CARD, fg=W.DIM, font=(W.FONT_UI, 8, "bold")).pack()
        tk.Label(t0, textvariable=self.v_blur_ratio, bg=W.CARD, fg=W.VIOLET, font=(W.FONT_MONO, 12, "bold")).pack()

        t1 = tk.Frame(pred_box, bg=W.CARD, padx=12, pady=2, highlightbackground=W.LINE, highlightthickness=1)
        t1.pack(side="left", padx=6)
        tk.Label(t1, text="PREDICTED NTU", bg=W.CARD, fg=W.DIM, font=(W.FONT_UI, 8, "bold")).pack()
        tk.Label(t1, textvariable=self.v_pred_ntu, bg=W.CARD, fg=W.AMBER, font=(W.FONT_MONO, 12, "bold")).pack()

        t2 = tk.Frame(pred_box, bg=W.CARD, padx=12, pady=2, highlightbackground=W.LINE, highlightthickness=1)
        t2.pack(side="left", padx=6)
        tk.Label(t2, text="PREDICTED SIZE", bg=W.CARD, fg=W.DIM, font=(W.FONT_UI, 8, "bold")).pack()
        tk.Label(t2, textvariable=self.v_pred_size, bg=W.CARD, fg=W.SHARP, font=(W.FONT_MONO, 12, "bold")).pack()

    def _build_sidebar(self, parent):
        outer = tk.Frame(parent, bg=W.PANEL, width=340)
        outer.pack(side="left", fill="y", padx=(0, 6))
        outer.pack_propagate(False)

        canvas = tk.Canvas(outer, bg=W.PANEL, highlightthickness=0)
        sb = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.sidebar_canvas = canvas
        self.sidebar_canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=W.PANEL, padx=8, pady=4)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))

        def _wheel(e):
            step = -e.delta if sys.platform == "darwin" else -int(e.delta / 120)
            canvas.yview_scroll(step, "units")
        outer.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _wheel))
        outer.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        self._build_ref_section(inner)

        self._build_camera_section(inner)

        self._build_algo_section(inner)

        self._build_gt_section(inner)

        self._build_dataset_section(inner)

        self.after(50, lambda: self.sidebar_canvas.yview_moveto(0.0))
        self.after(200, lambda: self.sidebar_canvas.yview_moveto(0.0))

    def _build_ref_section(self, p):
        self._sec_hdr(p, "1. REFERENCE IMAGE (BASELINE)")
        tk.Label(p, text="Clear baseline pattern without fluid/particles", bg=W.PANEL, fg=W.DIM,
                 font=(W.FONT_UI, 8)).pack(anchor="w", pady=(0, 4))

        bf = tk.Frame(p, bg=W.PANEL)
        bf.pack(fill="x", pady=2)
        tk.Button(bf, text="Upload Reference", command=self.upload_reference_dialog,
                  bg=W.BLUE, fg="#06110f", font=(W.FONT_UI, 9, "bold"), relief="flat", pady=3)\
            .pack(side="left", fill="x", expand=True)

        self.ref_info_lbl = tk.Label(p, textvariable=self.v_ref_info, bg=W.CARD, fg=W.TEXT,
                                     font=(W.FONT_MONO, 8), justify="left", anchor="w",
                                     padx=8, pady=6, relief="flat", highlightbackground=W.LINE,
                                     highlightthickness=1)
        self.ref_info_lbl.pack(fill="x", pady=4)

    def _build_camera_section(self, p):
        self._sec_hdr(p, "2. LIVE CAMERA / IMAGER")

        ctrl_f = tk.Frame(p, bg=W.PANEL)
        ctrl_f.pack(fill="x", pady=2)
        tk.Label(ctrl_f, text="Cam Index:", bg=W.PANEL, fg=W.DIM, font=(W.FONT_UI, 8)).pack(side="left")
        tk.Entry(ctrl_f, textvariable=self.v_cam_idx, width=4, bg=W.FIELD, fg=W.TEXT, relief="flat",
                 insertbackground=W.TEXT).pack(side="left", padx=4)
        tk.Label(ctrl_f, text="Avg Frames:", bg=W.PANEL, fg=W.DIM, font=(W.FONT_UI, 8)).pack(side="left", padx=(6, 0))
        tk.Entry(ctrl_f, textvariable=self.v_navg, width=4, bg=W.FIELD, fg=W.TEXT, relief="flat",
                 insertbackground=W.TEXT).pack(side="left", padx=4)

        b_row1 = tk.Frame(p, bg=W.PANEL)
        b_row1.pack(fill="x", pady=3)
        self.btn_cam_toggle = tk.Button(b_row1, text="Start Camera", command=self.toggle_camera,
                                        bg=W.SHARP, fg="#06110f", font=(W.FONT_UI, 9, "bold"),
                                        relief="flat", pady=4)
        self.btn_cam_toggle.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.btn_cap = tk.Button(b_row1, text="Capture", command=self.capture_frame,
                                 bg=W.AMBER, fg="#06110f", font=(W.FONT_UI, 9, "bold"),
                                 relief="flat", pady=4)
        self.btn_cap.pack(side="left", fill="x", expand=True, padx=(2, 0))

        b_row2 = tk.Frame(p, bg=W.PANEL)
        b_row2.pack(fill="x", pady=2)
        tk.Button(b_row2, text="Retake (Resume)", command=self.retake_frame,
                  bg=W.FIELD, fg="#06110f", font=(W.FONT_UI, 9, "bold"), relief="flat", pady=3)\
            .pack(side="left", fill="x", expand=True, padx=(0, 2))
        tk.Button(b_row2, text="Load Sample File", command=self.load_sample_file_dialog,
                  bg=W.FIELD, fg="#06110f", font=(W.FONT_UI, 9, "bold"), relief="flat", pady=3)\
            .pack(side="left", fill="x", expand=True, padx=(2, 0))

        tk.Button(p, text="Simulate Particle Scattering (Test)", command=self.simulate_turbid_sample,
                  bg=W.CARD, fg="#06110f", font=(W.FONT_UI, 8, "bold"), relief="flat", pady=3)\
            .pack(fill="x", pady=2)

    def _build_algo_section(self, p):
        self._sec_hdr(p, "3. ALGORITHM SELECTION")
        algos = ["Variance of Laplacian", "Tenengrad", "FFT High-Frequency Energy", "All Algorithms"]
        for a in algos:
            tk.Radiobutton(p, text=a, variable=self.v_algo, value=a, command=self._reprocess_current,
                           bg=W.PANEL, fg=W.TEXT, selectcolor=W.FIELD, activebackground=W.PANEL,
                           font=(W.FONT_UI, 9)).pack(anchor="w", pady=1)

        grid_f = tk.Frame(p, bg=W.PANEL)
        grid_f.pack(fill="x", pady=(6, 2))
        tk.Label(grid_f, text="Kernel Grid:", bg=W.PANEL, fg=W.DIM, font=(W.FONT_UI, 8, "bold")).pack(side="left")
        tk.Label(grid_f, text="Rows (m):", bg=W.PANEL, fg=W.DIM, font=(W.FONT_UI, 8)).pack(side="left", padx=(6, 2))
        e_r = tk.Entry(grid_f, textvariable=self.v_grid_rows, width=4, bg=W.FIELD, fg=W.TEXT, relief="flat",
                       insertbackground=W.TEXT)
        e_r.pack(side="left")
        e_r.bind("<Return>", lambda e: self._reprocess_current())
        tk.Label(grid_f, text="Cols (n):", bg=W.PANEL, fg=W.DIM, font=(W.FONT_UI, 8)).pack(side="left", padx=(6, 2))
        e_c = tk.Entry(grid_f, textvariable=self.v_grid_cols, width=4, bg=W.FIELD, fg=W.TEXT, relief="flat",
                       insertbackground=W.TEXT)
        e_c.pack(side="left")
        e_c.bind("<Return>", lambda e: self._reprocess_current())

    def _build_gt_section(self, p):
        self._sec_hdr(p, "4. GROUND TRUTH / EXPERIMENT INFO")
        tk.Label(p, text="Target values for future ML regression", bg=W.PANEL, fg=W.SHARP,
                 font=(W.FONT_UI, 8)).pack(anchor="w", pady=(0, 4))

        self._lbl_entry(p, "Experiment ID:", self.v_exp_id)
        self._lbl_entry(p, "Sample ID:", self.v_sample_id)
        self._lbl_entry(p, "Fluid Type:", self.v_fluid)
        self._lbl_entry(p, "Particle Material:", self.v_material)

        s_f = tk.Frame(p, bg=W.PANEL)
        s_f.pack(fill="x", pady=1)
        tk.Label(s_f, text="Particle Size:", bg=W.PANEL, fg=W.DIM, font=(W.FONT_UI, 8), width=13, anchor="w").pack(side="left")
        tk.Entry(s_f, textvariable=self.v_size, width=8, bg=W.FIELD, fg=W.TEXT, relief="flat", insertbackground=W.TEXT).pack(side="left", fill="x", expand=True)
        tk.Entry(s_f, textvariable=self.v_size_unit, width=4, bg=W.FIELD, fg=W.SHARP, relief="flat", insertbackground=W.TEXT).pack(side="left", padx=(4, 0))

        c_f = tk.Frame(p, bg=W.PANEL)
        c_f.pack(fill="x", pady=1)
        tk.Label(c_f, text="Concentration:", bg=W.PANEL, fg=W.DIM, font=(W.FONT_UI, 8), width=13, anchor="w").pack(side="left")
        tk.Entry(c_f, textvariable=self.v_conc, width=8, bg=W.FIELD, fg=W.TEXT, relief="flat", insertbackground=W.TEXT).pack(side="left", fill="x", expand=True)
        tk.Entry(c_f, textvariable=self.v_conc_unit, width=5, bg=W.FIELD, fg=W.SHARP, relief="flat", insertbackground=W.TEXT).pack(side="left", padx=(4, 0))

        self._lbl_entry(p, "Known NTU:", self.v_ntu)

    def _build_dataset_section(self, p):
        self._sec_hdr(p, "5. DATASET & MACHINE LEARNING")

        tk.Button(p, text="Save Experiment Record", command=self.save_experiment,
                  bg=W.AMBER, fg="#06110f", font=(W.FONT_UI, 10, "bold"), relief="flat", pady=5)\
            .pack(fill="x", pady=3)

        tk.Button(p, text="Export CSV Dataset", command=self.export_csv_dialog,
                  bg=W.FIELD, fg="#06110f", font=(W.FONT_UI, 9, "bold"), relief="flat", pady=3)\
            .pack(fill="x", pady=2)

        tk.Button(p, text="View Experiment History", command=lambda: self.notebook.select(self.tab_history),
                  bg=W.FIELD, fg="#06110f", font=(W.FONT_UI, 9, "bold"), relief="flat", pady=3)\
            .pack(fill="x", pady=2)

        tk.Button(p, text="Regression Models & ML", command=lambda: self.notebook.select(self.tab_regression),
                  bg=W.VIOLET, fg="#06110f", font=(W.FONT_UI, 9, "bold"), relief="flat", pady=4)\
            .pack(fill="x", pady=(4, 8))

    def _build_main_display(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(side="left", fill="both", expand=True)

        self.tab_dash = tk.Frame(self.notebook, bg=W.BG)
        self.notebook.add(self.tab_dash, text="  Dashboard & Matrices  ")
        self._build_dashboard_tab(self.tab_dash)

        self.tab_prep = tk.Frame(self.notebook, bg=W.BG)
        self.notebook.add(self.tab_prep, text="  Preprocessing Pipeline (11 Stages)  ")
        self._build_preprocessing_tab(self.tab_prep)

        self.tab_history = tk.Frame(self.notebook, bg=W.BG)
        self.notebook.add(self.tab_history, text="  Experiment History  ")
        self._build_history_tab(self.tab_history)

        self.tab_regression = tk.Frame(self.notebook, bg=W.BG)
        self.notebook.add(self.tab_regression, text="  Regression & Machine Learning  ")
        self._build_regression_tab(self.tab_regression)

    def _build_dashboard_tab(self, parent):

        top_grid = tk.Frame(parent, bg=W.BG)
        top_grid.pack(fill="both", expand=True, padx=4, pady=4)

        for col in range(4):
            top_grid.columnconfigure(col, weight=1)
        top_grid.rowconfigure(0, weight=1)

        self.card_ref = W.StageCard(top_grid, "R", "Reference Image", "Clean pattern baseline", W.BLUE, kind="image")
        self.card_ref.grid(row=0, column=0, sticky="nsew", padx=3, pady=3)

        self.card_cap = W.StageCard(top_grid, "C", "Captured / Sample Frame", "Live camera or captured frame", W.SHARP, kind="image")
        self.card_cap.grid(row=0, column=1, sticky="nsew", padx=3, pady=3)

        self.card_diff = W.StageCard(top_grid, "D", "Difference Image", "|Aligned - Reference|", W.AMBER, kind="image")
        self.card_diff.grid(row=0, column=2, sticky="nsew", padx=3, pady=3)

        self.card_overlay = W.StageCard(top_grid, "O", "Overlay Comparison", "Cyan: Ref | Red: Aligned Cap", W.VIOLET, kind="image")
        self.card_overlay.grid(row=0, column=3, sticky="nsew", padx=3, pady=3)

        bot_grid = tk.Frame(parent, bg=W.BG)
        bot_grid.pack(fill="both", expand=True, padx=4, pady=4)
        bot_grid.columnconfigure(0, weight=1)
        bot_grid.columnconfigure(1, weight=1)
        bot_grid.rowconfigure(0, weight=1)

        blur_box = tk.Frame(bot_grid, bg=W.CARD, highlightbackground=W.LINE, highlightthickness=1)
        blur_box.grid(row=0, column=0, sticky="nsew", padx=3, pady=3)
        b_head = tk.Frame(blur_box, bg=W.CARD, padx=8, pady=6)
        b_head.pack(fill="x")
        tk.Label(b_head, text="BLUR MATRIX (HEATMAP)", bg=W.CARD, fg=W.SHARP, font=(W.FONT_UI, 10, "bold")).pack(side="left")
        tk.Button(b_head, text="[ Numerical Matrix ]", command=self.show_blur_numerical_matrix,
                  bg=W.FIELD, fg="#06110f", font=(W.FONT_UI, 8, "bold"), relief="flat", padx=6, pady=2).pack(side="right")
        self.blur_view = W.MatrixView(blur_box, cmap=cv2.COLORMAP_JET)
        self.blur_view.pack(fill="both", expand=True, padx=8)
        self.blur_stats_lbl = tk.Label(blur_box, text="Mean: -- | Min: -- | Max: -- | Std: -- | Median: --",
                                       bg=W.CARD, fg=W.DIM, font=(W.FONT_MONO, 8), pady=4)
        self.blur_stats_lbl.pack(fill="x")

        bot_grid.columnconfigure(2, weight=1)
        rat_box = tk.Frame(bot_grid, bg=W.CARD, highlightbackground=W.LINE, highlightthickness=1)
        rat_box.grid(row=0, column=1, sticky="nsew", padx=3, pady=3)
        r_head = tk.Frame(rat_box, bg=W.CARD, padx=8, pady=6)
        r_head.pack(fill="x")
        tk.Label(r_head, text="BLUR RATIO MATRIX (1 - sample/ref)", bg=W.CARD, fg=W.VIOLET, font=(W.FONT_UI, 10, "bold")).pack(side="left")
        tk.Button(r_head, text="[ Numerical Matrix ]", command=self.show_ratio_numerical_matrix,
                  bg=W.FIELD, fg="#06110f", font=(W.FONT_UI, 8, "bold"), relief="flat", padx=6, pady=2).pack(side="right")
        self.ratio_view = W.MatrixView(rat_box, cmap=cv2.COLORMAP_INFERNO, fmt=W.fmt_ratio)
        self.ratio_view.pack(fill="both", expand=True, padx=8)
        self.ratio_stats_lbl = tk.Label(rat_box, text="Mean: -- | Min: -- | Max: -- | Std: -- | Median: --",
                                        bg=W.CARD, fg=W.DIM, font=(W.FONT_MONO, 8), pady=4)
        self.ratio_stats_lbl.pack(fill="x")

        ang_box = tk.Frame(bot_grid, bg=W.CARD, highlightbackground=W.LINE, highlightthickness=1)
        ang_box.grid(row=0, column=2, sticky="nsew", padx=3, pady=3)
        a_head = tk.Frame(ang_box, bg=W.CARD, padx=8, pady=6)
        a_head.pack(fill="x")
        tk.Label(a_head, text="SCATTERING ANGLE MATRIX (HEATMAP)", bg=W.CARD, fg=W.AMBER, font=(W.FONT_UI, 10, "bold")).pack(side="left")
        tk.Button(a_head, text="[ Numerical Matrix ]", command=self.show_angle_numerical_matrix,
                  bg=W.FIELD, fg="#06110f", font=(W.FONT_UI, 8, "bold"), relief="flat", padx=6, pady=2).pack(side="right")
        self.angle_view = W.MatrixView(ang_box, cmap=cv2.COLORMAP_TURBO)
        self.angle_view.pack(fill="both", expand=True, padx=8)
        self.angle_stats_lbl = tk.Label(ang_box, text="Mean: -- | Min: -- | Max: -- | Std: -- | Median: --",
                                        bg=W.CARD, fg=W.DIM, font=(W.FONT_MONO, 8), pady=4)
        self.angle_stats_lbl.pack(fill="x")

        feat_bar = tk.Frame(parent, bg=W.PANEL, padx=12, pady=6)
        feat_bar.pack(fill="x", padx=6, pady=4)
        tk.Label(feat_bar, text="BLUR METRICS (sample vs reference)", bg=W.PANEL, fg=W.AMBER,
                 font=(W.FONT_UI, 9, "bold")).pack(anchor="w")
        tbl = tk.Frame(feat_bar, bg=W.PANEL)
        tbl.pack(fill="x", pady=(4, 2))
        heads = ["Algorithm", "Sample", "Reference", "Blur ratio (global)",
                 "Cell ratio mean", "Cell ratio std", "Cell ratio P90", "Mean angle"]
        for c, h in enumerate(heads):
            tk.Label(tbl, text=h, bg=W.PANEL, fg=W.DIM, font=(W.FONT_UI, 8, "bold"),
                     anchor="w").grid(row=0, column=c, sticky="w", padx=(0, 16))
        self.metric_cells = {}
        for r, (key, nm) in enumerate((("laplacian", "Variance of Laplacian"),
                                       ("tenengrad", "Tenengrad"),
                                       ("fft", "FFT High-Freq (%)")), 1):
            row = []
            for c in range(len(heads)):
                lbl = tk.Label(tbl, text=nm if c == 0 else "--", bg=W.PANEL, fg=W.TEXT,
                               font=(W.FONT_MONO, 9), anchor="w")
                lbl.grid(row=r, column=c, sticky="w", padx=(0, 16))
                row.append(lbl)
            self.metric_cells[key] = row
        self.raw_feat_lbl = tk.Label(feat_bar, text="Alignment: --", bg=W.PANEL, fg=W.TEXT,
                                     font=(W.FONT_MONO, 9), anchor="w", justify="left")
        self.raw_feat_lbl.pack(anchor="w", pady=(4, 0))

    def _build_preprocessing_tab(self, parent):
        top_ctrl = tk.Frame(parent, bg=W.PANEL, padx=12, pady=8)
        top_ctrl.pack(fill="x", padx=4, pady=(4, 2))
        tk.Label(top_ctrl, text="Complete 11-Stage Preprocessing Pipeline (Section 5)", bg=W.PANEL, fg=W.SHARP,
                 font=(W.FONT_UI, 11, "bold")).pack(side="left")
        tk.Label(top_ctrl, text="Click any stage thumbnail to inspect in full size below", bg=W.PANEL, fg=W.DIM,
                 font=(W.FONT_UI, 9)).pack(side="right")

        outer = tk.Frame(parent, bg=W.BG)
        outer.pack(fill="both", expand=True, padx=4, pady=4)

        canvas = tk.Canvas(outer, bg=W.BG, highlightthickness=0)
        sb = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.prep_grid = tk.Frame(canvas, bg=W.BG)
        win = canvas.create_window((0, 0), window=self.prep_grid, anchor="nw")
        self.prep_grid.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))

        self.stage_cards = []
        for i, name in enumerate(preprocessing.STAGE_NAMES):
            row = i // 4
            col = i % 4
            card = W.StageCard(self.prep_grid, str(i + 1), name, "Waiting for image", W.BLUE, kind="image")
            card.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
            self.prep_grid.columnconfigure(col, weight=1)
            self.stage_cards.append(card)

    def _build_history_tab(self, parent):
        ctrl = tk.Frame(parent, bg=W.PANEL, padx=12, pady=8)
        ctrl.pack(fill="x", padx=4, pady=(4, 2))
        tk.Label(ctrl, text="Experiment History (Logged to data/experiments.csv)", bg=W.PANEL, fg=W.SHARP,
                 font=(W.FONT_UI, 11, "bold")).pack(side="left")
        tk.Button(ctrl, text="Refresh History", command=self.load_history_table, bg=W.FIELD, fg="#06110f",
                  font=(W.FONT_UI, 9, "bold"), relief="flat", padx=8, pady=3).pack(side="right", padx=4)

        tk.Button(ctrl, text="Load Selected Experiment", command=self.load_selected_history_item, bg=W.BLUE,
                  fg="#06110f", font=(W.FONT_UI, 9, "bold"), relief="flat", padx=8, pady=3).pack(side="right", padx=4)

        table_frame = tk.Frame(parent, bg=W.CARD)
        table_frame.pack(fill="both", expand=True, padx=4, pady=4)

        cols = [
            ("exp_id", "Exp ID", 70),
            ("timestamp", "Date / Time", 130),
            ("ref", "Reference Image", 140),
            ("cap", "Captured Image", 100),
            ("algo", "Algorithm", 130),
            ("fluid", "Fluid", 80),
            ("material", "Material", 80),
            ("size", "Size (µm)", 70),
            ("conc", "Conc (mg/L)", 80),
            ("ntu", "NTU", 70),
            ("blur", "Mean Blur", 85),
            ("angle", "Mean Angle", 85),
            ("status", "Status", 70),
        ]

        self.hist_tree = ttk.Treeview(table_frame, columns=[c[0] for c in cols], show="headings", height=18)
        for cid, text, w in cols:
            self.hist_tree.heading(cid, text=text)
            self.hist_tree.column(cid, width=w, anchor="center")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.hist_tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.hist_tree.xview)
        self.hist_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.hist_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        self.hist_tree.bind("<Double-Button-1>", lambda e: self.load_selected_history_item())

    def _build_regression_tab(self, parent):

        hdr = tk.Frame(parent, bg=W.PANEL, padx=14, pady=10)
        hdr.pack(fill="x", padx=6, pady=(6, 4))
        tk.Label(hdr, text="FUTURE MACHINE LEARNING & REGRESSION ARCHITECTURE", bg=W.PANEL, fg=W.SHARP,
                 font=(W.FONT_UI, 12, "bold")).pack(anchor="w")
        tk.Label(hdr, text="Learns relationships between Image Features -> NTU & Image Features -> Particle Size",
                 bg=W.PANEL, fg=W.DIM, font=(W.FONT_UI, 9)).pack(anchor="w", pady=(2, 6))

        self.reg_stats_card = tk.Label(hdr, text="Dataset Stats: Total: 0 | NTU: 0 | Particle Size: 0 | Complete: 0 | Incomplete: 0",
                                       bg=W.CARD, fg=W.AMBER, font=(W.FONT_MONO, 9), padx=10, pady=6,
                                       highlightbackground=W.LINE, highlightthickness=1)
        self.reg_stats_card.pack(fill="x", pady=4)

        b_bar = tk.Frame(hdr, bg=W.PANEL)
        b_bar.pack(fill="x", pady=4)
        tk.Button(b_bar, text="Export Regression Dataset (CSV)", command=self.export_regression_dataset_action,
                  bg=W.BLUE, fg="#06110f", font=(W.FONT_UI, 9, "bold"), relief="flat", padx=10, pady=4).pack(side="left", padx=(0, 6))
        tk.Button(b_bar, text="Train Target 1: NTU Model", command=lambda: self.train_model_action("known_NTU"),
                  bg=W.VIOLET, fg="#06110f", font=(W.FONT_UI, 9, "bold"), relief="flat", padx=10, pady=4).pack(side="left", padx=6)
        tk.Button(b_bar, text="Train Target 2: Particle Size Model", command=lambda: self.train_model_action("particle_size"),
                  bg=W.SHARP, fg="#06110f", font=(W.FONT_UI, 9, "bold"), relief="flat", padx=10, pady=4).pack(side="left", padx=6)
        tk.Button(b_bar, text="Predict on Current Frame", command=self.predict_current_frame,
                  bg=W.AMBER, fg="#06110f", font=(W.FONT_UI, 9, "bold"), relief="flat", padx=10, pady=4).pack(side="left", padx=6)

        body_split = tk.Frame(parent, bg=W.BG)
        body_split.pack(fill="both", expand=True, padx=6, pady=4)
        body_split.columnconfigure(0, weight=1)
        body_split.columnconfigure(1, weight=1)
        body_split.rowconfigure(0, weight=1)

        ntu_box = tk.Frame(body_split, bg=W.CARD, highlightbackground=W.LINE, highlightthickness=1)
        ntu_box.grid(row=0, column=0, sticky="nsew", padx=3, pady=3)
        tk.Label(ntu_box, text="Model 1: NTU Regression (Actual vs Predicted)", bg=W.CARD, fg=W.AMBER,
                 font=(W.FONT_UI, 10, "bold"), pady=6).pack()
        self.plot_ntu_view = W.ImageView(ntu_box, width=460, height=280)
        self.plot_ntu_view.pack(fill="both", expand=True, padx=8, pady=4)
        self.lbl_ntu_metrics = tk.Label(ntu_box, text="R2: --  |  RMSE: --", bg=W.CARD, fg=W.TEXT, font=(W.FONT_MONO, 9), pady=4)
        self.lbl_ntu_metrics.pack(fill="x")

        size_box = tk.Frame(body_split, bg=W.CARD, highlightbackground=W.LINE, highlightthickness=1)
        size_box.grid(row=0, column=1, sticky="nsew", padx=3, pady=3)
        tk.Label(size_box, text="Model 2: Particle Size Regression (Actual vs Predicted)", bg=W.CARD, fg=W.SHARP,
                 font=(W.FONT_UI, 10, "bold"), pady=6).pack()
        self.plot_size_view = W.ImageView(size_box, width=460, height=280)
        self.plot_size_view.pack(fill="both", expand=True, padx=8, pady=4)
        self.lbl_size_metrics = tk.Label(size_box, text="R2: --  |  RMSE: --", bg=W.CARD, fg=W.TEXT, font=(W.FONT_MONO, 9), pady=4)
        self.lbl_size_metrics.pack(fill="x")

    def _sec_hdr(self, p, text):
        tk.Label(p, text=text, bg=W.PANEL, fg=W.AMBER, font=(W.FONT_UI, 9, "bold")).pack(anchor="w", pady=(10, 2))

    def _lbl_entry(self, p, label_text, var):
        f = tk.Frame(p, bg=W.PANEL)
        f.pack(fill="x", pady=1)
        tk.Label(f, text=label_text, bg=W.PANEL, fg=W.DIM, font=(W.FONT_UI, 8), width=13, anchor="w").pack(side="left")
        e = tk.Entry(f, textvariable=var, bg=W.FIELD, fg=W.TEXT, relief="flat", insertbackground=W.TEXT)
        e.pack(side="left", fill="x", expand=True)
        return e

    def _try_load_default_reference(self):
        default_ref = os.path.join(ASSETS_DIR, "checkerboard.png")
        if os.path.exists(default_ref):
            self.set_reference_from_path(default_ref, warn=False)

    def upload_reference_dialog(self):
        p = filedialog.askopenfilename(
            title="Upload Reference Pattern Image",
            initialdir=ASSETS_DIR if os.path.isdir(ASSETS_DIR) else BASE_DIR,
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")]
        )
        if p:
            self.set_reference_from_path(p)

    def _confirm_upload(self, img, path, role, other_size=None):
        """Warn if an uploaded image is not the standard size/shape/format.
        Returns True to continue loading, False if the user cancels."""
        h, w = img.shape[:2]
        msgs = []
        if (w, h) != (STD_W, STD_H):
            how = "enlarged" if w * h < STD_W * STD_H else "shrunk"
            msgs.append(f"Size is {w}x{h}, not {STD_W}x{STD_H}. It will be {how}, "
                        f"which slightly changes the sharpness values.")
            if abs(w / h - STD_W / STD_H) > 0.02:
                msgs.append(f"Shape is different (aspect {w / h:.2f} vs {STD_W / STD_H:.2f}), "
                            f"so the picture will be STRETCHED and distorted.")
        if path.lower().endswith((".jpg", ".jpeg", ".jpe")):
            msgs.append("JPEG is compressed and softens edges. PNG is preferred.")
        if other_size is not None and tuple(other_size) != (w, h):
            msgs.append(f"The reference is {other_size[0]}x{other_size[1]} but this {role} is "
                        f"{w}x{h}. Use images from the same source and size.")
        if not msgs:
            return True
        self.v_status.set(f"Warning: {role} is not {STD_W}x{STD_H} ({w}x{h}).")
        text = "\n\n".join(f"- {m}" for m in msgs)
        return messagebox.askokcancel(
            "Image size warning",
            f"{role.capitalize()}: {os.path.basename(path)}\n\n{text}\n\n"
            f"For reliable results use live camera captures ({STD_W}x{STD_H}).\n\nLoad anyway?")

    def set_reference_from_path(self, path, warn=True):
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image")
            if warn and not self._confirm_upload(img, path, "reference"):
                return

            meta = data_manager.save_reference_image(img, os.path.basename(path))
            self.ref_img = img
            self.ref_meta = meta

            info_str = f"File: {meta['filename']}\nSize: {meta['width']} x {meta['height']} px\nFormat: {meta['format']}"
            self.v_ref_info.set(info_str)
            self.card_ref.view.set_image(img)
            self.card_ref.set_title("Reference Image", meta["filename"])
            self.v_status.set(f"Reference pattern loaded: {meta['filename']}")

            if self.captured_img is not None:
                self._run_pipeline()
        except Exception as e:
            messagebox.showerror("Error Loading Reference", f"Failed to load reference image:\n{e}")

    def toggle_camera(self):
        if self.is_camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        try:
            idx = int(self.v_cam_idx.get().strip() or 0)
        except ValueError:
            idx = 0
        try:
            if self.cam is not None:
                self.cam.release()
            self.cam = Camera(index=idx, width=480, height=360)
            self.is_camera_running = True
            self.btn_cam_toggle.config(text="Stop Camera", bg=W.RED, fg="#ffffff")
            self.v_status.set(f"Camera connected (Index {idx}). Live preview active.")
        except Exception as e:
            self.cam = None
            self.is_camera_running = False
            self.v_status.set(f"Camera connection failed: {e}. You can load an image file.")
            messagebox.showwarning("Camera Warning", f"Could not connect to camera index {idx}.\n{e}\n\nYou can use 'Load Sample File' or 'Simulate Particle Scattering'.")

    def stop_camera(self):
        if self.cam is not None:
            self.cam.release()
            self.cam = None
        self.is_camera_running = False
        self.btn_cam_toggle.config(text="Start Camera", bg=W.SHARP, fg="#06110f")
        self.v_status.set("Camera stopped.")

    def _camera_tick(self):
        try:
            if self.is_camera_running and self.cam is not None and self.captured_img is None:
                frame = self.cam.read_frame()
                if frame is not None:
                    self.card_cap.view.set_image(frame)
        except Exception:
            pass
        self.after(50, self._camera_tick)

    def capture_frame(self):
        if self.is_camera_running and self.cam is not None:
            try:
                n = max(1, int(self.v_navg.get().strip() or 1))
            except ValueError:
                n = 1

            acc = None
            for _ in range(n):
                fr = self.cam.read_frame().astype(np.float32)
                acc = fr if acc is None else acc + fr
            frame = np.clip(acc / n + 0.5, 0, 255).astype(np.uint8)
            self.captured_img = frame
            self.active_exp_id = data_manager.get_next_experiment_id()
            self.v_exp_id.set(self.active_exp_id)
            self.v_status.set(f"Frame captured. ID: {self.active_exp_id}. Processing pipeline...")
            self._run_pipeline()
        elif self.captured_img is not None:
            self._run_pipeline()
        else:
            messagebox.showinfo("Capture", "Camera is not running. Please start camera, load a sample file, or simulate.")

    def retake_frame(self):
        self.captured_img = None
        self.proc_res = None
        self.active_exp_id = data_manager.get_next_experiment_id()
        self.v_exp_id.set(self.active_exp_id)
        self.v_status.set("Capture cleared. Resuming live preview.")
        self.card_cap.set_title("Captured / Sample Frame", "Live camera active")

    def load_sample_file_dialog(self):
        p = filedialog.askopenfilename(
            title="Load Sample Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")]
        )
        if p:
            data = np.fromfile(p, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                other = (self.ref_meta["width"], self.ref_meta["height"]) if self.ref_meta else None
                if not self._confirm_upload(img, p, "sample", other):
                    return
                self.captured_img = img
                self.active_exp_id = data_manager.get_next_experiment_id()
                self.v_exp_id.set(self.active_exp_id)
                self.v_status.set(f"Sample loaded: {os.path.basename(p)}. Processing pipeline...")
                self._run_pipeline()

    def simulate_turbid_sample(self):

        if self.ref_img is None:
            messagebox.showwarning("No Reference", "Please load a reference image first.")
            return

        try:
            ntu_val = float(self.v_ntu.get().strip() or 25.0)
        except ValueError:
            ntu_val = 25.0

        ksize = int(max(3, min(31, round(ntu_val / 5.0) * 2 + 1)))
        sigma = max(0.5, ntu_val / 10.0)

        blurred = cv2.GaussianBlur(self.ref_img, (ksize, ksize), sigma)

        noise = np.random.normal(0, 2.0, blurred.shape).astype(np.float32)
        sim_sample = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        self.captured_img = sim_sample
        self.active_exp_id = data_manager.get_next_experiment_id()
        self.v_exp_id.set(self.active_exp_id)
        self.v_status.set(f"Simulated scattering generated (ksize={ksize}, sigma={sigma:.1f}).")
        self._run_pipeline()

    def _reprocess_current(self):
        if self.captured_img is not None and self.ref_img is not None:
            self._run_pipeline()

    def _run_pipeline(self):
        if self.captured_img is None:
            return
        if self.ref_img is None:

            self.ref_img = np.full((360, 480, 3), 128, dtype=np.uint8)
            self.ref_meta = {"filename": "auto_neutral.png", "width": 480, "height": 360, "format": "PNG"}

        try:
            rows = int(self.v_grid_rows.get().strip() or 12)
            cols = int(self.v_grid_cols.get().strip() or 16)
        except ValueError:
            rows, cols = 12, 16

        algo_key = self.v_algo.get().lower()
        if "laplacian" in algo_key:
            selected_method = "laplacian"
        elif "tenengrad" in algo_key:
            selected_method = "tenengrad"
        elif "fft" in algo_key:
            selected_method = "fft"
        else:
            selected_method = "all"

        res = preprocessing.run_preprocessing_pipeline(
            self.ref_img,
            self.captured_img,
            grid_cols=cols,
            grid_rows=rows,
            method=selected_method,
            target_size=(480, 360)
        )
        self.proc_res = res

        self.card_ref.view.set_image(res["ref_bgr"])
        self.card_cap.view.set_image(res["aligned_bgr"])
        self.card_diff.view.set_image(res["diff_color"])
        self.card_overlay.view.set_image(res["overlay_bgr"])

        self.card_cap.set_title("Aligned Capture", f"Shift: dx={res['shift_dx']:.1f}, dy={res['shift_dy']:.1f}px")
        self.card_diff.set_title("Difference Image", f"Mean diff: {res['diff_mean']:.2f}")

        bm = res["blur_matrix"]
        b_stats = res["blur_stats"]
        self.blur_view.set_matrix(bm)
        self.blur_stats_lbl.config(
            text=f"Mean: {b_stats['mean']:.2f}  |  Min: {b_stats['min']:.2f}  |  Max: {b_stats['max']:.2f}  |  Std: {b_stats['std']:.2f}  |  Median: {b_stats['median']:.2f}"
        )

        am = res["angle_matrix"]
        a_stats = res["angle_stats"]
        self.angle_view.set_matrix(am, vmin=0.0, vmax=45.0)
        self.angle_stats_lbl.config(
            text=f"Mean: {a_stats['mean']:.2f}°  |  Min: {a_stats['min']:.2f}°  |  Max: {a_stats['max']:.2f}°  |  Std: {a_stats['std']:.2f}°  |  Median: {a_stats['median']:.2f}°"
        )

        self._update_metrics(res)

        for i, st in enumerate(res["stages"]):
            if i < len(self.stage_cards):
                self.stage_cards[i].view.set_image(st["image"])
                self.stage_cards[i].set_title(st["name"], st["desc"])

        self._predict_inline()

        self.v_status.set(f"Pipeline executed for {self.v_exp_id.get()} (Grid: {rows}x{cols}). Ready to save.")

    def _update_metrics(self, res):
        m = res["metrics"]
        active = res.get("active_method", "laplacian")
        single = "all" not in self.v_algo.get().lower()
        for key, cells in self.metric_cells.items():
            d = m[key]
            sf = (lambda v: f"{v:.2f}%") if key == "fft" else W.fmt_compact
            vals = [None, sf(d["sample"]), sf(d["ref"]), f"{d['global_ratio']:.3f}",
                    f"{d['ratio_mean']:.3f}", f"{d['ratio_std']:.3f}",
                    f"{d['ratio_p90']:.3f}", f"{d['angle_mean']:.2f}\u00b0"]
            hl = single and key == active
            for c, lbl in enumerate(cells):
                if vals[c] is not None:
                    lbl.config(text=vals[c])
                lbl.config(fg=W.AMBER if hl else W.TEXT)

        rm = res["ratio_matrix"]
        self.ratio_view.set_matrix(rm, vmin=0.0, vmax=1.0)
        rs = blur_matrix.matrix_stats(rm)
        self.ratio_stats_lbl.config(
            text=f"Mean: {rs['mean']:.3f}  |  Min: {rs['min']:.3f}  |  Max: {rs['max']:.3f}  |  "
                 f"Std: {rs['std']:.3f}  |  Median: {rs['median']:.3f}")
        self.v_blur_ratio.set(f"{m[active]['ratio_mean']:.3f}")

        kh, kw = res.get("kernel_px", (0, 0))
        clip = res["clipped_pct"]
        self.raw_feat_lbl.config(
            text=(f"Shift dx={res['shift_dx']:.1f} dy={res['shift_dy']:.1f}px  |  "
                  f"Diff mean={res['diff_mean']:.2f} std={res['diff_std']:.2f}  |  "
                  f"Intensity ratio={res['intensity_ratio']:.3f}  |  Contrast ratio={res['contrast_ratio']:.3f}\n"
                  f"Kernel {kh}x{kw}px  |  Blur matrix {res['grid_rows']}x{res['grid_cols']} "
                  f"({res['grid_rows'] * res['grid_cols']} cells)  |  Clipped pixels={clip:.2f}%"
                  + ("  !! >2% clipped: lower exposure" if clip > 2.0 else "")),
            fg=W.RED if clip > 2.0 else W.TEXT)

    def show_ratio_numerical_matrix(self):
        if self.proc_res is None or "ratio_matrix" not in self.proc_res:
            messagebox.showinfo("No Data", "Capture or load an image first to compute the Blur Ratio Matrix.")
            return
        W.show_numerical_matrix_dialog(self, self.proc_res["ratio_matrix"], title="Numerical Blur Ratio Matrix", unit="")

    def show_blur_numerical_matrix(self):
        if self.proc_res is None or "blur_matrix" not in self.proc_res:
            messagebox.showinfo("No Data", "Capture or load an image first to compute the Blur Matrix.")
            return
        W.show_numerical_matrix_dialog(self, self.proc_res["blur_matrix"], title="Numerical Blur Matrix", unit="")

    def show_angle_numerical_matrix(self):
        if self.proc_res is None or "angle_matrix" not in self.proc_res:
            messagebox.showinfo("No Data", "Capture or load an image first to compute the Scattering Angle Matrix.")
            return
        W.show_numerical_matrix_dialog(self, self.proc_res["angle_matrix"], title="Numerical Scattering Angle Matrix", unit="°")

    def save_experiment(self):
        if self.proc_res is None or self.captured_img is None:
            messagebox.showwarning("Save Incomplete", "Please capture or load a sample image before saving.")
            return

        exp_id = self.v_exp_id.get().strip() or data_manager.get_next_experiment_id()

        gt = {
            "fluid_type": self.v_fluid.get().strip(),
            "particle_material": self.v_material.get().strip(),
            "particle_size": self.v_size.get().strip(),
            "particle_size_unit": self.v_size_unit.get().strip(),
            "particle_concentration": self.v_conc.get().strip(),
            "concentration_unit": self.v_conc_unit.get().strip(),
            "known_NTU": self.v_ntu.get().strip(),
            "sample_id": self.v_sample_id.get().strip(),
            "algorithm": self.v_algo.get().strip(),
        }

        row = data_manager.save_experiment_record(
            exp_id,
            self.ref_meta,
            self.captured_img,
            self.proc_res,
            gt
        )

        messagebox.showinfo("Experiment Saved", f"Successfully saved {exp_id} to dataset!\n\nCaptured Image: data/captures/{row['captured_image_name']}\nBlur Matrix: data/blur_matrices/{row['blur_matrix_file']}\nAngles Matrix: data/scattering_angles/{row['scattering_angles_file']}\nMaster CSV: data/experiments.csv")

        self.active_exp_id = data_manager.get_next_experiment_id()
        self.v_exp_id.set(self.active_exp_id)
        self.v_status.set(f"Saved {exp_id}. Next ID: {self.active_exp_id}")

        self.load_history_table()
        self.refresh_regression_stats()

    def export_csv_dialog(self):
        dest = filedialog.asksaveasfilename(
            title="Export Experiments CSV Dataset",
            initialfile="experiments_export.csv",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if dest:
            import shutil
            shutil.copy2(data_manager.CSV_PATH, dest)
            messagebox.showinfo("Export Complete", f"Exported complete experiment dataset to:\n{dest}")

    def load_history_table(self):
        for item in self.hist_tree.get_children():
            self.hist_tree.delete(item)

        records = data_manager.load_experiments()
        for r in reversed(records):
            vals = [
                r.get("experiment_id", ""),
                r.get("timestamp", ""),
                r.get("reference_image_name", ""),
                r.get("captured_image_name", ""),
                r.get("algorithm", ""),
                r.get("fluid_type", ""),
                r.get("particle_material", ""),
                r.get("particle_size", ""),
                r.get("particle_concentration", ""),
                r.get("known_NTU", ""),
                r.get("mean_blur", ""),
                r.get("mean_scattering_angle", ""),
                r.get("status", "Saved"),
            ]
            self.hist_tree.insert("", "end", values=vals)

    def load_selected_history_item(self):
        sel = self.hist_tree.selection()
        if not sel:
            messagebox.showinfo("Select Experiment", "Please click on an experiment row first.")
            return

        item = self.hist_tree.item(sel[0])
        exp_id = item["values"][0]

        detail = data_manager.load_experiment_detail(exp_id)
        if not detail:
            messagebox.showerror("Error", f"Could not load files for {exp_id}.")
            return

        rec = detail["record"]

        self.v_exp_id.set(rec.get("experiment_id", ""))
        self.v_fluid.set(rec.get("fluid_type", ""))
        self.v_material.set(rec.get("particle_material", ""))
        self.v_size.set(rec.get("particle_size", ""))
        self.v_size_unit.set(rec.get("particle_size_unit", "µm"))
        self.v_conc.set(rec.get("particle_concentration", ""))
        self.v_conc_unit.set(rec.get("concentration_unit", "mg/L"))
        self.v_ntu.set(rec.get("known_NTU", ""))
        self.v_sample_id.set(rec.get("sample_id", ""))
        self.v_algo.set(rec.get("algorithm", "All Algorithms"))

        if detail["reference_image"] is not None:
            self.ref_img = detail["reference_image"]
            self.card_ref.view.set_image(self.ref_img)
            self.v_ref_info.set(f"Historical Ref: {rec.get('reference_image_name', '')}")
        if detail["captured_image"] is not None:
            self.captured_img = detail["captured_image"]

        self._run_pipeline()
        self.notebook.select(self.tab_dash)
        self.v_status.set(f"Loaded historical experiment {exp_id}.")

    def refresh_regression_stats(self):
        stats = data_manager.get_dataset_stats()
        txt = (f"Dataset Stats: Total: {stats['total']} | "
               f"With Known NTU: {stats['has_ntu']} | "
               f"With Known Particle Size: {stats['has_size']} | "
               f"Complete Records: {stats['complete']} | "
               f"Incomplete: {stats['incomplete']}")
        self.reg_stats_card.config(text=txt)

    def export_regression_dataset_action(self):
        dest, count = data_manager.export_regression_dataset()
        self.refresh_regression_stats()
        messagebox.showinfo("Regression Dataset Exported",
                            f"Exported clean regression dataset with {count} valid records.\n\nFile:\n{dest}\n\nFeatures: Laplacian, Tenengrad, FFT, Blur stats, Scattering stats, Diff stats.\nTargets: known_NTU, particle_size, particle_concentration.")

    def train_model_action(self, target="known_NTU"):
        records = data_manager.load_experiments()
        try:
            meta = regression_engine.train_regression_model(records, target=target, model_type="ridge")
            plot_img = regression_engine.render_regression_plot(meta, width=460, height=280)

            if target == "known_NTU":
                self.plot_ntu_view.set_image(plot_img)
                self.lbl_ntu_metrics.config(text=f"R2: {meta['r2']:.3f}  |  RMSE: {meta['rmse']:.2f} NTU  |  MAE: {meta['mae']:.2f}")
            else:
                self.plot_size_view.set_image(plot_img)
                self.lbl_size_metrics.config(text=f"R2: {meta['r2']:.3f}  |  RMSE: {meta['rmse']:.2f} µm  |  MAE: {meta['mae']:.2f}")

            messagebox.showinfo("Model Trained", f"Successfully trained {target} regression model!\n\nSamples: {meta['n_samples']}\nR2 Score: {meta['r2']:.3f}\nRMSE: {meta['rmse']:.3f}")
            self._predict_inline()
        except Exception as e:
            messagebox.showwarning("Training Error", str(e))

    def _extract_current_features_dict(self):
        if self.proc_res is None:
            return None
        scores = self.proc_res["all_blur_scores"]
        ref_scores = self.proc_res.get("ref_blur_scores", {})
        b_s = self.proc_res["blur_stats"]
        a_s = self.proc_res["angle_stats"]

        return {
            "laplacian_variance": scores.get("laplacian", 0.0),
            "tenengrad": scores.get("tenengrad", 0.0),
            "fft_high_frequency_energy": scores.get("fft", 0.0),
            "mean_blur": b_s.get("mean", 0.0),
            "min_blur": b_s.get("min", 0.0),
            "max_blur": b_s.get("max", 0.0),
            "median_blur": b_s.get("median", 0.0),
            "std_blur": b_s.get("std", 0.0),
            "mean_scattering_angle": a_s.get("mean", 0.0),
            "min_scattering_angle": a_s.get("min", 0.0),
            "max_scattering_angle": a_s.get("max", 0.0),
            "median_scattering_angle": a_s.get("median", 0.0),
            "std_scattering_angle": a_s.get("std", 0.0),
            "diff_mean": self.proc_res.get("diff_mean", 0.0),
            "diff_std": self.proc_res.get("diff_std", 0.0),
            "ref_laplacian_variance": ref_scores.get("laplacian", 0.0),
            "ref_tenengrad": ref_scores.get("tenengrad", 0.0),
            "ref_fft_energy": ref_scores.get("fft", 0.0),
        }

    def _predict_inline(self):
        feats = self._extract_current_features_dict()
        if feats is None:
            return

        ntu_mod = regression_engine.load_model("known_NTU")
        if ntu_mod:
            p = regression_engine.predict_target(ntu_mod, feats)
            if p.get("value") is not None:
                self.v_pred_ntu.set(f"{p['value']:.1f} NTU")

        size_mod = regression_engine.load_model("particle_size")
        if size_mod:
            p2 = regression_engine.predict_target(size_mod, feats)
            if p2.get("value") is not None:
                self.v_pred_size.set(f"{p2['value']:.1f} µm")

    def predict_current_frame(self):
        feats = self._extract_current_features_dict()
        if feats is None:
            messagebox.showwarning("No Features", "Please capture or load a frame first.")
            return

        ntu_mod = regression_engine.load_model("known_NTU")
        size_mod = regression_engine.load_model("particle_size")

        msg = []
        if ntu_mod:
            r1 = regression_engine.predict_target(ntu_mod, feats)
            msg.append(f"Predicted NTU: {r1['value']:.2f} NTU (Model R2: {r1['r2']:.2f})")
            self.v_pred_ntu.set(f"{r1['value']:.1f} NTU")
        else:
            msg.append("NTU Model: Not yet trained. Save >= 3 samples with Known NTU to train.")

        if size_mod:
            r2 = regression_engine.predict_target(size_mod, feats)
            msg.append(f"Predicted Particle Size: {r2['value']:.2f} µm (Model R2: {r2['r2']:.2f})")
            self.v_pred_size.set(f"{r2['value']:.1f} µm")
        else:
            msg.append("Particle Size Model: Not yet trained. Save >= 3 samples with Particle Size to train.")

        messagebox.showinfo("Regression Predictions", "\n\n".join(msg))

    def _on_close(self):
        if self.cam is not None:
            self.cam.release()
        self.destroy()

if __name__ == "__main__":
    app = TurbidityApp()
    app.mainloop()