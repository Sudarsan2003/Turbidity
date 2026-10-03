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

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np

import rank_patterns
import rigcore
import ui_widgets as W
from camera import Camera
from hardware import Projector, flush
import image_select
from model import TurbidityModel, fit_from_sweep

BASE = os.path.dirname(os.path.abspath(__file__))

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Turbidity rig - operator")
        self.configure(bg=W.BG)
        self.geometry("1320x820")
        self.cam = self.proj = self.ref = self.model = None
        self.busy = False
        self.live = False
        self.file_mode = False
        self._last_meas = 0.0
        self.q = queue.Queue()
        self.v = {k: tk.StringVar(value=val) for k, val in dict(
            cam="0", exposure="", gain="", navg="10", px="1920", py="0", pw="1280",
            ph="720", windowed="0", patterns="checker,vstripe,hstripe,dots,circles,random",
            feats="12,24,48", fluid="formazin", label="S1", ntu="0", particle="",
            reps="5", blank="0", mpat="", mfeat="").items()}
        self._build()
        self.after(100, self._tick)
        self.after(100, self._drain)
        self.protocol("WM_DELETE_WINDOW", self._quit)

    def _entry(self, parent, text, key, width=8):
        f = tk.Frame(parent, bg=W.PANEL)
        f.pack(fill="x", pady=1)
        tk.Label(f, text=text, bg=W.PANEL, fg=W.DIM, font=(W.FONT_UI, 9),
                 width=11, anchor="w").pack(side="left")
        tk.Entry(f, textvariable=self.v[key], width=width, bg=W.FIELD, fg=W.TEXT,
                 insertbackground=W.TEXT, relief="flat").pack(side="left", fill="x", expand=True)

    def _btn(self, parent, text, cmd, color=W.SHARP):
        tk.Button(parent, text=text, command=cmd, bg=color, fg="#06110f", relief="flat",
                  font=(W.FONT_UI, 10, "bold"), pady=4).pack(fill="x", pady=3)

    def _section(self, parent, text):
        tk.Label(parent, text=text, bg=W.PANEL, fg=W.AMBER,
                 font=(W.FONT_UI, 10, "bold")).pack(anchor="w", pady=(10, 2))

    def _scroll_panel(self):

        outer = tk.Frame(self, bg=W.PANEL, width=300)
        outer.pack(side="left", fill="y", padx=(8, 4), pady=8)
        outer.pack_propagate(False)
        canvas = tk.Canvas(outer, bg=W.PANEL, highlightthickness=0)
        sb = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=W.PANEL)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))

        def wheel(e):
            if sys.platform == "darwin":
                step = -e.delta
            elif sys.platform.startswith("win"):
                step = -int(e.delta / 120)
            else:
                step = -1 if e.num == 4 else 1
            canvas.yview_scroll(step, "units")

        def bind(_e):
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                canvas.bind_all(ev, wheel)

        def unbind(_e):
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                canvas.unbind_all(ev)

        outer.bind("<Enter>", bind)
        outer.bind("<Leave>", unbind)
        return inner

    def _build(self):
        left = self._scroll_panel()
        self._section(left, "1  Hardware")
        for t, k in (("camera idx", "cam"), ("exposure", "exposure"), ("gain", "gain"),
                     ("frames avg", "navg"), ("proj x / y", "px"), ("", "py"),
                     ("proj w / h", "pw"), ("", "ph")):
            self._entry(left, t, k)
        tk.Checkbutton(left, text="projector windowed (test)", variable=self.v["windowed"],
                       onvalue="1", offvalue="0", bg=W.PANEL, fg=W.TEXT,
                       selectcolor=W.FIELD, activebackground=W.PANEL).pack(anchor="w")
        self._btn(left, "Connect / reconnect", self.connect)
        self._section(left, "2  Clear-water references")
        self._entry(left, "patterns", "patterns", 22)
        self._entry(left, "feature px", "feats", 22)
        self._btn(left, "Capture references", self.do_refs, W.BLUE)
        self._section(left, "3  Sample sweep")
        for t, k in (("fluid", "fluid"), ("label", "label"), ("NTU", "ntu"),
                     ("particle um", "particle"), ("repeats", "reps")):
            self._entry(left, t, k)
        tk.Checkbutton(left, text="fresh clear water (noise blank)", variable=self.v["blank"],
                       onvalue="1", offvalue="0", bg=W.PANEL, fg=W.TEXT,
                       selectcolor=W.FIELD, activebackground=W.PANEL).pack(anchor="w")
        self._btn(left, "Run sample sweep", self.do_sample, W.BLUE)
        self._section(left, "4-6  Analyse and measure")
        self._entry(left, "pattern", "mpat")
        self._entry(left, "feature px", "mfeat")
        self._btn(left, "Rank patterns", self.do_rank, W.VIOLET)
        self._btn(left, "Fit model", self.do_fit, W.VIOLET)
        self._btn(left, "Analyze image file...", self.analyze_file, W.BLUE)
        self._btn(left, "Use file as reference...", self.ref_from_file, W.BLUE)
        self.live_btn = tk.Button(left, text="Live measure: OFF", command=self.toggle_live,
                                  bg=W.AMBER, fg="#1a1204", relief="flat",
                                  font=(W.FONT_UI, 10, "bold"), pady=4)
        self.live_btn.pack(fill="x", pady=3)

        right = tk.Frame(self, bg=W.BG)
        right.pack(side="left", fill="both", expand=True, padx=(4, 8), pady=8)
        cards = tk.Frame(right, bg=W.BG)
        cards.pack(fill="both", expand=True)
        self.c_live = W.StageCard(cards, "A", "Live camera", "locked exposure", W.SHARP, kind="image")
        self.c_ref = W.StageCard(cards, "B", "Clear-water reference", "same pattern", W.BLUE, kind="image")
        self.c_map = W.StageCard(cards, "C", "Blur vs reference", "0 = clear water, 1 = blurred",
                                 W.AMBER, kind="matrix", fmt=W.fmt_ratio)
        for c in (self.c_live, self.c_ref, self.c_map):
            c.pack(side="left", fill="both", expand=True, padx=4)
        self.c_map.colorbar.set_range("0", "1")
        self.read = tk.Label(right, text="NTU  --      size  --", bg=W.CARD, fg=W.SHARP,
                             font=(W.FONT_MONO, 22, "bold"), pady=10)
        self.read.pack(fill="x", pady=6)
        self.log = tk.Text(right, height=13, bg=W.PANEL, fg=W.TEXT, relief="flat",
                           font=(W.FONT_MONO, 9), insertbackground=W.TEXT)
        self.log.pack(fill="x")
        self.say("Start with 1 Connect. Clear water must be in the cell for step 2.")

    def say(self, s):
        self.log.insert("end", s + "\n")
        self.log.see("end")

    def _drain(self):
        try:
            while True:
                item = self.q.get_nowait()
                item() if callable(item) else self.say(item)
        except queue.Empty:
            pass
        self.after(100, self._drain)

    def _num(self, key, cast=float):
        s = self.v[key].get().strip()
        return None if s == "" else cast(s)

    def _hw_argv(self):
        a = ["--cam", self.v["cam"].get(), "--n-avg", self.v["navg"].get(),
             "--proj-x", self.v["px"].get(), "--proj-y", self.v["py"].get(),
             "--proj-w", self.v["pw"].get(), "--proj-h", self.v["ph"].get()]
        for k, flag in (("exposure", "--exposure"), ("gain", "--gain")):
            if self.v[k].get().strip():
                a += [flag, self.v[k].get().strip()]
        if self.v["windowed"].get() == "1":
            a.append("--proj-windowed")
        return a

    def _release(self):
        self.live = False
        self.live_btn.config(text="Live measure: OFF")
        if self.proj:
            self.proj.close()
        if self.cam:
            self.cam.release()
        self.cam = self.proj = None

    def connect(self):
        if self.busy:
            return
        self._release()
        self.file_mode = False
        try:
            self.cam = Camera(self._num("cam", int) or 0, 480, 360)
            rigcore.lock_camera(self.cam, self._num("exposure"), self._num("gain"))
            self.proj = Projector(self._num("pw", int), self._num("ph", int),
                                  self._num("px", int), self._num("py", int),
                                  self.v["windowed"].get() == "1")
            self.say("connected: camera locked (auto exposure/WB/focus off).")
        except Exception as e:
            self.cam = self.proj = None
            self.say(f"CONNECT FAILED: {e}")

    def _sub(self, argv, done_msg):
        if self.busy:
            return
        self._release()
        self.busy = True
        cmd = [sys.executable, "-u", os.path.join(BASE, "pattern_sweep.py")] + argv

        def work():
            p = subprocess.Popen(cmd, cwd=BASE, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                self.q.put(line.rstrip())
            p.wait()
            def fin():
                self.busy = False
                self.say(done_msg if p.returncode == 0 else f"FAILED (exit {p.returncode})")
                self.connect()
            self.q.put(fin)
        threading.Thread(target=work, daemon=True).start()

    def do_refs(self):
        self.say("capturing references - keep CLEAR water in the cell...")
        self._sub(["reference", "--patterns", self.v["patterns"].get(),
                   "--feats", self.v["feats"].get()] + self._hw_argv(), "references done.")

    def do_sample(self):
        try:
            ntu = float(self.v["ntu"].get())
        except ValueError:
            return self.say("enter a numeric NTU")
        argv = ["sample", "--patterns", self.v["patterns"].get(), "--feats", self.v["feats"].get(),
                "--fluid", self.v["fluid"].get(), "--label", self.v["label"].get(),
                "--ntu", str(ntu), "--reps", self.v["reps"].get()]
        if self.v["particle"].get().strip():
            argv += ["--particle", self.v["particle"].get().strip()]
        if self.v["blank"].get() == "1" or ntu == 0:
            argv.append("--blank")
        self.say(f"sweeping sample {self.v['label'].get()} ({ntu:g} NTU)...")
        self._sub(argv + self._hw_argv(), "sample done.")

    def _thread(self, fn):
        def work():
            try:
                fn()
            except Exception as e:
                self.q.put(f"ERROR: {e}")
        threading.Thread(target=work, daemon=True).start()

    def do_rank(self):
        def fn():
            txt, _ = rank_patterns.run(10)
            self.q.put(txt)
        self._thread(fn)

    def do_fit(self):
        pat = self.v["mpat"].get().strip() or None
        feat = self._num("mfeat", int) if pat else None
        def fn():
            m = fit_from_sweep(pat, feat, self.v["fluid"].get().strip() or None)
            self.q.put(m.summary())
            self.q.put("model saved.")
        self._thread(fn)

    def analyze_file(self):
\
\
\
\

        path = image_select.pick_image(self, "Select an image to analyze")
        if not path:
            return
        model = None
        try:
            model = TurbidityModel.load()
        except Exception:
            pass
        try:
            if model is not None:
                ref = rigcore.load_reference(model.pattern, model.feat, model.grid_cols)
                if ref is None:
                    return self.say("reference image for the model pattern is missing")
            else:
                pat = self.v["mpat"].get().strip()
                ft = self._num("mfeat", int)
                ref = rigcore.load_reference(pat, ft) if pat and ft is not None else None
                if ref is None:
                    self.say("no model yet - pick the CLEAR-WATER image to compare against")
                    rp = image_select.pick_image(self, "Select the CLEAR-WATER image")
                    if not rp:
                        return
                    ref = image_select.reference_from_file(rp)
            r = image_select.analyze_file(path, ref, model)
        except Exception as e:
            return self.say(f"cannot analyze: {e}")
        self.model, self.ref = model, ref
        self.live = False
        self.live_btn.config(text="Live measure: OFF")
        self.file_mode = True
        f, res = r["features"], r["result"]
        name = os.path.basename(path)
        if res is not None:
            lod = model.d["lod_ntu"]
            ntu = f"<{lod:.1f}" if res["below_lod"] else f"{res['ntu']:.1f}"
            size = "--" if res["size_um"] is None else f"{res['size_um']:.2f}um"
            self.read.config(text=f"NTU {ntu}     size {size}",
                             fg=W.RED if res["flags"] else W.SHARP)
            foot = "; ".join(res["flags"]) or "within calibration"
            msg = f"{name}: NTU {ntu}, size {size}"
        else:
            lr = f["laplacian_ratio_mean"]
            self.read.config(text=f"blur {lr:.2f}   (no model yet)", fg=W.AMBER)
            foot = "no model: 0 = like clear water, 1 = fully blurred"
            msg = (f"{name}: blur ratio lap={lr:.3f} fft={f['fft_ratio_mean']:.3f} "
                   f"int={f['int_ratio']:.3f}  (fit a model for NTU)")
        self.c_live.view.set_image(r["image"])
        self.c_live.set_title("Image file", name)
        self.c_ref.view.set_image(ref["image"])
        mats = rigcore.matrices(r["image"], ref["grid_cols"])["laplacian"]
        rm = np.clip(1 - mats / np.maximum(ref["mats"]["laplacian"], 1e-9), 0, 1)
        rm[~ref["masks"]["laplacian"]] = 0
        self.c_map.view.set_matrix(rm, 0, 1)
        self.c_map.set_footer(foot)
        self.say(msg + ("  (resized to 480x360)" if r["resized"] else ""))

    def ref_from_file(self):

        pat = self.v["mpat"].get().strip()
        feat = self._num("mfeat", int)
        if not pat or feat is None:
            return self.say("type the pattern and feature px (section 4-6) first")
        path = image_select.pick_image(self, "Select the CLEAR-WATER image")
        if not path:
            return
        try:
            out, rs = image_select.save_as_reference(path, pat, feat)
            self.say(f"reference for {pat} f{feat} stored from {os.path.basename(path)}"
                     + ("  (resized to 480x360)" if rs else ""))
        except Exception as e:
            self.say(f"cannot store reference: {e}")

    def toggle_live(self):
        self.file_mode = False
        if self.live:
            self.live = False
            self.live_btn.config(text="Live measure: OFF")
            return
        if not self.cam:
            return self.say("connect first")
        try:
            self.model = TurbidityModel.load()
            self.ref = rigcore.load_reference(self.model.pattern, self.model.feat,
                                              self.model.grid_cols)
            if self.ref is None:
                return self.say("reference image for the model pattern is missing")
            self.proj.show(self.model.pattern, self.model.feat)
            flush(self.cam)
        except Exception as e:
            return self.say(f"cannot start live: {e}")
        self.c_ref.view.set_image(self.ref["image"])
        self.c_ref.set_title("Clear-water reference", f"{self.model.pattern} f{self.model.feat}")
        self.live = True
        self.live_btn.config(text="Live measure: ON")

    def _tick(self):
        try:
            if self.cam and not self.busy and not self.file_mode:
                self.c_live.view.set_image(self.cam.read_frame())
                if self.live and time.time() - self._last_meas > 1.2:
                    self._measure()
        except Exception as e:
            self.say(f"camera error: {e}")
            self.live = False
        self.after(120, self._tick)

    def _measure(self):
        img, sat = rigcore.capture_average(self.cam, 5)
        f = rigcore.frame_features(img, self.ref)
        r = self.model.predict(f)
        lod = self.model.d["lod_ntu"]
        ntu = f"<{lod:.1f}" if r["below_lod"] else f"{r['ntu']:.1f}"
        size = "--" if r["size_um"] is None else f"{r['size_um']:.2f}um"
        self.read.config(text=f"NTU {ntu}     size {size}",
                         fg=W.RED if r["flags"] or sat > 0.02 else W.SHARP)
        mats = rigcore.matrices(img, self.ref["grid_cols"])["laplacian"]
        rm = np.clip(1 - mats / np.maximum(self.ref["mats"]["laplacian"], 1e-9), 0, 1)
        rm[~self.ref["masks"]["laplacian"]] = 0
        self.c_map.view.set_matrix(rm, 0, 1)
        self.c_map.set_footer("; ".join(r["flags"]) or "within calibration")
        self._last_meas = time.time()

    def _quit(self):
        self._release()
        self.destroy()

if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    App().mainloop()
