import sys
import tkinter as tk
from tkinter import font as tkfont

import cv2
import numpy as np
from PIL import Image, ImageTk


BG = "#0b0e12"
PANEL = "#12171d"
CARD = "#171d25"
FIELD = "#1e2630"
LINE = "#252d37"
TEXT = "#dde5ec"
DIM = "#7f8c99"
SHARP = "#3ED6C6"
AMBER = "#F2B84B"
BLUE = "#5aa9ff"
VIOLET = "#c792ea"
RED = "#ff6b6b"

if sys.platform == "darwin":
    FONT_UI, FONT_MONO = "Helvetica Neue", "Menlo"
elif sys.platform.startswith("win"):
    FONT_UI, FONT_MONO = "Segoe UI", "Consolas"
else:
    FONT_UI, FONT_MONO = "DejaVu Sans", "DejaVu Sans Mono"


def fmt_compact(v):
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.1f}M"
    if a >= 1e4:
        return f"{v / 1e3:.0f}k"
    if a >= 1e3:
        return f"{v / 1e3:.1f}k"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def fmt_ratio(v):
    return f"{v:.2f}"


def fmt_angle(v):
    return f"{v:.1f}"


_FONT_CACHE = {}


def _mono(pt):
    if pt not in _FONT_CACHE:
        _FONT_CACHE[pt] = tkfont.Font(family=FONT_MONO, size=pt)
    return _FONT_CACHE[pt]


def fit_font(maxlen, cw, ch, hi=11, lo=5):
    for pt in range(hi, lo - 1, -1):
        f = _mono(pt)
        if f.measure("0" * maxlen) <= cw - 3 and f.metrics("linespace") <= ch - 1:
            return pt
    return None


_LUT_CACHE = {}


def make_lut(cmap):
    if cmap not in _LUT_CACHE:
        ramp = np.arange(256, dtype=np.uint8).reshape(1, 256)
        bgr = cv2.applyColorMap(ramp, cmap)[0]
        _LUT_CACHE[cmap] = [
            ("#%02x%02x%02x" % (int(r), int(g), int(b)),
             0.299 * r + 0.587 * g + 0.114 * b)
            for b, g, r in bgr
        ]
    return _LUT_CACHE[cmap]


class MatrixView(tk.Canvas):
    def __init__(self, parent, cmap=cv2.COLORMAP_JET, fmt=fmt_compact,
                 aspect=None, hover_cb=None, dbl_cb=None, interactive=True, **kw):
        kw.setdefault("width", 220)
        kw.setdefault("height", 165)
        super().__init__(parent, bg=CARD, highlightthickness=0, **kw)
        self._lut = make_lut(cmap)
        self.fmt = fmt
        self.aspect = aspect
        self.hover_cb = hover_cb
        self.matrix = None
        self.vmin = self.vmax = None
        self._msg = ""
        self._geom = None
        self._hover_cell = None
        self._hl_cell = None

        self.bind("<Configure>", lambda e: self.redraw())
        if interactive:
            self.bind("<Motion>", self._on_motion)
            self.bind("<Leave>", self._on_leave)
        if dbl_cb:
            self.bind("<Double-Button-1>", lambda e: dbl_cb())
            self.configure(cursor="hand2")


    def set_matrix(self, m, vmin=None, vmax=None):
        self.matrix = np.asarray(m, dtype=np.float64)
        self.vmin = float(self.matrix.min() if vmin is None else vmin)
        self.vmax = float(self.matrix.max() if vmax is None else vmax)
        self.redraw()

    def clear(self, msg=""):
        self.matrix = None
        self._msg = msg
        self.redraw()

    def highlight(self, cell):
        self._hl_cell = cell
        self._draw_highlight()


    def redraw(self):
        self.delete("all")
        W, H = self.winfo_width(), self.winfo_height()
        if W < 20 or H < 20:
            return
        if self.matrix is None:
            self._geom = None
            self.create_text(W / 2, H / 2, text=self._msg or "—", fill=DIM,
                             font=(FONT_MONO, 10))
            return

        pad = 4
        aw, ah = W - 2 * pad, H - 2 * pad
        if self.aspect is None:
            dw, dh = aw, ah
        elif aw / ah > self.aspect:
            dh, dw = ah, ah * self.aspect
        else:
            dw, dh = aw, aw / self.aspect
        x0, y0 = (W - dw) / 2, (H - dh) / 2
        rows, cols = self.matrix.shape
        cw, ch = dw / cols, dh / rows
        self._geom = (x0, y0, cw, ch, rows, cols)

        span = (self.vmax - self.vmin) or 1.0
        idx = np.clip((self.matrix - self.vmin) / span, 0, 1) * 255
        idx = idx.astype(int)

        texts = [[self.fmt(self.matrix[r, c]) for c in range(cols)] for r in range(rows)]
        maxlen = max(len(t) for row in texts for t in row)
        pt = fit_font(maxlen, cw, ch)
        show_text = pt is not None

        for r in range(rows):
            for c in range(cols):
                colour, lum = self._lut[idx[r, c]]
                xa, ya = x0 + c * cw, y0 + r * ch
                self.create_rectangle(xa, ya, xa + cw, ya + ch, fill=colour,
                                      outline=CARD, width=1)
                if show_text:
                    self.create_text(xa + cw / 2, ya + ch / 2, text=texts[r][c],
                                     fill="#0b0e12" if lum > 130 else "#f4f7fa",
                                     font=(FONT_MONO, pt))
        self._draw_highlight()

    def _draw_highlight(self):
        self.delete("hl")
        if self._geom is None or self._hl_cell is None:
            return
        x0, y0, cw, ch, rows, cols = self._geom
        r, c = self._hl_cell
        if 0 <= r < rows and 0 <= c < cols:
            self.create_rectangle(x0 + c * cw, y0 + r * ch,
                                  x0 + (c + 1) * cw, y0 + (r + 1) * ch,
                                  outline="#ffffff", width=2, tags="hl")


    def _cell_at(self, x, y):
        if self._geom is None:
            return None
        x0, y0, cw, ch, rows, cols = self._geom
        c, r = int((x - x0) // cw), int((y - y0) // ch)
        if 0 <= r < rows and 0 <= c < cols and x >= x0 and y >= y0:
            return (r, c)
        return None

    def _on_motion(self, e):
        cell = self._cell_at(e.x, e.y)
        if cell != self._hover_cell:
            self._hover_cell = cell
            if self.hover_cb:
                self.hover_cb(cell)

    def _on_leave(self, _e):
        self._hover_cell = None
        if self.hover_cb:
            self.hover_cb(None)


class ImageView(tk.Canvas):
    def __init__(self, parent, **kw):
        kw.setdefault("width", 220)
        kw.setdefault("height", 165)
        super().__init__(parent, bg=CARD, highlightthickness=0, **kw)
        self._bgr = None
        self._tk = None
        self.bind("<Configure>", lambda e: self.redraw())

    def set_image(self, bgr):
        self._bgr = bgr
        self.redraw()

    def redraw(self):
        self.delete("all")
        W, H = self.winfo_width(), self.winfo_height()
        if W < 20 or H < 20:
            return
        if self._bgr is None:
            self.create_text(W / 2, H / 2, text="no image", fill=DIM,
                             font=(FONT_MONO, 10))
            return
        h, w = self._bgr.shape[:2]
        s = min((W - 8) / w, (H - 8) / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        rgb = cv2.cvtColor(self._bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR)
        self._tk = ImageTk.PhotoImage(im)
        self.create_image(W // 2, H // 2, image=self._tk)
        self.create_rectangle((W - nw) // 2, (H - nh) // 2,
                              (W + nw) // 2, (H + nh) // 2, outline=LINE)


class ColorBar(tk.Frame):
    def __init__(self, parent, cmap):
        super().__init__(parent, bg=CARD)
        self._lut = make_lut(cmap)
        self.lo = tk.Label(self, text="", bg=CARD, fg=DIM, font=(FONT_MONO, 8))
        self.lo.pack(side="left")
        self.bar = tk.Canvas(self, height=8, bg=CARD, highlightthickness=0)
        self.bar.pack(side="left", fill="x", expand=True, padx=6)
        self.hi = tk.Label(self, text="", bg=CARD, fg=DIM, font=(FONT_MONO, 8))
        self.hi.pack(side="right")
        self.bar.bind("<Configure>", self._draw)

    def _draw(self, _e=None):
        w = self.bar.winfo_width()
        self.bar.delete("all")
        n = 48
        step = w / n
        for i in range(n):
            colour = self._lut[int(i / (n - 1) * 255)][0]
            self.bar.create_rectangle(i * step, 0, (i + 1) * step + 1, 8,
                                      fill=colour, outline="")

    def set_range(self, lo_text, hi_text):
        self.lo.config(text=lo_text)
        self.hi.config(text=hi_text)


class StageCard(tk.Frame):
    def __init__(self, parent, badge, title, subtitle, accent, kind="matrix",
                 cmap=cv2.COLORMAP_JET, fmt=fmt_compact, hover_cb=None, dbl_cb=None):
        super().__init__(parent, bg=CARD, highlightbackground=LINE,
                         highlightthickness=1)
        head = tk.Frame(self, bg=CARD)
        head.pack(fill="x", padx=10, pady=(10, 0))
        tk.Label(head, text=badge, bg=accent, fg="#06110f",
                 font=(FONT_UI, 9, "bold"), width=2).pack(side="left")
        self.title_lbl = tk.Label(head, text=title, bg=CARD, fg=TEXT,
                                  font=(FONT_UI, 10, "bold"))
        self.title_lbl.pack(side="left", padx=8)
        self.sub_lbl = tk.Label(self, text=subtitle, bg=CARD, fg=DIM,
                                font=(FONT_MONO, 8), anchor="w")
        self.sub_lbl.pack(fill="x", padx=10, pady=(3, 6))

        if kind == "matrix":
            self.view = MatrixView(self, cmap=cmap, fmt=fmt,
                                   hover_cb=hover_cb, dbl_cb=dbl_cb)
        else:
            self.view = ImageView(self)
        self.view.pack(fill="both", expand=True, padx=8)

        self.colorbar = None
        if kind == "matrix":
            self.colorbar = ColorBar(self, cmap)
            self.colorbar.pack(fill="x", padx=10, pady=(8, 0))
        self.foot = tk.Label(self, text="", bg=CARD, fg=DIM,
                             font=(FONT_MONO, 8), anchor="w")
        self.foot.pack(fill="x", padx=10, pady=(4, 8))

    def set_title(self, title, subtitle=None):
        self.title_lbl.config(text=title)
        if subtitle is not None:
            self.sub_lbl.config(text=subtitle)

    def set_footer(self, text):
        self.foot.config(text=text)