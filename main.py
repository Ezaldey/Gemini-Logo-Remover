#!/usr/bin/env python3
"""
AI Logo Remover
================
A small desktop app (Tkinter) that lets you open an image, mark a logo /
watermark region — either automatically (heuristic corner scan) or manually
(click-drag a box) — and removes it using OpenCV inpainting.

Requirements:
    pip install opencv-python pillow numpy
Optional (enables real drag-and-drop of files onto the window):
    pip install tkinterdnd2

Run:
    python logo_remover.py

Notes on "auto-detect":
    There is no public model that specifically recognizes the Gemini logo,
    and shipping one (or embedding the real logo image) here would be a
    trademark/copyright problem regardless. What IS reliable is a general
    watermark heuristic: it looks in the four corners of the image for a
    small, colorful, high-edge-density blob sitting on a comparatively
    plain background (the pattern almost every app/AI logo watermark
    follows). It works well for typical bottom-right/corner logos but is
    not perfect — that's why manual selection is always available as a
    100%-accurate fallback.
"""

import sys
import os

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
except ImportError as e:
    print("Missing dependency:", e)
    print("Install requirements with:\n    pip install opencv-python pillow numpy")
    sys.exit(1)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Optional drag-and-drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False


# ---------------------------------------------------------------------------
# Color palette / theme
# ---------------------------------------------------------------------------
BG_DARK = "#14141c"
BG_PANEL = "#1e1e2b"
BG_CARD = "#262639"
ACCENT_1 = "#5b8def"   # blue
ACCENT_2 = "#a566ff"   # purple
ACCENT_GOOD = "#3ddc97"
ACCENT_WARN = "#ffb454"
TEXT_MAIN = "#f2f2f7"
TEXT_MUTED = "#9494a8"
BORDER = "#33334a"

MAX_PREVIEW = 480  # max side length for on-screen preview


# ---------------------------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------------------------
def analyze_region_score(roi_bgr):
    """Score a region on how 'watermark-logo-like' it is.

    Combines edge density and color/saturation variance — logos tend to be
    compact shapes with defined edges and noticeable color, unlike plain
    photo background.
    """
    if roi_bgr.size == 0:
        return 0.0, None

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)

    # Dilate edges a bit so nearby strokes merge into one blob
    kernel = np.ones((5, 5), np.uint8)
    edges_d = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, None

    # Pick the largest contour that isn't basically the whole ROI (that
    # would just be a busy photo, not a compact logo)
    roi_area = roi_bgr.shape[0] * roi_bgr.shape[1]
    best_c, best_area = None, 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 40:
            continue
        if area > roi_area * 0.55:
            continue
        if area > best_area:
            best_area = area
            best_c = c

    if best_c is None:
        return 0.0, None

    x, y, w, h = cv2.boundingRect(best_c)
    bbox_roi = roi_bgr[y:y + h, x:x + w]
    if bbox_roi.size == 0:
        return 0.0, None

    hsv = cv2.cvtColor(bbox_roi, cv2.COLOR_BGR2HSV)
    sat_std = float(np.std(hsv[:, :, 1]))
    edge_density = float(np.count_nonzero(edges[y:y + h, x:x + w])) / max(w * h, 1)

    score = sat_std * 0.6 + edge_density * 400
    return score, (x, y, w, h)


def detect_logo(cv_img, corner_fraction=0.30):
    """Scan the four corners of the image, return the best-scoring bbox
    (in full-image coordinates) plus its score, or (None, 0) if nothing
    stood out."""
    h, w = cv_img.shape[:2]
    cs_h = max(int(h * corner_fraction), 40)
    cs_w = max(int(w * corner_fraction), 40)

    corners = {
        "top-left": (0, 0),
        "top-right": (w - cs_w, 0),
        "bottom-left": (0, h - cs_h),
        "bottom-right": (w - cs_w, h - cs_h),
    }

    best_bbox, best_score = None, 0.0
    for _, (cx, cy) in corners.items():
        roi = cv_img[cy:cy + cs_h, cx:cx + cs_w]
        score, bbox = analyze_region_score(roi)
        if bbox is None:
            continue
        if score > best_score:
            best_score = score
            bx, by, bw, bh = bbox
            best_bbox = (cx + bx, cy + by, bw, bh)

    return best_bbox, best_score


def refine_bbox_in_region(cv_img, rect, min_score=3.0):
    """Given a user-dragged rectangle, look inside it for a compact,
    logo-shaped blob and return a tighter bbox around just that shape.

    Falls back to the original rectangle (matched=False) if nothing
    distinct enough is found inside it, e.g. when the user already drew
    a tight box, or the area is a uniform patch of background.
    """
    x, y, w, h = rect
    roi = cv_img[y:y + h, x:x + w]
    score, bbox = analyze_region_score(roi)

    if bbox is None or score < min_score:
        return rect, False

    bx, by, bw, bh = bbox
    # If the detected shape basically fills the whole dragged box, there's
    # nothing meaningful to tighten — just use the original rectangle.
    if bw >= w * 0.92 and bh >= h * 0.92:
        return rect, False

    return (x + bx, y + by, bw, bh), True


def remove_region(cv_img, bbox, padding=8, radius=9):
    """Inpaint over bbox (x, y, w, h) with some padding, feathering the
    mask edge slightly for a smoother blend."""
    x, y, w, h = bbox
    H, W = cv_img.shape[:2]
    x0, y0 = max(x - padding, 0), max(y - padding, 0)
    x1, y1 = min(x + w + padding, W), min(y + h + padding, H)

    mask = np.zeros((H, W), np.uint8)
    mask[y0:y1, x0:x1] = 255
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)

    result = cv2.inpaint(cv_img, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    return result


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class LogoRemoverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Logo Remover")
        self.root.geometry("1040x680")
        self.root.minsize(860, 560)
        self.root.configure(bg=BG_DARK)

        self.history = []            # list of image states (BGR numpy arrays)
        self.history_index = -1      # pointer into self.history (current state)
        self.file_path = None
        self.scale = 1.0             # canvas <-> image scale factor
        self.bbox_image_coords = None  # last detected/selected bbox (image px)
        self.img_origin = (0, 0)
        self._scrollregion = (0, 0, 1, 1)

        self.sel_start = None
        self.sel_rect_id = None

        self._build_style()
        self._build_layout()

        self.root.bind("<Control-z>", lambda _e: self.undo())
        self.root.bind("<Control-y>", lambda _e: self.redo())
        self.root.bind("<Control-Shift-Z>", lambda _e: self.redo())

        if DND_AVAILABLE:
            self.canvas.drop_target_register(DND_FILES)
            self.canvas.dnd_bind("<<Drop>>", self._on_drop)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=BG_DARK)
        style.configure("Panel.TFrame", background=BG_PANEL)
        style.configure("TLabel", background=BG_DARK, foreground=TEXT_MAIN,
                         font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=BG_DARK, foreground=TEXT_MUTED,
                         font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=BG_DARK, foreground=TEXT_MAIN,
                         font=("Segoe UI Semibold", 15))
        style.configure("Status.TLabel", background=BG_PANEL, foreground=TEXT_MUTED,
                         font=("Segoe UI", 9))

        style.configure("Accent.TButton", font=("Segoe UI Semibold", 10),
                         padding=(14, 9), background=ACCENT_1, foreground="white",
                         borderwidth=0)
        style.map("Accent.TButton",
                  background=[("active", ACCENT_2), ("disabled", BORDER)])

        style.configure("Ghost.TButton", font=("Segoe UI", 10),
                         padding=(12, 8), background=BG_CARD, foreground=TEXT_MAIN,
                         borderwidth=0)
        style.map("Ghost.TButton",
                  background=[("active", BORDER), ("disabled", BG_CARD)])

        style.configure("Good.TButton", font=("Segoe UI Semibold", 10),
                         padding=(14, 9), background=ACCENT_GOOD, foreground="#08281c",
                         borderwidth=0)
        style.map("Good.TButton",
                  background=[("active", "#33c184"), ("disabled", BORDER)])

        style.configure("TScrollbar", background=BG_CARD, troughcolor=BG_PANEL,
                         bordercolor=BG_PANEL, arrowcolor=TEXT_MUTED, borderwidth=0)
        style.map("TScrollbar", background=[("active", BORDER)])

    def _build_layout(self):
        # Header
        header = tk.Frame(self.root, bg=BG_DARK)
        header.pack(fill="x", padx=24, pady=(20, 6))
        tk.Label(header, text="AI Logo Remover", bg=BG_DARK, fg=TEXT_MAIN,
                  font=("Segoe UI Semibold", 18)).pack(side="left")
        tag = tk.Label(header, text="detect  •  select  •  erase", bg=BG_DARK,
                        fg=TEXT_MUTED, font=("Segoe UI", 10))
        tag.pack(side="left", padx=14)

        # Toolbar
        toolbar = tk.Frame(self.root, bg=BG_DARK)
        toolbar.pack(fill="x", padx=24, pady=(4, 12))

        self.btn_open = ttk.Button(toolbar, text="📂 Open Image", style="Accent.TButton",
                                    command=self.open_image)
        self.btn_open.pack(side="left", padx=(0, 8))

        self.btn_detect = ttk.Button(toolbar, text="✨ Auto-Detect", style="Ghost.TButton",
                                      command=self.run_auto_detect, state="disabled")
        self.btn_detect.pack(side="left", padx=8)

        self.btn_clear_sel = ttk.Button(toolbar, text="✕ Clear Selection", style="Ghost.TButton",
                                         command=self.clear_selection, state="disabled")
        self.btn_clear_sel.pack(side="left", padx=8)

        self.btn_remove = ttk.Button(toolbar, text="🧽 Remove Logo", style="Good.TButton",
                                      command=self.run_remove, state="disabled")
        self.btn_remove.pack(side="left", padx=8)

        self.btn_undo = ttk.Button(toolbar, text="↶ Undo", style="Ghost.TButton",
                                    command=self.undo, state="disabled")
        self.btn_undo.pack(side="left", padx=8)

        self.btn_redo = ttk.Button(toolbar, text="↷ Redo", style="Ghost.TButton",
                                    command=self.redo, state="disabled")
        self.btn_redo.pack(side="left", padx=(0, 8))

        self.btn_save = ttk.Button(toolbar, text="💾 Save Result", style="Ghost.TButton",
                                    command=self.save_result, state="disabled")
        self.btn_save.pack(side="right")

        # Zoom / pan control row
        zoom_bar = tk.Frame(self.root, bg=BG_DARK)
        zoom_bar.pack(fill="x", padx=24, pady=(0, 10))

        tk.Label(zoom_bar, text="Left-drag: select area    •    Right-drag: pan    •    "
                                 "Scroll: zoom",
                 bg=BG_DARK, fg=TEXT_MUTED, font=("Segoe UI", 9)).pack(side="left")

        zoom_controls = tk.Frame(zoom_bar, bg=BG_DARK)
        zoom_controls.pack(side="right")

        self.btn_zoom_out = ttk.Button(zoom_controls, text="–", style="Ghost.TButton",
                                        width=3, command=self.zoom_out, state="disabled")
        self.btn_zoom_out.pack(side="left", padx=(0, 4))

        self.zoom_label_var = tk.StringVar(value="100%")
        tk.Label(zoom_controls, textvariable=self.zoom_label_var, bg=BG_DARK,
                  fg=TEXT_MAIN, font=("Segoe UI Semibold", 9), width=6, anchor="center"
                  ).pack(side="left")

        self.btn_zoom_in = ttk.Button(zoom_controls, text="+", style="Ghost.TButton",
                                       width=3, command=self.zoom_in, state="disabled")
        self.btn_zoom_in.pack(side="left", padx=4)

        self.btn_zoom_fit = ttk.Button(zoom_controls, text="⤢ Fit", style="Ghost.TButton",
                                        command=self.zoom_fit, state="disabled")
        self.btn_zoom_fit.pack(side="left", padx=(8, 0))

        # Canvas card (with scrollbars so zoomed-in images can be panned)
        card = tk.Frame(self.root, bg=BG_PANEL, highlightbackground=BORDER,
                         highlightthickness=1)
        card.pack(fill="both", expand=True, padx=24, pady=(0, 8))
        card.rowconfigure(0, weight=1)
        card.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(card, bg=BG_CARD, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(16, 0), pady=(16, 0))

        vbar = ttk.Scrollbar(card, orient="vertical", command=self.canvas.yview)
        vbar.grid(row=0, column=1, sticky="ns", pady=(16, 0))
        hbar = ttk.Scrollbar(card, orient="horizontal", command=self.canvas.xview)
        hbar.grid(row=1, column=0, sticky="ew", padx=(16, 0))
        self.canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

        self.placeholder_text = self.canvas.create_text(
            0, 0, text="", fill=TEXT_MUTED, font=("Segoe UI", 12), justify="center")
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Left-drag: select area to remove
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        # Right-drag: pan around when zoomed in
        self.canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_move)

        # Scroll wheel: zoom in/out centered on the cursor
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)      # Windows / macOS
        self.canvas.bind("<Button-4>", self._on_mousewheel)        # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_mousewheel)        # Linux scroll down

        # Status bar
        status_bar = tk.Frame(self.root, bg=BG_PANEL, highlightbackground=BORDER,
                               highlightthickness=1)
        status_bar.pack(fill="x", padx=24, pady=(0, 20))
        self.status_var = tk.StringVar(
            value="Open an image, then drag a box over the logo (or try Auto-Detect).")
        ttk.Label(status_bar, textvariable=self.status_var, style="Status.TLabel"
                  ).pack(side="left", padx=12, pady=8)

        self._update_placeholder()

    def _update_placeholder(self):
        msg = "Drag & drop an image here\n(or click Open Image)" if DND_AVAILABLE else \
              "Click 'Open Image' to begin"
        self.canvas.itemconfig(self.placeholder_text, text=msg)
        self._center_placeholder()

    def _center_placeholder(self):
        w = self.canvas.winfo_width() or 400
        h = self.canvas.winfo_height() or 300
        self.canvas.coords(self.placeholder_text, w // 2, h // 2)

    def _on_canvas_resize(self, _event):
        if self._current_image() is None:
            self._center_placeholder()
        else:
            self._render_current()

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------
    def _on_drop(self, event):
        path = event.data.strip("{}")
        if os.path.isfile(path):
            self._load_image(path)

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")])
        if path:
            self._load_image(path)

    def _load_image(self, path):
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", "Could not read that image file.")
            return

        self.file_path = path
        self.history = [img]
        self.history_index = 0
        self.bbox_image_coords = None

        self.btn_detect.config(state="normal")
        self.btn_remove.config(state="disabled")
        self.btn_save.config(state="normal")
        self.btn_clear_sel.config(state="disabled")
        self.btn_zoom_in.config(state="normal")
        self.btn_zoom_out.config(state="normal")
        self.btn_zoom_fit.config(state="normal")
        self._update_undo_redo_buttons()

        self.status_var.set(f"Loaded {os.path.basename(path)} — "
                             f"{img.shape[1]}×{img.shape[0]}px. Drag a box over the logo, "
                             f"or try Auto-Detect.")
        self.root.update_idletasks()
        self._compute_fit_scale()
        self._render_current()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self._update_zoom_label()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _current_image(self):
        if 0 <= self.history_index < len(self.history):
            return self.history[self.history_index]
        return None

    def _current_display_image(self):
        return self._current_image()

    def _compute_fit_scale(self):
        """Set self.scale so the current image fits inside the visible canvas."""
        img = self._current_display_image()
        if img is None:
            return
        h, w = img.shape[:2]
        canvas_w = max(self.canvas.winfo_width(), 100)
        canvas_h = max(self.canvas.winfo_height(), 100)
        scale = min((canvas_w - 20) / w, (canvas_h - 20) / h)
        self.scale = max(min(scale, 8.0), 0.05)

    def _render_current(self):
        img = self._current_display_image()
        if img is None:
            return

        self.canvas.itemconfig(self.placeholder_text, text="")
        self.canvas.delete("img")
        canvas_w = max(self.canvas.winfo_width(), 100)
        canvas_h = max(self.canvas.winfo_height(), 100)

        h, w = img.shape[:2]
        disp_w, disp_h = max(int(w * self.scale), 1), max(int(h * self.scale), 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((disp_w, disp_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(pil_img)

        # Center the image when it's smaller than the viewport; otherwise pin
        # it at the scrollregion origin and let the scrollbars handle the rest.
        origin_x = (canvas_w - disp_w) // 2 if disp_w <= canvas_w else 0
        origin_y = (canvas_h - disp_h) // 2 if disp_h <= canvas_h else 0
        self.img_origin = (origin_x, origin_y)

        self.canvas.create_image(origin_x, origin_y, anchor="nw",
                                  image=self.tk_img, tags="img")
        self.canvas.tag_lower("img")

        region_w = max(disp_w, canvas_w)
        region_h = max(disp_h, canvas_h)
        self._scrollregion = (0, 0, region_w, region_h)
        self.canvas.configure(scrollregion=self._scrollregion)

        self._draw_selection_box()
        self._update_zoom_label()

    def _draw_selection_box(self):
        self.canvas.delete("selbox")
        if self.bbox_image_coords is None:
            return
        x, y, w, h = self.bbox_image_coords
        ox, oy = self.img_origin
        x0 = ox + x * self.scale
        y0 = oy + y * self.scale
        x1 = ox + (x + w) * self.scale
        y1 = oy + (y + h) * self.scale
        self.canvas.create_rectangle(x0, y0, x1, y1, outline=ACCENT_2, width=2,
                                      dash=(6, 3), tags="selbox")

    # ------------------------------------------------------------------
    # Manual selection (mouse drag)
    # ------------------------------------------------------------------
    def _on_mouse_down(self, event):
        if self._current_image() is None:
            return
        self.sel_start = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))

    def _on_mouse_drag(self, event):
        if self.sel_start is None:
            return
        self.canvas.delete("dragbox")
        x0, y0 = self.sel_start
        x1, y1 = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.canvas.create_rectangle(x0, y0, x1, y1, outline=ACCENT_1,
                                      width=2, tags="dragbox")

    def _on_mouse_up(self, event):
        img = self._current_image()
        if self.sel_start is None or img is None:
            return
        x0, y0 = self.sel_start
        x1, y1 = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.sel_start = None
        self.canvas.delete("dragbox")

        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            return  # ignore accidental clicks

        ox, oy = self.img_origin
        img_h, img_w = img.shape[:2]

        ix0 = max(min(x0, x1) - ox, 0) / self.scale
        iy0 = max(min(y0, y1) - oy, 0) / self.scale
        ix1 = min((max(x0, x1) - ox) / self.scale, img_w)
        iy1 = min((max(y0, y1) - oy) / self.scale, img_h)

        bw, bh = ix1 - ix0, iy1 - iy0
        if bw <= 2 or bh <= 2:
            return

        raw_bbox = (int(ix0), int(iy0), int(bw), int(bh))

        # Try to find the actual logo within the box the user drew, rather
        # than treating the whole dragged rectangle as the thing to erase.
        refined_bbox, matched = refine_bbox_in_region(img, raw_bbox)
        self.bbox_image_coords = refined_bbox

        self.btn_remove.config(state="normal")
        self.btn_clear_sel.config(state="normal")
        if matched:
            self.status_var.set(
                "Found a logo-shaped region inside your selection — tightened the box "
                "to it. Click 'Remove Logo', or drag again to adjust.")
        else:
            self.status_var.set(
                "Couldn't isolate a distinct shape inside the selection, so the full "
                "box will be removed. Click 'Remove Logo', or drag again to adjust.")
        self._draw_selection_box()

    # ------------------------------------------------------------------
    # Panning (right-click drag)
    # ------------------------------------------------------------------
    def _on_pan_start(self, event):
        if self._current_image() is None:
            return
        self.canvas.scan_mark(event.x, event.y)

    def _on_pan_move(self, event):
        if self._current_image() is None:
            return
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------
    def _update_zoom_label(self):
        self.zoom_label_var.set(f"{round(self.scale * 100)}%")

    def _zoom_at(self, factor, viewport_x, viewport_y):
        if self._current_image() is None:
            return
        old_scale = self.scale
        new_scale = max(min(old_scale * factor, 8.0), 0.05)
        if abs(new_scale - old_scale) < 1e-6:
            return

        # Image-space point currently under the cursor/anchor
        canvas_x = self.canvas.canvasx(viewport_x)
        canvas_y = self.canvas.canvasy(viewport_y)
        img_x = (canvas_x - self.img_origin[0]) / old_scale
        img_y = (canvas_y - self.img_origin[1]) / old_scale

        self.scale = new_scale
        self._render_current()

        # Scroll so that same image point stays under the cursor/anchor
        new_canvas_x = self.img_origin[0] + img_x * new_scale
        new_canvas_y = self.img_origin[1] + img_y * new_scale
        region_w = max(self._scrollregion[2], 1)
        region_h = max(self._scrollregion[3], 1)
        frac_x = (new_canvas_x - viewport_x) / region_w
        frac_y = (new_canvas_y - viewport_y) / region_h
        self.canvas.xview_moveto(max(0.0, min(frac_x, 1.0)))
        self.canvas.yview_moveto(max(0.0, min(frac_y, 1.0)))

    def zoom_in(self):
        self._zoom_at(1.25, self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)

    def zoom_out(self):
        self._zoom_at(0.8, self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)

    def zoom_fit(self):
        if self._current_image() is None:
            return
        self._compute_fit_scale()
        self._render_current()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def _on_mousewheel(self, event):
        if self._current_image() is None:
            return
        if getattr(event, "num", None) == 4:
            factor = 1.1
        elif getattr(event, "num", None) == 5:
            factor = 1 / 1.1
        elif getattr(event, "delta", 0) > 0:
            factor = 1.1
        else:
            factor = 1 / 1.1
        self._zoom_at(factor, event.x, event.y)

    def clear_selection(self):
        self.bbox_image_coords = None
        self.btn_remove.config(state="disabled")
        self.btn_clear_sel.config(state="disabled")
        self.canvas.delete("selbox")
        self.status_var.set("Selection cleared. Drag a new box, or try Auto-Detect.")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def run_auto_detect(self):
        img = self._current_image()
        if img is None:
            return
        self.status_var.set("Scanning corners for a logo-like region…")
        self.root.update_idletasks()

        bbox, score = detect_logo(img)
        if bbox is None or score < 8:
            self.status_var.set(
                "Auto-detect couldn't confidently find a logo. Drag a box over it manually.")
            messagebox.showinfo(
                "No confident match",
                "Auto-detect didn't find a clear logo-like region.\n\n"
                "This heuristic works best for compact, colorful watermarks in a corner. "
                "For anything else, just drag a rectangle over the logo yourself — "
                "that always works.")
            return

        self.bbox_image_coords = bbox
        self.btn_remove.config(state="normal")
        self.btn_clear_sel.config(state="normal")
        self.status_var.set(
            f"Found a likely logo region (confidence score {score:.1f}). "
            f"Adjust by dragging a new box if needed, or click Remove Logo.")
        self._draw_selection_box()

    def run_remove(self):
        img = self._current_image()
        if img is None or self.bbox_image_coords is None:
            return
        self.status_var.set("Removing region and reconstructing background…")
        self.root.update_idletasks()

        try:
            result = remove_region(img, self.bbox_image_coords)
        except Exception as exc:
            messagebox.showerror("Error", f"Removal failed: {exc}")
            return

        # Push the new state onto the undo/redo history, discarding any
        # states that were undone past this point.
        self.history = self.history[:self.history_index + 1]
        self.history.append(result)
        self.history_index += 1

        self.bbox_image_coords = None
        self.btn_remove.config(state="disabled")
        self.btn_clear_sel.config(state="disabled")
        self.btn_save.config(state="normal")
        self._update_undo_redo_buttons()
        self.status_var.set(
            "Logo removed. Select another area to remove more, or Undo to go back.")
        self._render_current()

    def undo(self):
        if self.history_index <= 0:
            return
        self.history_index -= 1
        self.bbox_image_coords = None
        self.btn_remove.config(state="disabled")
        self.btn_clear_sel.config(state="disabled")
        self._update_undo_redo_buttons()
        self.status_var.set("Undid last change.")
        self._render_current()

    def redo(self):
        if self.history_index >= len(self.history) - 1:
            return
        self.history_index += 1
        self.bbox_image_coords = None
        self.btn_remove.config(state="disabled")
        self.btn_clear_sel.config(state="disabled")
        self._update_undo_redo_buttons()
        self.status_var.set("Redid change.")
        self._render_current()

    def _update_undo_redo_buttons(self):
        self.btn_undo.config(state="normal" if self.history_index > 0 else "disabled")
        self.btn_redo.config(
            state="normal" if self.history_index < len(self.history) - 1 else "disabled")

    def save_result(self):
        img = self._current_image()
        if img is None:
            return
        base = os.path.splitext(os.path.basename(self.file_path or "image"))[0]
        path = filedialog.asksaveasfilename(
            title="Save result as",
            initialfile=f"{base}_cleaned.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All files", "*.*")])
        if not path:
            return
        ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
        if ok:
            buf.tofile(path)
            self.status_var.set(f"Saved to {path}")
        else:
            messagebox.showerror("Error", "Could not save the image.")


def main():
    root = TkinterDnD.Tk() if DND_AVAILABLE else tk.Tk()
    LogoRemoverApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
