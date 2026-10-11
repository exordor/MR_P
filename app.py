"""Tk desktop UI; long-running operations stay off the UI thread."""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import tkinter as tk
import zipfile
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from core import (DISPLAY_LIMIT, Settings, display_indices, evaluate_model, label_template, process_batch,
                  read_cloud, train_model)


def tooltip(widget, text):
    """Show text in a small popup while the pointer is over widget."""
    tip = None

    def show(event):
        nonlocal tip
        tip = tk.Toplevel(widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{event.x_root + 12}+{event.y_root + 12}")
        ttk.Label(tip, text=text, background="#ffffe0", relief="solid", borderwidth=1,
                  padding=4, justify="left").pack()

    def hide(event):
        nonlocal tip
        if tip:
            tip.destroy()
            tip = None
    widget.bind("<Enter>", show)
    widget.bind("<Leave>", hide)


def use_bundled_tcl_tk():
    """Point Tk at the interpreter's own Tcl/Tk scripts when a venv hides them (e.g. uv-installed Python)."""
    lib = Path(sys.base_prefix) / "lib"
    for var, folder, script in (("TCL_LIBRARY", "tcl8.6", "init.tcl"), ("TK_LIBRARY", "tk8.6", "tk.tcl")):
        if var not in os.environ and (lib / folder / script).is_file():
            os.environ[var] = str(lib / folder)


class App(tk.Tk):
    def __init__(self):
        use_bundled_tcl_tk()
        super().__init__()
        self.title("Object Detection")
        height = min(1050, self.winfo_screenheight() - 80)
        self.geometry(f"1320x{height}")
        self.minsize(1200, min(700, height))
        self.events = queue.Queue()
        self.busy = False
        self.paths = []
        self.run_path = None
        self.current_data = None
        self.selected_object = None
        self.show_objects = tk.BooleanVar(value=False)
        self.meters = tk.BooleanVar(value=False)
        self.model_path = tk.StringVar()
        self.manifest_path = tk.StringVar()
        self.output_path = tk.StringVar(value=str((Path.cwd() / "results").resolve()))
        self.crs = tk.StringVar()
        self.datum = tk.StringVar(value="unspecified")
        self.threshold = tk.StringVar(value="0.7")
        self.radius = tk.StringVar(value="0.3")
        self.minimum = tk.StringVar(value="10")
        self.neighbors = tk.StringVar(value="20")
        self.limit = tk.StringVar(value="500000")
        self.backend = tk.StringVar(value="random_forest")
        self.iterations = tk.StringVar(value="100")
        self.display_limit = tk.StringVar(value=str(DISPLAY_LIMIT))
        self.area = tk.StringVar(value="unresolved")
        self.build_ui()
        self.after(100, self.poll)
        self.protocol("WM_DELETE_WINDOW", self.close_app)

    def build_ui(self):
        top = ttk.Frame(self, padding=12)
        top.pack(fill="x")
        ttk.Label(top, text="Object Detection", font=("Arial", 17, "bold")).pack(anchor="w")
        notebooks = ttk.Notebook(self)
        notebooks.pack(fill="both", expand=True, padx=12)
        processing, training = ttk.Frame(notebooks, padding=10), ttk.Frame(notebooks, padding=10)
        notebooks.add(processing, text="Process and View")
        notebooks.add(training, text="Train and Evaluate")
        self.path_row(processing, "Model", self.model_path, self.pick_model, 0)
        self.path_row(processing, "Results directory", self.output_path, self.pick_output, 1)
        controls = ttk.Frame(processing)
        controls.grid(row=2, column=0, columnspan=3, sticky="ew", pady=6)
        ttk.Button(controls, text="Add LAS / LAZ", command=self.add_files).pack(side="left")
        ttk.Button(controls, text="Clear list", command=self.clear_files).pack(side="left", padx=5)
        ttk.Button(controls, text="Inspect input", command=self.inspect).pack(side="left", padx=5)
        ttk.Button(controls, text="Process", command=self.process).pack(side="left", padx=5)
        ttk.Button(controls, text="Open saved results", command=self.open_run).pack(side="left", padx=5)
        ttk.Button(controls, text="Synthetic demo", command=self.open_demo).pack(side="left", padx=5)
        self.inputs = tk.Listbox(processing, height=3, exportselection=False)
        self.inputs.grid(row=3, column=0, columnspan=3, sticky="ew")
        viewer_controls = ttk.Frame(processing)
        viewer_controls.grid(row=4, column=0, columnspan=3, sticky="ew", pady=6)
        self.dataset = ttk.Combobox(viewer_controls, state="readonly", width=45)
        self.dataset.pack(side="left")
        ttk.Button(viewer_controls, text="View", command=self.view).pack(side="left", padx=8)
        ttk.Checkbutton(viewer_controls, text="Show objects", variable=self.show_objects,
                        command=self.redraw).pack(side="left")
        ttk.Label(viewer_controls, text="Display points").pack(side="left", padx=(12, 4))
        ttk.Combobox(viewer_controls, textvariable=self.display_limit,
                     values=["1000", "5000", "10000", "30000"], state="readonly", width=8).pack(side="left")
        ttk.Button(viewer_controls, text="Apply", command=self.view).pack(side="left", padx=4)
        self.view_note = ttk.Label(processing, text="")
        self.view_note.grid(row=5, column=0, columnspan=3, sticky="w")
        body = ttk.Frame(processing)
        body.grid(row=6, column=0, columnspan=3, sticky="nsew")
        self.figure = Figure(figsize=(7, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=body)
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
        self.object_list = tk.Listbox(body, width=28, exportselection=False)
        self.object_list.pack(side="right", fill="y")
        self.object_list.bind("<<ListboxSelect>>", self.select_object)
        toolbar_frame = ttk.Frame(processing)
        toolbar_frame.grid(row=7, column=0, columnspan=3, sticky="ew")
        NavigationToolbar2Tk(self.canvas, toolbar_frame)
        processing.columnconfigure(1, weight=1)
        processing.rowconfigure(6, weight=1)
        self.path_row(training, "Dataset manifest JSON", self.manifest_path, self.pick_manifest, 0)
        labels_hint = ttk.Label(training, text="Labels: point_index,class_id  (0 = background, -1 = unlabelled)")
        labels_hint.grid(row=1, column=0, columnspan=3, sticky="w", pady=15)
        tooltip(labels_hint, "One row per labelled point; positive IDs are target classes.\n"
                             "Train, validation and test must come from different scenes.")
        row = ttk.Frame(training)
        row.grid(row=2, column=0, columnspan=3, sticky="w")
        ttk.Label(row, text="Method").pack(side="left")
        method = ttk.Combobox(row, textvariable=self.backend, values=["random_forest", "mlp"], state="readonly", width=20)
        method.pack(side="left", padx=5)
        iterations_label = ttk.Label(row, text="Trees")
        iterations_label.pack(side="left", padx=8)
        method.bind("<<ComboboxSelected>>", lambda e: iterations_label.configure(
            text="Trees" if self.backend.get() == "random_forest" else "Max epochs"))
        ttk.Entry(row, textvariable=self.iterations, width=8).pack(side="left")
        buttons = ttk.Frame(training)
        buttons.grid(row=3, column=0, columnspan=3, sticky="w", pady=12)
        ttk.Button(buttons, text="Create label template", command=self.make_labels).pack(side="left")
        ttk.Button(buttons, text="Train and save model", command=self.train).pack(side="left", padx=8)
        ttk.Button(buttons, text="Evaluate on test set", command=self.evaluate).pack(side="left", padx=8)
        training.columnconfigure(1, weight=1)
        settings = ttk.LabelFrame(self, text="Settings", padding=8)
        # Docked to the bottom before the notebook, so a short window shrinks the plot instead of hiding these.
        settings.pack(side="bottom", fill="x", padx=12, pady=8, before=notebooks)
        self.field(settings, "Fallback CRS", self.crs, 0, 0, 18)
        self.field(settings, "Vertical datum", self.datum, 0, 2, 18)
        ttk.Checkbutton(settings, text="X/Y/Z in metres", variable=self.meters).grid(row=0, column=4, columnspan=4, sticky="w")
        self.field(settings, "Score threshold", self.threshold, 1, 0, 8)
        self.field(settings, "Cluster radius m", self.radius, 1, 2, 8)
        self.field(settings, "Min object points", self.minimum, 1, 4, 8)
        ttk.Label(settings, text="Area method").grid(row=1, column=6, padx=6)
        area = ttk.Combobox(settings, textvariable=self.area, values=["unresolved", "convex_hull_xy"], state="readonly", width=19)
        area.grid(row=1, column=7)
        tooltip(area, "unresolved: leave Area blank\nconvex_hull_xy: area of the XY convex hull")
        self.field(settings, "Neighbours", self.neighbors, 2, 0, 8)
        self.field(settings, "Max points / file", self.limit, 2, 2, 10)
        self.log_box = tk.Text(self, height=5, state="disabled", wrap="word")
        self.log_box.pack(side="bottom", fill="x", padx=12, before=settings)
        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.pack(side="bottom", fill="x", padx=12, pady=6, before=self.log_box)

    @staticmethod
    def field(parent, title, variable, row, col, width):
        ttk.Label(parent, text=title).grid(row=row, column=col, sticky="w", padx=4, pady=3)
        ttk.Entry(parent, textvariable=variable, width=width).grid(row=row, column=col + 1, sticky="w")

    @staticmethod
    def path_row(parent, title, variable, command, row):
        ttk.Label(parent, text=title).grid(row=row, column=0, sticky="w", padx=(0, 10))
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew")
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, padx=5)

    def settings(self):
        return Settings(neighbors=int(self.neighbors.get()), max_points=int(self.limit.get()),
                        threshold=float(self.threshold.get()), cluster_eps=float(self.radius.get()),
                        min_points=int(self.minimum.get()), area_method=self.area.get(),
                        crs_override=self.crs.get().strip(), coordinates_in_meters=self.meters.get(),
                        vertical_datum=self.datum.get().strip())

    def log(self, message):
        self.events.put(("log", str(message)))

    def job(self, fn, callback=None):
        if self.busy:
            messagebox.showinfo("Processing", "A task is already running.")
            return
        self.busy = True
        self.progress.start()
        def work():
            try:
                result = fn()
                self.events.put(("done", (result, callback)))
            except Exception as exc:
                self.events.put(("error", f"{type(exc).__name__}: {exc}"))
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "log":
                    self.log_box.configure(state="normal")
                    self.log_box.insert("end", value + "\n")
                    self.log_box.see("end")
                    self.log_box.configure(state="disabled")
                else:
                    self.busy = False
                    self.progress.stop()
                    if kind == "error":
                        self.log(value)
                        messagebox.showerror("Task failed", value)
                    else:
                        result, callback = value
                        self.log(f"Completed: {result}")
                        if callback:
                            callback(result)
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def pick_model(self):
        path = filedialog.askopenfilename(filetypes=[("Model", "*.joblib")])
        if path:
            self.model_path.set(path)

    def pick_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_path.set(path)

    def pick_manifest(self):
        path = filedialog.askopenfilename(filetypes=[("Dataset manifest", "*.json")])
        if path:
            self.manifest_path.set(path)

    def add_files(self):
        for path in filedialog.askopenfilenames(filetypes=[("Point clouds", "*.las *.laz")]):
            if path not in self.paths:
                self.paths.append(path)
                self.inputs.insert("end", path)

    def clear_files(self):
        self.paths = []
        self.inputs.delete(0, "end")

    def safe_action(self, action):
        try:
            action()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))

    def inspect(self):
        def action():
            if not self.paths:
                raise ValueError("Add a LAS / LAZ file first")
            cfg, path = self.settings(), self.paths[0]
            self.job(lambda: json.dumps(read_cloud(path, cfg)["info"], ensure_ascii=False, indent=2))
        self.safe_action(action)

    def process(self):
        def action():
            cfg, paths, model, output = self.settings(), self.paths.copy(), self.model_path.get(), self.output_path.get()
            if not model or not paths or not output:
                raise ValueError("Select input files, a model and a results directory")
            self.job(lambda: process_batch(paths, model, output, cfg, self.log), self.load_run)
        self.safe_action(action)

    def train(self):
        def action():
            cfg, manifest = self.settings(), self.manifest_path.get()
            backend, iterations = self.backend.get(), int(self.iterations.get())
            if not manifest:
                raise ValueError("Select a dataset manifest JSON")
            dest = filedialog.asksaveasfilename(defaultextension=".joblib", filetypes=[("Trained model", "*.joblib")])
            if dest:
                self.job(lambda: train_model(manifest, dest, cfg, backend, iterations, self.log), self.model_path.set)
        self.safe_action(action)

    def evaluate(self):
        def action():
            cfg, model = self.settings(), self.model_path.get()
            if not model:
                raise ValueError("Select a model")
            dest = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Evaluation", "*.json")])
            if dest:
                self.job(lambda: evaluate_model(model, dest, cfg, self.log))
        self.safe_action(action)

    def make_labels(self):
        def action():
            path = filedialog.askopenfilename(filetypes=[("Point cloud", "*.las *.laz")])
            if not path:
                return
            cfg = self.settings()
            dest = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=Path(path).stem + ".labels.csv")
            if dest:
                self.job(lambda: (label_template(path, dest, cfg), dest)[1])
        self.safe_action(action)

    def open_run(self):
        path = filedialog.askdirectory(title="Select a results directory containing run_metadata.json")
        if path:
            self.safe_action(lambda: self.load_run(path))

    def open_demo(self):
        def action():
            demo = Path(__file__).resolve().parent / "synthetic_demo"
            candidates = sorted((demo / "results").glob("*/run_metadata.json"))
            if not candidates:
                raise ValueError("Synthetic demo missing; follow README to generate it.")
            self.model_path.set(str(demo / "SYNTHETIC_MODEL.joblib"))
            self.meters.set(True)
            self.datum.set("synthetic_local_z")
            self.area.set("convex_hull_xy")
            self.clear_files()
            self.paths = [str(demo / "SYNTHETIC_test.laz")]
            self.inputs.insert("end", self.paths[0])
            self.load_run(candidates[-1].parent)
            self.show_objects.set(True)
            self.view()
            self.log("Loaded synthetic demo.")
        self.safe_action(action)

    def load_run(self, path):
        self.run_path = Path(path)
        metadata = json.loads((self.run_path / "run_metadata.json").read_text(encoding="utf-8"))
        self.datasets = {e["dataset_id"]: e for e in metadata["datasets"] if e["status"].startswith("success")}
        self.dataset["values"] = [f"{key} | {Path(e['source']).name}" for key, e in self.datasets.items()]
        self.current_data = None
        self.figure.clear()
        self.canvas.draw_idle()
        self.object_list.delete(0, "end")
        failed = sum(e["status"] == "failed" for e in metadata["datasets"])
        self.log(f"Loaded {len(self.datasets)} datasets, {failed} failed")
        if self.datasets:
            self.dataset.current(0)
        else:
            self.dataset.set("")
            self.view_note.configure(text="No datasets to view.")

    def view(self):
        def action():
            if not self.run_path or not self.dataset.get():
                raise ValueError("Process a dataset or open saved results first")
            key = self.dataset.get().split(" | ", 1)[0]
            folder = self.run_path / self.datasets[key]["result_path"]
            self.current_objects = json.loads((folder / "objects.json").read_text(encoding="utf-8"))
            cache = folder / "view.npz"
            # Reject oversized legacy caches before decompressing full arrays.
            with zipfile.ZipFile(cache) as archive:
                if sum(item.file_size for item in archive.infolist()) > 32 * 1024 * 1024:
                    raise ValueError("Saved display cache is too large. Reprocess with this version to create a downsampled cache.")
            with np.load(cache, allow_pickle=False) as data:
                xyz, instance = data["xyz"], data["instance"]
                indices = display_indices(instance, int(self.display_limit.get()))
                origin = data["origin"] if "origin" in data else xyz.mean(axis=0)
                total = int(data["total_points"]) if "total_points" in data else len(xyz)
                self.current_data = (xyz[indices], instance[indices], origin)
            self.selected_object = None
            self.object_list.delete(0, "end")
            for obj in self.current_objects:
                self.object_list.insert("end", f"{obj['object_id']} ({obj['class_name']})")
            self.view_note.configure(text=f"{len(indices):,} / {total:,} points shown · {len(self.current_objects)} objects")
            self.redraw()
        self.safe_action(action)

    def select_object(self, event=None):
        selection = self.object_list.curselection()
        if selection:
            obj = self.current_objects[selection[0]]
            self.selected_object = obj["instance_index"]
            self.show_objects.set(True)
            self.log(f"{obj['object_id']}: {obj['point_count']} points, confidence={obj['confidence']:.3f}")
            self.redraw()

    def redraw(self):
        if self.current_data is None:
            return
        xyz, instance, origin = self.current_data
        local = xyz - origin
        camera = None
        if self.figure.axes:
            old = self.figure.axes[0]
            camera = (old.elev, old.azim, old.roll)
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="3d")
        ax.mouse_init(rotate_btn=1, pan_btn=2, zoom_btn=3)
        if camera is not None:
            ax.view_init(*camera)
        colors = np.tile([0.58, 0.62, 0.67, 0.28], (len(xyz), 1))
        if self.show_objects.get():
            from matplotlib import colormaps
            for obj in self.current_objects:
                idx = obj["instance_index"]
                # Golden-angle hues keep distinct colours beyond the first 20 objects.
                color = colormaps["hsv"]((idx * 0.61803398875) % 1)
                colors[instance == idx] = color
            if self.selected_object:
                colors[(instance > 0) & (instance != self.selected_object), 3] = 0.18
        ax.scatter(*local.T, c=colors, s=3, depthshade=False)
        span = np.ptp(local, axis=0)
        ax.set_box_aspect(np.maximum(span, 0.1))
        ax.set(xlabel="X relative (m)", ylabel="Y relative (m)", zlabel="Z relative (m)")
        from matplotlib.ticker import MaxNLocator
        ax.zaxis.set_major_locator(MaxNLocator(nbins=2))
        ax.tick_params(labelsize=8, pad=1)
        if self.selected_object and self.show_objects.get():
            obj = next(o for o in self.current_objects if o["instance_index"] == self.selected_object)
            center = np.asarray(obj["centroid"]) - origin
            ax.text(*center, obj["object_id"], fontsize=8)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def close_app(self):
        if self.busy:
            messagebox.showinfo("Task running", "Wait for the current task to finish.")
            return
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
