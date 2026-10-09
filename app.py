"""Tk desktop UI; long-running operations stay off the UI thread."""
from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from core import (VERSION, Settings, evaluate_model, label_template, process_batch,
                  read_cloud, train_model)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"MOST Robotics | YellowScan Object Detection | {VERSION}")
        self.geometry("1320x900")
        self.minsize(1200, 750)
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
        self.area = tk.StringVar(value="unresolved")
        self.build_ui()
        self.after(100, self.poll)
        self.protocol("WM_DELETE_WINDOW", self.close_app)

    def build_ui(self):
        top = ttk.Frame(self, padding=12)
        top.pack(fill="x")
        ttk.Label(top, text="YellowScan Underwater Object Detection", font=("Arial", 17, "bold")).pack(anchor="w")
        ttk.Label(top, text="Research prototype | No pretrained underwater model | Full waveform not integrated").pack(anchor="w", pady=(2, 9))
        notebooks = ttk.Notebook(self)
        notebooks.pack(fill="both", expand=True, padx=12)
        processing, training = ttk.Frame(notebooks, padding=10), ttk.Frame(notebooks, padding=10)
        notebooks.add(processing, text="Process and View")
        notebooks.add(training, text="Train and Evaluate")
        self.path_row(processing, "Trusted MOST model", self.model_path, self.pick_model, 0)
        self.path_row(processing, "Results directory", self.output_path, self.pick_output, 1)
        controls = ttk.Frame(processing)
        controls.grid(row=2, column=0, columnspan=3, sticky="ew", pady=6)
        ttk.Button(controls, text="Add LAS / LAZ", command=self.add_files).pack(side="left")
        ttk.Button(controls, text="Clear list", command=self.clear_files).pack(side="left", padx=5)
        ttk.Button(controls, text="Inspect input", command=self.inspect).pack(side="left", padx=5)
        ttk.Button(controls, text="Process single / batch", command=self.process).pack(side="left", padx=5)
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
        self.view_note = ttk.Label(processing, text="Click View after processing. Display: up to 30,000 points; exports retain all points.")
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
        ttk.Label(training, text="Labels CSV: point_index,class_id. 0=background; positive=target; blank/-1=unlabelled.\n"
                  "Split independent scenes into train / validation / test; duplicate files and scene IDs are checked.\n"
                  "Prepare annotations externally, e.g. with CloudCompare. No annotation editor is included.",
                  justify="left").grid(row=1, column=0, columnspan=3, sticky="w", pady=15)
        row = ttk.Frame(training)
        row.grid(row=2, column=0, columnspan=3, sticky="w")
        ttk.Label(row, text="Method").pack(side="left")
        ttk.Combobox(row, textvariable=self.backend, values=["random_forest", "mlp"], state="readonly", width=20).pack(side="left", padx=5)
        ttk.Label(row, text="Trees / maximum iterations").pack(side="left", padx=8)
        ttk.Entry(row, textvariable=self.iterations, width=8).pack(side="left")
        buttons = ttk.Frame(training)
        buttons.grid(row=3, column=0, columnspan=3, sticky="w", pady=12)
        ttk.Button(buttons, text="Create label template", command=self.make_labels).pack(side="left")
        ttk.Button(buttons, text="Train and save model", command=self.train).pack(side="left", padx=8)
        ttk.Button(buttons, text="Evaluate held-out test set", command=self.evaluate).pack(side="left", padx=8)
        ttk.Label(training, text="Random Forest: default CPU baseline.\n"
                  "MLP: a three-hidden-layer feature network, not PointNet or Pointcept.\n"
                  "Validation metrics are saved with the model. Test labels are read only for explicit evaluation.\n"
                  "Per-class DBSCAN converts point predictions into instances; touching objects may merge.",
                  justify="left").grid(row=4, column=0, columnspan=3, sticky="w", pady=10)
        training.columnconfigure(1, weight=1)
        settings = ttk.LabelFrame(self, text="Coordinates and processing settings (all X/Y/Z coordinates must be in metres)", padding=8)
        settings.pack(fill="x", padx=12, pady=8)
        self.field(settings, "CRS if missing", self.crs, 0, 0, 18)
        self.field(settings, "Vertical datum", self.datum, 0, 2, 18)
        ttk.Checkbutton(settings, text="Confirm X/Y/Z are in metres", variable=self.meters).grid(row=0, column=4, columnspan=4, sticky="w")
        self.field(settings, "Score threshold", self.threshold, 1, 0, 8)
        self.field(settings, "Cluster radius m", self.radius, 1, 2, 8)
        self.field(settings, "Min object points", self.minimum, 1, 4, 8)
        ttk.Label(settings, text="Area method").grid(row=1, column=6, padx=6)
        ttk.Combobox(settings, textvariable=self.area, values=["unresolved", "convex_hull_xy"], state="readonly", width=19).grid(row=1, column=7)
        self.field(settings, "Neighbours", self.neighbors, 2, 0, 8)
        self.field(settings, "Max points / file", self.limit, 2, 2, 10)
        ttk.Label(settings, text="unresolved: blank area; convex_hull_xy: XY convex hull (agreement required)").grid(row=2, column=4, columnspan=4, sticky="w")
        self.log_box = tk.Text(self, height=5, state="disabled", wrap="word")
        self.log_box.pack(fill="x", padx=12)
        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.pack(fill="x", padx=12, pady=6)

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
            messagebox.showinfo("Processing", "Please wait for the current task to finish.")
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
        path = filedialog.askopenfilename(filetypes=[("MOST model (trusted only)", "*.joblib")])
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
            dest = filedialog.asksaveasfilename(defaultextension=".joblib", filetypes=[("MOST model", "*.joblib")])
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
            self.log("Synthetic demonstration only; this model has not been validated on real YellowScan data.")
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
        self.log(f"{len(self.datasets)} successful datasets, {failed} failed; see run_metadata.json")
        if self.datasets:
            self.dataset.current(0)
        else:
            self.dataset.set("")
            self.view_note.configure(text="No successful dataset to view; check the processing log.")

    def view(self):
        def action():
            if not self.run_path or not self.dataset.get():
                raise ValueError("Process a dataset or open saved results first")
            key = self.dataset.get().split(" | ", 1)[0]
            folder = self.run_path / self.datasets[key]["result_path"]
            self.current_objects = json.loads((folder / "objects.json").read_text(encoding="utf-8"))
            with np.load(folder / "view.npz", allow_pickle=False) as data:
                xyz, instance = data["xyz"], data["instance"]
                # Deterministic display subset; exports/metrics retain original points.
                indices = np.linspace(0, len(xyz) - 1, min(30000, len(xyz)), dtype=int)
                self.current_data = (xyz[indices], instance[indices], xyz.mean(axis=0))
            self.selected_object = None
            self.object_list.delete(0, "end")
            for obj in self.current_objects:
                self.object_list.insert("end", f"{obj['object_id']} ({obj['class_name']})")
            self.view_note.configure(text=f"{len(self.current_objects)} objects; scene-centred display. Left: rotate; middle: pan; right: zoom."
                                     if self.current_objects else "Processing succeeded; no objects detected.")
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
            messagebox.showinfo("Task still running", "Wait for the current task to finish before closing to avoid incomplete results.")
            return
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
