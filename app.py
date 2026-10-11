"""PySide6 desktop UI with an embedded WebGL 3D view; long-running operations stay off the UI thread."""
from __future__ import annotations

import json
import sys
import threading
import webbrowser
from pathlib import Path

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, QSortFilterProxyModel, Qt, QUrl, Signal
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
                               QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSplitter,
                               QTableView, QTabWidget, QVBoxLayout, QWidget)

from core import (Settings, evaluate_model, label_template, process_batch, read_cloud, train_model,
                  write_web_view)


class ObjectTable(QAbstractTableModel):
    """Detected objects; numbers stay numeric so column sorting is numeric."""
    HEADERS = ["Object ID", "Class", "Points", "Confidence", "X", "Y", "Z"]

    def __init__(self):
        super().__init__()
        self.rows = []

    def load(self, objects):
        self.beginResetModel()
        self.rows = [(o["object_id"], o["class_name"], o["point_count"], o["confidence"], *o["centroid"])
                     for o in objects]
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return len(self.HEADERS)

    def data(self, index, role=Qt.DisplayRole):
        value = self.rows[index.row()][index.column()]
        column = index.column()
        if role == Qt.DisplayRole:
            if column == 2:
                return f"{value:,}"
            return f"{value:.3f}" if column >= 3 else value
        if role == Qt.UserRole:  # Sort key.
            return value
        if role == Qt.TextAlignmentRole and column >= 2:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]
        return None


class ViewerPage(QWebEnginePage):
    """Forwards 'PICK <object id>' console messages from the 3D page to the window."""

    def __init__(self, on_pick, parent=None):
        super().__init__(parent)
        self.on_pick = on_pick

    def javaScriptConsoleMessage(self, level, message, line, source):
        if message.startswith("PICK "):
            self.on_pick(message[5:])


class Bridge(QObject):
    """Carries worker-thread events to the UI thread (queued signal)."""
    event = Signal(str, object)


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection")
        self.resize(1400, 950)
        self.bridge = Bridge()
        self.bridge.event.connect(self.handle_event)
        self.busy = False
        self.paths = []
        self.run_path = None
        self.datasets = {}
        self.loaded_page = None
        self.pending_focus = None
        self.build_ui()

    # ---- layout -------------------------------------------------------------
    def build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        title = QLabel("Object Detection")
        title.setStyleSheet("font-size: 17pt; font-weight: bold;")
        layout.addWidget(title)
        tabs = QTabWidget()
        layout.addWidget(tabs, 1)
        tabs.addTab(self.build_processing(), "Process and View")
        tabs.addTab(self.build_training(), "Train and Evaluate")
        layout.addWidget(self.build_settings())
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        layout.addWidget(self.progress)
        self.log_box = QPlainTextEdit(readOnly=True)
        self.log_box.setFixedHeight(90)
        layout.addWidget(self.log_box)
        self.setCentralWidget(central)

    def path_row(self, grid, row, title, edit, command):
        grid.addWidget(QLabel(title), row, 0)
        grid.addWidget(edit, row, 1)
        button = QPushButton("Browse")
        button.clicked.connect(command)
        grid.addWidget(button, row, 2)

    def button(self, text, command, tip=None):
        button = QPushButton(text)
        button.clicked.connect(command)
        if tip:
            button.setToolTip(tip)
        return button

    def build_processing(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        grid = QGridLayout()
        self.model_path = QLineEdit()
        self.output_path = QLineEdit(str((Path.cwd() / "results").resolve()))
        self.path_row(grid, 0, "Model", self.model_path, self.pick_model)
        self.path_row(grid, 1, "Results directory", self.output_path, self.pick_output)
        layout.addLayout(grid)
        controls = QHBoxLayout()
        for text, command in (("Add LAS / LAZ", self.add_files), ("Clear list", self.clear_files),
                              ("Inspect input", self.inspect), ("Process", self.process),
                              ("Open saved results", self.open_run), ("Synthetic demo", self.open_demo)):
            controls.addWidget(self.button(text, command))
        controls.addStretch()
        layout.addLayout(controls)
        self.inputs = QListWidget()
        self.inputs.setFixedHeight(60)
        layout.addWidget(self.inputs)
        viewer_controls = QHBoxLayout()
        self.dataset = QComboBox()
        self.dataset.setMinimumWidth(380)
        self.dataset.currentIndexChanged.connect(lambda _: self.view())
        viewer_controls.addWidget(self.dataset)
        self.view_note = QLabel("")
        viewer_controls.addWidget(self.view_note)
        viewer_controls.addStretch()
        viewer_controls.addWidget(self.button("Open in browser", self.open_browser_view,
                                              "Open the same 3D view in your web browser, e.g. for a second screen."))
        layout.addLayout(viewer_controls)
        splitter = QSplitter(Qt.Horizontal)
        self.objects = ObjectTable()
        self.sorted_objects = QSortFilterProxyModel()
        self.sorted_objects.setSourceModel(self.objects)
        self.sorted_objects.setSortRole(Qt.UserRole)
        self.table = QTableView()
        self.table.setModel(self.sorted_objects)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(-1, Qt.AscendingOrder)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setToolTip("Click an object to show it in the 3D view. Click a heading to sort.")
        self.table.selectionModel().currentRowChanged.connect(self.table_selected)
        splitter.addWidget(self.table)
        self.web = QWebEngineView()
        # Keep a Python reference: Qt calls the overridden method, so the wrapper must outlive the view.
        self.viewer_page = ViewerPage(self.picked_in_view, self.web)
        self.web.setPage(self.viewer_page)
        self.web.loadFinished.connect(self.page_loaded)
        splitter.addWidget(self.web)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([380, 1000])
        layout.addWidget(splitter, 1)
        return page

    def build_training(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        grid = QGridLayout()
        self.manifest_path = QLineEdit()
        self.path_row(grid, 0, "Dataset manifest JSON", self.manifest_path, self.pick_manifest)
        layout.addLayout(grid)
        hint = QLabel("Labels: point_index,class_id  (0 = background, -1 = unlabelled)")
        hint.setToolTip("One row per labelled point; positive IDs are target classes.\n"
                        "Train, validation and test must come from different scenes.")
        layout.addWidget(hint)
        row = QHBoxLayout()
        row.addWidget(QLabel("Method"))
        self.backend = QComboBox()
        self.backend.addItems(["random_forest", "mlp"])
        row.addWidget(self.backend)
        iterations_label = QLabel("Trees")
        self.backend.currentTextChanged.connect(
            lambda text: iterations_label.setText("Trees" if text == "random_forest" else "Max epochs"))
        row.addWidget(iterations_label)
        self.iterations = QLineEdit("100")
        self.iterations.setFixedWidth(70)
        row.addWidget(self.iterations)
        row.addStretch()
        layout.addLayout(row)
        buttons = QHBoxLayout()
        for text, command in (("Create label template", self.make_labels), ("Train and save model", self.train),
                              ("Evaluate on test set", self.evaluate)):
            buttons.addWidget(self.button(text, command))
        buttons.addStretch()
        layout.addLayout(buttons)
        layout.addStretch()
        return page

    def build_settings(self):
        box = QGroupBox("Settings")
        grid = QGridLayout(box)
        self.crs, self.datum = QLineEdit(), QLineEdit("unspecified")
        self.meters = QCheckBox("X/Y/Z in metres")
        self.threshold, self.radius, self.minimum = QLineEdit("0.7"), QLineEdit("0.3"), QLineEdit("10")
        self.neighbors, self.limit = QLineEdit("20"), QLineEdit("0")
        self.limit.setToolTip("0 = no limit: the whole file is processed.\n"
                              "Very large files are refused only if they would not fit in memory.")
        self.area = QComboBox()
        self.area.addItems(["unresolved", "convex_hull_xy"])
        self.area.setToolTip("unresolved: leave Area blank\nconvex_hull_xy: area of the XY convex hull")
        fields = [[("Fallback CRS", self.crs), ("Vertical datum", self.datum), (None, self.meters)],
                  [("Score threshold", self.threshold), ("Cluster radius m", self.radius),
                   ("Min object points", self.minimum), ("Area method", self.area)],
                  [("Neighbours", self.neighbors), ("Max points / file", self.limit)]]
        for row, items in enumerate(fields):
            for col, (title, widget) in enumerate(items):
                if title:
                    grid.addWidget(QLabel(title), row, col * 2)
                grid.addWidget(widget, row, col * 2 + 1)
        grid.setColumnStretch(8, 1)
        return box

    # ---- tasks ----------------------------------------------------------------
    def settings(self):
        return Settings(neighbors=int(self.neighbors.text()), max_points=int(self.limit.text()),
                        threshold=float(self.threshold.text()), cluster_eps=float(self.radius.text()),
                        min_points=int(self.minimum.text()), area_method=self.area.currentText(),
                        crs_override=self.crs.text().strip(), coordinates_in_meters=self.meters.isChecked(),
                        vertical_datum=self.datum.text().strip())

    def log(self, message):
        """Thread-safe: may be called from the worker thread."""
        self.bridge.event.emit("log", str(message))

    def job(self, fn, callback=None):
        if self.busy:
            QMessageBox.information(self, "Processing", "A task is already running.")
            return
        self.busy = True
        self.progress.setRange(0, 0)  # Busy indicator.
        def work():
            try:
                self.bridge.event.emit("done", (fn(), callback))
            except Exception as exc:
                self.bridge.event.emit("error", f"{type(exc).__name__}: {exc}")
        threading.Thread(target=work, daemon=True).start()

    def handle_event(self, kind, value):
        if kind == "log":
            self.log_box.appendPlainText(value)
            return
        self.busy = False
        self.progress.setRange(0, 1)
        if kind == "error":
            self.log_box.appendPlainText(value)
            QMessageBox.critical(self, "Task failed", value)
        else:
            result, callback = value
            self.log_box.appendPlainText(f"Completed: {result}")
            if callback:
                self.safe_action(lambda: callback(result))

    def safe_action(self, action):
        try:
            action()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))

    def pick_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Model", "", "Model (*.joblib)")
        if path:
            self.model_path.setText(path)

    def pick_output(self):
        path = QFileDialog.getExistingDirectory(self, "Results directory")
        if path:
            self.output_path.setText(path)

    def pick_manifest(self):
        path, _ = QFileDialog.getOpenFileName(self, "Dataset manifest", "", "Dataset manifest (*.json)")
        if path:
            self.manifest_path.setText(path)

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Point clouds", "", "Point clouds (*.las *.laz)")
        for path in paths:
            if path not in self.paths:
                self.paths.append(path)
                self.inputs.addItem(path)

    def clear_files(self):
        self.paths = []
        self.inputs.clear()

    def inspect(self):
        def action():
            if not self.paths:
                raise ValueError("Add a LAS / LAZ file first")
            cfg, path = self.settings(), self.paths[0]
            self.job(lambda: json.dumps(read_cloud(path, cfg)["info"], ensure_ascii=False, indent=2))
        self.safe_action(action)

    def process(self):
        def action():
            cfg, paths = self.settings(), self.paths.copy()
            model, output = self.model_path.text(), self.output_path.text()
            if not model or not paths or not output:
                raise ValueError("Select input files, a model and a results directory")
            self.job(lambda: process_batch(paths, model, output, cfg, self.log), self.load_run)
        self.safe_action(action)

    def train(self):
        def action():
            cfg, manifest = self.settings(), self.manifest_path.text()
            backend, iterations = self.backend.currentText(), int(self.iterations.text())
            if not manifest:
                raise ValueError("Select a dataset manifest JSON")
            dest, _ = QFileDialog.getSaveFileName(self, "Trained model", "", "Trained model (*.joblib)")
            if dest:
                dest = dest if dest.endswith(".joblib") else dest + ".joblib"
                self.job(lambda: train_model(manifest, dest, cfg, backend, iterations, self.log),
                         self.model_path.setText)
        self.safe_action(action)

    def evaluate(self):
        def action():
            cfg, model = self.settings(), self.model_path.text()
            if not model:
                raise ValueError("Select a model")
            dest, _ = QFileDialog.getSaveFileName(self, "Evaluation", "", "Evaluation (*.json)")
            if dest:
                dest = dest if dest.endswith(".json") else dest + ".json"
                self.job(lambda: evaluate_model(model, dest, cfg, self.log))
        self.safe_action(action)

    def make_labels(self):
        def action():
            path, _ = QFileDialog.getOpenFileName(self, "Point cloud", "", "Point cloud (*.las *.laz)")
            if not path:
                return
            cfg = self.settings()
            dest, _ = QFileDialog.getSaveFileName(self, "Label template", Path(path).stem + ".labels.csv",
                                                  "CSV (*.csv)")
            if dest:
                self.job(lambda: (label_template(path, dest, cfg), dest)[1])
        self.safe_action(action)

    # ---- results and 3D view -----------------------------------------------------
    def open_run(self):
        path = QFileDialog.getExistingDirectory(self, "Select a results directory containing run_metadata.json")
        if path:
            self.safe_action(lambda: self.load_run(path))

    def open_demo(self):
        def action():
            demo = Path(__file__).resolve().parent / "synthetic_demo"
            candidates = sorted((demo / "results").glob("*/run_metadata.json"))
            if not candidates:
                raise ValueError("Synthetic demo missing; follow README to generate it.")
            self.model_path.setText(str(demo / "SYNTHETIC_MODEL.joblib"))
            self.meters.setChecked(True)
            self.datum.setText("synthetic_local_z")
            self.area.setCurrentText("convex_hull_xy")
            self.clear_files()
            self.paths = [str(demo / "SYNTHETIC_test.laz")]
            self.inputs.addItem(self.paths[0])
            self.load_run(candidates[-1].parent)
            self.log("Loaded synthetic demo.")
        self.safe_action(action)

    def load_run(self, path):
        self.run_path = Path(path)
        metadata = json.loads((self.run_path / "run_metadata.json").read_text(encoding="utf-8"))
        self.datasets = {e["dataset_id"]: e for e in metadata["datasets"] if e["status"].startswith("success")}
        failed = sum(e["status"] == "failed" for e in metadata["datasets"])
        self.log(f"Loaded {len(self.datasets)} datasets, {failed} failed")
        self.loaded_page = None
        self.dataset.blockSignals(True)
        self.dataset.clear()
        self.dataset.addItems([f"{key} | {Path(e['source']).name}" for key, e in self.datasets.items()])
        self.dataset.blockSignals(False)
        if self.datasets:
            self.view()
        else:
            self.objects.load([])
            self.web.setHtml("")
            self.view_note.setText("No datasets to view.")

    def selected_folder(self):
        if not self.run_path or not self.dataset.currentText():
            raise ValueError("Process a dataset or open saved results first")
        key = self.dataset.currentText().split(" | ", 1)[0]
        return key, self.run_path / self.datasets[key]["result_path"]

    def view_page(self, key, folder):
        page = folder / "view3d.html"
        if not page.exists():  # Results saved before the 3D page existed: build it from the display cache.
            objects = json.loads((folder / "objects.json").read_text(encoding="utf-8"))
            with np.load(folder / "view.npz", allow_pickle=False) as data:
                write_web_view(page, Path(self.datasets[key]["source"]).name, data["xyz"], data["instance"],
                               objects, data["origin"], int(data["total_points"]))
        return page

    def view(self):
        def action():
            key, folder = self.selected_folder()
            self.objects.load(json.loads((folder / "objects.json").read_text(encoding="utf-8")))
            total = self.datasets[key].get("input", {}).get("point_count")
            points = f"{total:,} points · " if total else ""
            self.view_note.setText(f"{points}{self.objects.rowCount():,} objects")
            page = self.view_page(key, folder)
            if page != self.loaded_page:
                self.loaded_page = page
                self.pending_focus = None
                # Loaded by URL: setHtml() is limited to 2 MB and the page can be tens of MB.
                self.web.load(QUrl.fromLocalFile(str(page.resolve())))
        self.safe_action(action)

    def page_loaded(self, ok):
        if ok and self.pending_focus:
            self.focus_in_view(self.pending_focus)

    def focus_in_view(self, object_id):
        self.pending_focus = object_id
        self.web.page().runJavaScript(f"window.focusObject && focusObject({json.dumps(object_id)})")

    def table_selected(self, current, previous):
        if current.isValid():
            source = self.sorted_objects.mapToSource(current)
            self.focus_in_view(self.objects.rows[source.row()][0])

    def picked_in_view(self, object_id):
        """A point clicked in the 3D view selects its object in the table (or clears the selection)."""
        self.table.selectionModel().blockSignals(True)
        self.table.clearSelection()
        for row, values in enumerate(self.objects.rows):
            if values[0] == object_id:
                index = self.sorted_objects.mapFromSource(self.objects.index(row, 0))
                self.table.selectRow(index.row())
                self.table.scrollTo(index)
                break
        self.table.selectionModel().blockSignals(False)

    def open_browser_view(self):
        def action():
            key, folder = self.selected_folder()
            webbrowser.open(self.view_page(key, folder).resolve().as_uri())
        self.safe_action(action)

    def closeEvent(self, event):
        if self.busy:
            QMessageBox.information(self, "Task running", "Wait for the current task to finish.")
            event.ignore()
            return
        event.accept()


def main():
    application = QApplication(sys.argv)
    window = App()
    window.show()
    return application.exec()


if __name__ == "__main__":
    sys.exit(main())
