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
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QListWidget, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QSplitter, QTableView, QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

from core import (Settings, evaluate_model, inspect_file, label_template, process_batch, train_model,
                  write_web_view)


def size_text(count):
    for unit in ("B", "KB", "MB", "GB"):
        if count < 1024 or unit == "GB":
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1024


class InspectDialog(QDialog):
    """Header summary of every input file; read-only, so it needs no settings."""
    HEADERS = ["File", "Points", "File size", "Width X m", "Width Y m", "Z min", "Z max", "CRS", "Memory needed",
               "Check"]

    def __init__(self, summaries, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Input files")
        layout = QVBoxLayout(self)
        table = QTableWidget(len(summaries), len(self.HEADERS))
        table.setHorizontalHeaderLabels(self.HEADERS)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for row, info in enumerate(summaries):
            if "error" in info:
                cells = [info["name"]] + [""] * 8 + [info["error"]]
            else:
                (x0, y0, z0), (x1, y1, z1) = info["mins"], info["maxs"]
                memory = size_text(info["memory_bytes"])
                if info["ram_bytes"]:
                    memory += f" of {size_text(info['ram_bytes'])}"
                cells = [info["name"], f"{info['point_count']:,}", size_text(info["file_bytes"]),
                         f"{x1 - x0:,.2f}", f"{y1 - y0:,.2f}", f"{z0:,.2f}", f"{z1:,.2f}",
                         info["crs"] or "none", memory, "; ".join(info["issues"]) or "OK"]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if 1 <= column <= 6:
                    item.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))
                if "error" not in info:
                    item.setToolTip(f"{info['source']}\nLAS {info['las_version']}, point format "
                                    f"{info['point_format']}\nCRS unit: {info['horizontal_unit'] or 'unknown'}\n"
                                    f"Fields: {', '.join(info['fields'])}")
                table.setItem(row, column, item)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        frame = table.frameWidth() * 2
        width = table.verticalHeader().width() + table.horizontalHeader().length() + frame
        height = (table.horizontalHeader().height() + table.verticalHeader().length() + frame
                  + table.horizontalScrollBar().sizeHint().height())
        screen = self.screen().availableGeometry()
        table.setMinimumSize(min(width, screen.width() - 80), min(height, screen.height() // 2))
        layout.addWidget(table)
        note = QLabel("Read from the file headers only. Z units are not stored in LAS; "
                      "confirm metres in Settings before processing.")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


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
        self.run_metadata = {}
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

    def path_row(self, grid, row, title, edit, command, tip):
        label = QLabel(title)
        grid.addWidget(label, row, 0)
        grid.addWidget(edit, row, 1)
        button = QPushButton("Browse")
        button.clicked.connect(command)
        grid.addWidget(button, row, 2)
        for widget in (label, edit):
            widget.setToolTip(tip)
        button.setToolTip("浏览选择：" + tip)

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
        self.path_row(grid, 0, "Model", self.model_path, self.pick_model,
                      "训练好的模型文件（.joblib），处理点云时用它判断每个点的类别。")
        self.path_row(grid, 1, "Results directory", self.output_path, self.pick_output,
                      "保存处理结果的文件夹，每次处理会在里面新建一个子文件夹。")
        layout.addLayout(grid)
        controls = QHBoxLayout()
        for text, command, tip in (
                ("Add LAS / LAZ", self.add_files, "添加要处理的 LAS / LAZ 点云文件，可一次选多个。"),
                ("Clear list", self.clear_files, "清空下方的待处理文件列表，不会删除文件本身。"),
                ("Inspect input", self.inspect, "只读文件头，快速查看点数、范围、坐标系等信息，检查文件能否处理。"),
                ("Process", self.process, "用所选模型处理列表中的全部文件，识别并分割目标，结果存到结果文件夹。"),
                ("Open saved results", self.open_run, "打开以前处理好的结果文件夹，在表格和 3D 视图中查看。"),
                ("Synthetic demo", self.open_demo, "打开自带的合成示例数据和结果，用来熟悉界面。")):
            controls.addWidget(self.button(text, command, tip))
        controls.addStretch()
        layout.addLayout(controls)
        self.inputs = QListWidget()
        self.inputs.setFixedHeight(60)
        self.inputs.setToolTip("待处理的点云文件列表。")
        layout.addWidget(self.inputs)
        viewer_controls = QHBoxLayout()
        self.dataset = QComboBox()
        self.dataset.setMinimumWidth(380)
        self.dataset.setToolTip("选择要在表格和 3D 视图中显示哪个文件的结果。")
        self.dataset.currentIndexChanged.connect(lambda _: self.view())
        viewer_controls.addWidget(self.dataset)
        self.view_note = QLabel("")
        viewer_controls.addWidget(self.view_note)
        viewer_controls.addStretch()
        viewer_controls.addWidget(self.button("Open in browser", self.open_browser_view,
                                              "在网页浏览器中打开同一个 3D 视图，例如放到第二块屏幕上看。"))
        layout.addLayout(viewer_controls)
        run_row = QHBoxLayout()
        self.run_info = QLabel("")
        self.run_info.setWordWrap(True)
        self.run_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.run_info.setToolTip("生成这批结果时所用的设置（不是当前下方的设置）。")
        run_row.addWidget(self.run_info, 1)
        self.reuse_button = self.button("Use these settings", self.reuse_run_settings,
                                        "把这次结果所用的设置和模型复制到当前设置中，方便按同样参数再处理。")
        self.reuse_button.setVisible(False)
        run_row.addWidget(self.reuse_button)
        layout.addLayout(run_row)
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
        self.table.setToolTip("点击某个目标，在 3D 视图中定位显示它；点击表头可排序。")
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
        self.path_row(grid, 0, "Dataset manifest JSON", self.manifest_path, self.pick_manifest,
                      "数据集清单 JSON 文件，列出训练、验证、测试所用的点云和标签文件。")
        layout.addLayout(grid)
        hint = QLabel("Labels: point_index,class_id  (0 = background, -1 = unlabelled)")
        hint.setToolTip("标签文件每行一个点：点序号,类别号；正数是目标类别。\n"
                        "训练、验证、测试数据必须来自不同的场景。")
        layout.addWidget(hint)
        row = QHBoxLayout()
        method_label = QLabel("Method")
        row.addWidget(method_label)
        self.backend = QComboBox()
        self.backend.addItems(["random_forest", "mlp"])
        for widget in (method_label, self.backend):
            widget.setToolTip("训练方法：random_forest 是随机森林，稳定、训练快；mlp 是小型神经网络。")
        row.addWidget(self.backend)
        iterations_label = QLabel("Trees")
        self.backend.currentTextChanged.connect(
            lambda text: iterations_label.setText("Trees" if text == "random_forest" else "Max epochs"))
        row.addWidget(iterations_label)
        self.iterations = QLineEdit("100")
        self.iterations.setFixedWidth(70)
        tip = ("随机森林的树数量，或神经网络的最大训练轮数。\n"
               "调大通常更准但训练更慢；调小训练更快但可能欠拟合。")
        iterations_label.setToolTip(tip)
        self.iterations.setToolTip(tip)
        row.addWidget(self.iterations)
        row.addStretch()
        layout.addLayout(row)
        buttons = QHBoxLayout()
        for text, command, tip in (
                ("Create label template", self.make_labels,
                 "为一个点云生成空白标签 CSV 模板，再填入每个点的类别用于训练。"),
                ("Train and save model", self.train, "按数据集清单训练模型，并保存为 .joblib 文件。"),
                ("Evaluate on test set", self.evaluate, "用上方 Model 中的模型在测试集上评估准确率，结果存为 JSON。")):
            buttons.addWidget(self.button(text, command, tip))
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

        self.area = QComboBox()
        self.area.addItems(["unresolved", "convex_hull_xy"])
        fields = [[("Fallback CRS", self.crs), ("Vertical datum", self.datum), (None, self.meters)],
                  [("Score threshold", self.threshold), ("Cluster radius m", self.radius),
                   ("Min object points", self.minimum), ("Area method", self.area)],
                  [("Neighbours", self.neighbors), ("Max points / file", self.limit)]]
        tips = {
            self.crs: "文件自身没有坐标系时使用的投影坐标系，例如 EPSG:32632；文件里有坐标系时忽略此项。",
            self.datum: "高程基准名称，只记录到结果中，不影响计算。",
            self.meters: "确认 X/Y/Z 坐标（包括高程 Z）都以米为单位；不勾选就无法处理。",
            self.threshold: "点被判为目标所需的最低置信度（0 到 1）。\n"
                            "调高结果更可靠但会漏掉一些目标；调低能找到更多目标但误检也更多。",
            self.radius: "把目标点聚成一个物体时的搜索半径（米）。\n"
                         "调大会把相邻物体合并成一个；调小会把一个物体拆成几块。",
            self.minimum: "一个物体至少包含的点数，点数更少的会被丢弃。\n"
                          "调高可去掉零碎噪点，但小目标可能被丢掉；调低保留更多小目标。",
            self.area: "物体面积的计算方式：unresolved 不计算面积；convex_hull_xy 用俯视（XY）凸包面积。",
            self.neighbors: "计算每个点的局部形状特征时参考的邻近点数（3 到 128），训练时使用；\n"
                            "处理时自动采用模型训练时的值。调大特征更平滑但更慢；调小更细致但易受噪声影响。",
            self.limit: "每个文件允许的最大点数，0 表示不限制、处理整个文件。\n"
                        "设一个上限可避免误处理超大文件；超过上限的文件会被拒绝。",
        }
        for row, items in enumerate(fields):
            for col, (title, widget) in enumerate(items):
                widget.setToolTip(tips[widget])
                if title:
                    label = QLabel(title)
                    label.setToolTip(tips[widget])
                    grid.addWidget(label, row, col * 2)
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
            summaries = []
            for path in self.paths:
                try:
                    summaries.append(inspect_file(path))
                except Exception as exc:
                    summaries.append({"name": Path(path).name, "error": f"{type(exc).__name__}: {exc}"})
            InspectDialog(summaries, self).exec()
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
        self.show_run_info(metadata)
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

    def show_run_info(self, metadata):
        self.run_metadata = metadata
        cfg = metadata.get("settings", {})
        model = metadata.get("model_path")
        model_text = Path(model).name if model else f"model {metadata.get('model_id', '?')[:8]}"
        started = metadata.get("started_at", "")[:16].replace("T", " ")
        parts = [started, f"{model_text} ({metadata.get('backend', '?')})"]
        if cfg:
            parts += [f"threshold {cfg['threshold']}", f"cluster radius {cfg['cluster_eps']} m",
                      f"min points {cfg['min_points']}", f"neighbours {cfg['neighbors']}",
                      f"area {cfg['area_method']}"]
            if cfg.get("crs_override"):
                parts.append(f"fallback CRS {cfg['crs_override']}")
        self.run_info.setText("This run: " + " · ".join(parts))
        self.reuse_button.setVisible(bool(cfg))

    def reuse_run_settings(self):
        def action():
            cfg = self.run_metadata["settings"]
            self.threshold.setText(str(cfg["threshold"]))
            self.radius.setText(str(cfg["cluster_eps"]))
            self.minimum.setText(str(cfg["min_points"]))
            self.neighbors.setText(str(cfg["neighbors"]))
            self.limit.setText(str(cfg["max_points"]))
            self.area.setCurrentText(cfg["area_method"])
            self.crs.setText(cfg["crs_override"])
            self.datum.setText(cfg["vertical_datum"])
            self.meters.setChecked(cfg["coordinates_in_meters"])
            model = self.run_metadata.get("model_path")
            if model and Path(model).exists():
                self.model_path.setText(model)
                self.log("Copied this run's settings and model.")
            else:
                self.log("Copied this run's settings; select its model manually "
                         f"(model ID {self.run_metadata.get('model_id', '?')}).")
        self.safe_action(action)

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
