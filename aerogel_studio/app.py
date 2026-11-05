"""Native Chinese desktop UI; all application logic is Python."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import uuid
from datetime import datetime

os.environ.setdefault("QT_API", "pyside6")
from PySide6 import QtCore, QtGui, QtWidgets as W


STYLE = """
QMainWindow, QWidget#workspace { background: #f3f6f8; color: #263544; }
QWidget { font-family: 'Microsoft YaHei UI','Segoe UI'; font-size: 12px; }
QFrame#header { background: #102b37; border: none; }
QLabel#brand { color: white; font-size: 21px; font-weight: 700; }
QLabel#subtitle { color: #a9c4cf; }
QLabel#tag { color: #b5eadf; background:#24444d; border-radius:11px; padding:5px 12px; }
QFrame#card, QGroupBox { background:white; border:1px solid #dce5ea; border-radius:9px; }
QGroupBox { margin-top:12px; padding:16px 12px 9px; font-weight:600; }
QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 5px; color:#55717c; }
QLabel#section { font-size:15px; font-weight:600; color:#153e4a; }
QLabel#hint { color:#73838e; font-size:11px; }
QLabel#value { font-size:19px; font-weight:600; color:#116f76; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background:white; color:#243b48; border:1px solid #ccd9df; border-radius:5px; min-height:28px; padding:2px 7px; selection-background-color:#168387; }
QComboBox QAbstractItemView { background:white; color:#243b48; selection-background-color:#e3f3f2; }
QPushButton, QToolButton { background:white; color:#355461; border:1px solid #d4dfe4; border-radius:5px; padding:7px 12px; }
QPushButton:hover, QToolButton:hover { background:#eaf5f4; border-color:#86b8b9; }
QPushButton#primary { background:#127d82; color:white; border:0; font-size:14px; font-weight:600; padding:11px; }
QPushButton#primary:hover { background:#0d686e; }
QPushButton:disabled, QToolButton:disabled { color:#9eabb3; background:#eef2f4; border-color:#e0e6ea; }
QProgressBar { border:none; background:#e5edef; border-radius:3px; height:6px; }
QProgressBar::chunk { background:#178b8c; border-radius:3px; }
QCheckBox { color:#405b68; spacing:6px; }
QPlainTextEdit { background:#172e3a; color:#d4e7ea; border:0; border-radius:6px; font-family:Consolas,'Microsoft YaHei'; font-size:11px; }
QScrollArea { background:transparent; border:0; }
QScrollBar:vertical { background:transparent; width:7px; margin:0; }
QScrollBar::handle:vertical { background:#cad7dd; border-radius:3px; min-height:30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QStatusBar { background:white; color:#637c87; border-top:1px solid #dce5ea; }
"""


def number(value: float, low: float, high: float, decimals: int = 2, suffix: str = "") -> W.QDoubleSpinBox:
    widget = W.QDoubleSpinBox()
    widget.setRange(low, high)
    widget.setDecimals(decimals)
    widget.setValue(value)
    widget.setSuffix(suffix)
    widget.setKeyboardTracking(False)
    return widget


class StudioWindow(W.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Aerogel Studio · 气凝胶微结构生成器")
        self.resize(1360, 870)
        self.setMinimumSize(1080, 700)
        self.result: dict | None = None
        self.process: QtCore.QProcess | None = None
        self.cancelled = False
        self.buffer = ""
        self.failure = ""
        self.operation = ""
        self.events_path: Path | None = None
        self.events_offset = 0
        self.poll = QtCore.QTimer(self)
        self.poll.setInterval(200)
        self.poll.timeout.connect(self._read)
        self.full_mesh = None
        self.view = None
        self.previous_model = None
        self.seed_counts = {"nanowire": 400, "sheet": 0}
        self.voxel_sizes = {"nanowire": 5.0, "sheet": 2.5}
        self.last_output = Path.home() / "Documents" / "AerogelStudio"
        self._build()
        self._init_view()
        self._set_kind()

    def _build(self) -> None:
        root = W.QWidget(objectName="workspace")
        self.setCentralWidget(root)
        outer = W.QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        header = W.QFrame(objectName="header")
        h = W.QHBoxLayout(header)
        h.setContentsMargins(25, 16, 25, 16)
        title = W.QVBoxLayout()
        title.addWidget(W.QLabel("AEROGEL STUDIO", objectName="brand"))
        title.addWidget(W.QLabel("周期 Voronoi 微结构  /  氧化铝片层与纳米线", objectName="subtitle"))
        h.addLayout(title)
        h.addStretch()
        h.addWidget(W.QLabel("PYTHON  ·  3D MODELING", objectName="tag"))
        outer.addWidget(header)
        body = W.QHBoxLayout()
        body.setContentsMargins(18, 18, 18, 14)
        body.setSpacing(16)
        outer.addLayout(body, 1)

        sidebar = W.QWidget()
        sidebar.setFixedWidth(355)
        left = W.QVBoxLayout(sidebar)
        left.setContentsMargins(0, 0, 0, 0)
        scroll = W.QScrollArea()
        scroll.setWidgetResizable(True)
        panel = W.QWidget()
        self.inputs = panel
        form = W.QVBoxLayout(panel)
        form.setContentsMargins(0, 0, 6, 0)
        form.setSpacing(8)
        self.kind = W.QComboBox()
        self.kind.addItem("纳米线气凝胶", "nanowire")
        self.kind.addItem("氧化铝气凝胶 · 片层", "sheet")
        basic = W.QGroupBox("01   结构参数")
        f = W.QFormLayout(basic)
        f.setVerticalSpacing(10)
        self.porosity = number(95, 1, 99.99, 2, " %")
        self.size = number(1000, 20, 100000, 1, " nm")
        self.tolerance = number(0.5, 0.01, 5, 2, " 百分点")
        f.addRow("气凝胶类型", self.kind)
        f.addRow("目标孔隙率", self.porosity)
        f.addRow("计算域边长", self.size)
        self.wire_box = W.QWidget()
        wf = W.QFormLayout(self.wire_box)
        wf.setContentsMargins(0, 0, 0, 0)
        self.rod_min = number(16, 0.1, 10000, 2, " nm")
        self.rod_max = number(24, 0.1, 10000, 2, " nm")
        wf.addRow("纳米线最小直径", self.rod_min)
        wf.addRow("纳米线最大直径", self.rod_max)
        f.addRow(self.wire_box)
        self.sheet_box = W.QWidget()
        sf = W.QFormLayout(self.sheet_box)
        sf.setContentsMargins(0, 0, 0, 0)
        self.thickness = number(5, 0.1, 1000, 2, " nm")
        self.sheet_min = number(50, 1, 10000, 1, " nm")
        self.sheet_max = number(80, 1, 10000, 1, " nm")
        sf.addRow("片层厚度", self.thickness)
        sf.addRow("偏好等效片径下限", self.sheet_min)
        sf.addRow("偏好等效片径上限", self.sheet_max)
        f.addRow(self.sheet_box)
        self.shape_hint = W.QLabel(objectName="hint")
        self.shape_hint.setWordWrap(True)
        f.addRow(self.shape_hint)
        form.addWidget(basic)

        numerical = W.QGroupBox("02   计算与复现")
        n = W.QFormLayout(numerical)
        self.seeds = W.QSpinBox()
        self.seeds.setRange(0, 20000)
        self.seeds.setSpecialValueText("自动估算")
        self.seeds.setValue(400)
        self.seed = W.QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(42)
        self.voxel = number(5, 0.1, 1000, 2, " nm")
        self.sampling = W.QComboBox()
        self.sampling.addItem("标准 · 2¹⁸ 控制 / 2¹⁹ 核验", (18, 19))
        self.sampling.addItem("快速检查 · 2¹⁶ / 2¹⁷", (16, 17))
        self.sampling.addItem("精细 · 2²⁰ / 2²⁰", (20, 20))
        n.addRow("Voronoi 种子数", self.seeds)
        n.addRow("随机种子", self.seed)
        n.addRow("实体体素间距", self.voxel)
        n.addRow("孔隙率采样", self.sampling)
        n.addRow("孔隙率容差", self.tolerance)
        self.resolution_hint = W.QLabel(objectName="hint")
        self.resolution_hint.setWordWrap(True)
        n.addRow(self.resolution_hint)
        form.addWidget(numerical)

        files = W.QGroupBox("03   文件输出")
        fl = W.QVBoxLayout(files)
        row = W.QHBoxLayout()
        self.formats: dict[str, W.QCheckBox] = {}
        for extension in ("stl", "step", "obj", "ply"):
            check = W.QCheckBox(extension.upper())
            check.setChecked(extension == "stl")
            self.formats[extension] = check
            row.addWidget(check)
        fl.addLayout(row)
        unit_row = W.QFormLayout()
        self.unit = W.QComboBox()
        for label, data in (("μm · 微米（推荐）", "um"), ("nm · 纳米", "nm"), ("mm · 毫米", "mm")):
            self.unit.addItem(label, data)
        unit_row.addRow("导出坐标单位", self.unit)
        fl.addLayout(unit_row)
        out_row = W.QHBoxLayout()
        self.output = W.QLineEdit(str(self.last_output))
        choose = W.QPushButton("浏览")
        choose.clicked.connect(self._choose_output)
        out_row.addWidget(self.output, 1)
        out_row.addWidget(choose)
        fl.addLayout(out_row)
        note = W.QLabel("STL 附 FDTD 导入脚本；STEP 为封闭分面实体。\n每次生成保存到独立文件夹。", objectName="hint")
        note.setWordWrap(True)
        fl.addWidget(note)
        form.addWidget(files)
        form.addStretch()
        scroll.setWidget(panel)
        left.addWidget(scroll, 1)
        self.generate_button = W.QPushButton("生成微观模型", objectName="primary")
        self.generate_button.clicked.connect(self._generate)
        left.addWidget(self.generate_button)
        bottom = W.QHBoxLayout()
        self.open_button = W.QPushButton("打开已有结果")
        self.open_button.clicked.connect(self._open_result)
        self.cancel_button = W.QPushButton("停止计算")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        bottom.addWidget(self.open_button)
        self.demo_button = W.QPushButton("载入示例")
        self.demo_button.clicked.connect(self._open_demo)
        bottom.addWidget(self.demo_button)
        bottom.addWidget(self.cancel_button)
        left.addLayout(bottom)
        body.addWidget(sidebar)

        right = W.QVBoxLayout()
        body.addLayout(right, 1)
        toolbar = W.QHBoxLayout()
        toolbar.addWidget(W.QLabel("三维结构预览", objectName="section"))
        toolbar.addStretch()
        for label, callback in (("等轴测", lambda: self._camera("iso")), ("XY", lambda: self._camera("xy")), ("XZ", lambda: self._camera("xz")), ("YZ", lambda: self._camera("yz"))):
            button = W.QPushButton(label)
            button.clicked.connect(callback)
            toolbar.addWidget(button)
        self.bounds = W.QCheckBox("胞元边框")
        self.bounds.setChecked(True)
        self.bounds.toggled.connect(self._toggle_box)
        toolbar.addWidget(self.bounds)
        right.addLayout(toolbar)
        self.viewer_frame = W.QFrame(objectName="card")
        self.viewer_layout = W.QVBoxLayout(self.viewer_frame)
        self.viewer_layout.setContentsMargins(1, 1, 1, 1)
        right.addWidget(self.viewer_frame, 1)
        self.preview_note = W.QLabel("左键拖动旋转  ·  滚轮缩放  ·  Shift＋拖动平移", objectName="hint")
        right.addWidget(self.preview_note)
        cards = W.QFrame(objectName="card")
        stats = W.QHBoxLayout(cards)
        self.stats: dict[str, W.QLabel] = {}
        for key, label in (("kind", "模型"), ("porosity", "独立核验孔隙率"), ("voxel", "实体网格孔隙率"), ("count", "网格三角面")):
            stack = W.QVBoxLayout()
            stack.addWidget(W.QLabel(label, objectName="hint"))
            value = W.QLabel("—", objectName="value")
            self.stats[key] = value
            stack.addWidget(value)
            stats.addLayout(stack, 1)
        right.addWidget(cards)
        actions = W.QHBoxLayout()
        self.status_label = W.QLabel("就绪 · 设置参数后生成结构", objectName="hint")
        actions.addWidget(self.status_label, 1)
        self.export_button = W.QPushButton("导出模型文件")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export)
        self.image_button = W.QPushButton("保存当前视图")
        self.image_button.setEnabled(False)
        self.image_button.clicked.connect(self._save_image)
        self.folder_button = W.QPushButton("结果文件夹")
        self.folder_button.setEnabled(False)
        self.folder_button.clicked.connect(self._open_folder)
        for button in (self.export_button, self.image_button, self.folder_button):
            actions.addWidget(button)
        right.addLayout(actions)
        self.progress = W.QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        right.addWidget(self.progress)
        self.log_toggle = W.QPushButton("显示运行详情")
        self.log_toggle.setFlat(True)
        self.log_toggle.clicked.connect(self._toggle_log)
        right.addWidget(self.log_toggle, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.log = W.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(125)
        self.log.hide()
        right.addWidget(self.log)
        self.statusBar().showMessage("长度参数：nm  |  孔隙率：%  |  周期几何与离散实体分别记录核验结果")
        self.kind.currentIndexChanged.connect(self._set_kind)
        self.size.valueChanged.connect(self._resolution)
        self.voxel.valueChanged.connect(self._resolution)
        for control in (self.porosity, self.size, self.rod_min, self.rod_max, self.thickness,
                        self.sheet_min, self.sheet_max, self.seeds, self.seed, self.voxel, self.tolerance):
            control.valueChanged.connect(self._mark_dirty)
        self.kind.currentIndexChanged.connect(self._mark_dirty)
        self.sampling.currentIndexChanged.connect(self._mark_dirty)
        self._resolution()

    def _mark_dirty(self) -> None:
        if self.result is not None and self.process is None:
            self.status_label.setText("参数已修改 · 当前预览和导出仍为上次结果，请重新生成以应用新参数")

    def _init_view(self) -> None:
        try:
            from pyvistaqt import QtInteractor
            self.view = QtInteractor(self.viewer_frame, auto_update=False, multi_samples=4)
            self.viewer_layout.addWidget(self.view.interactor)
            self.view.set_background("#f4f7f9")
            self.view.enable_lightkit()
            self.view.enable_parallel_projection()
            self.view.add_text("VORONOI MICROSTRUCTURE", position="upper_left", color="#81949e", font_size=10)
            self.view.render()
        except Exception as error:
            self.viewer_layout.addWidget(W.QLabel("三维视窗初始化失败；仍可生成和导出模型。\n" + str(error)))
            self.log.appendPlainText(str(error))
            self.view = None

    def _set_kind(self) -> None:
        model = self.kind.currentData()
        if self.previous_model is not None:
            self.seed_counts[self.previous_model] = self.seeds.value()
            self.voxel_sizes[self.previous_model] = self.voxel.value()
        wire = model == "nanowire"
        self.wire_box.setVisible(wire)
        self.sheet_box.setVisible(not wire)
        self.seeds.setMinimum(5 if wire else 0)
        self.seeds.setValue(self.seed_counts[model])
        self.voxel.setValue(self.voxel_sizes[model])
        self.previous_model = model
        self.shape_hint.setText("直径按范围均匀随机分配；两值相等时为固定直径。线长由 Voronoi 剖分决定。" if wire else "片径范围为选面偏好；片层厚度独立指定。")

    def _resolution(self) -> None:
        import math
        count = math.ceil(self.size.value() / self.voxel.value())
        self.resolution_hint.setText(f"约 {count}³ = {count ** 3 / 1e6:.1f} 百万体素。减小间距能细化实体，也会增加内存与计算时间。")

    def _choose_output(self) -> None:
        path = W.QFileDialog.getExistingDirectory(self, "选择输出位置", self.output.text())
        if path:
            self.output.setText(path)

    def _request(self) -> dict:
        model = self.kind.currentData()
        power, validation_power = self.sampling.currentData()
        common = dict(cube_size_nm=self.size.value(), target_porosity=self.porosity.value() / 100,
                      porosity_tolerance=self.tolerance.value() / 100, random_seed=self.seed.value(),
                      qmc_power=power, validation_qmc_power=validation_power,
                      voxel_size_nm=self.voxel.value())
        if model == "nanowire":
            if self.rod_min.value() > self.rod_max.value():
                raise ValueError("最小纳米线直径不能大于最大直径。")
            if self.rod_max.value() >= self.size.value():
                raise ValueError("纳米线直径必须小于计算域边长。")
            common.update(num_seeds=self.seeds.value(), rod_diameter_min_nm=self.rod_min.value(),
                          rod_diameter_max_nm=self.rod_max.value(), seed_min_distance_factor=0.3)
        else:
            if self.sheet_min.value() >= self.sheet_max.value():
                raise ValueError("偏好片径下限必须小于上限。")
            common.update(thickness_nm=self.thickness.value(), diameter_min_nm=self.sheet_min.value(),
                          diameter_max_nm=self.sheet_max.value(), num_seeds=self.seeds.value() or None)
        smallest = self.rod_min.value() if model == "nanowire" else self.thickness.value()
        limit = smallest / 2 if model == "nanowire" else smallest
        if self.voxel.value() > limit:
            raise ValueError("纳米线体素间距不能超过最小直径的一半；片层间距不能大于片层厚度。")
        count = __import__("math").ceil(self.size.value() / self.voxel.value())
        if count > 512:
            raise ValueError("当前体素网格超过 512³；请增大间距或缩小计算域，避免内存不足。")
        base = Path(self.output.text()).expanduser().resolve()
        run_dir = base / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + model + "_" + uuid.uuid4().hex[:5])
        common["output_dir"] = str(run_dir)
        return dict(action="generate", model=model, config=common, formats=self._formats(), unit=self.unit.currentData())

    def _formats(self) -> list[str]:
        return [key for key, check in self.formats.items() if check.isChecked()]

    def _generate(self) -> None:
        try:
            request = self._request()
            self._start(request)
        except Exception as error:
            W.QMessageBox.warning(self, "请检查参数", str(error))

    def _start(self, request: dict) -> None:
        if self.process is not None:
            return
        base = Path(request.get("config", {}).get("output_dir", self.output.text())).expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
        request_path = base / ("studio_request_" + uuid.uuid4().hex[:8] + ".json")
        self.events_path = request_path.with_suffix(".events.jsonl")
        self.events_offset = 0
        request["events_path"] = str(self.events_path)
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        self.operation = request["action"]
        self.buffer = ""
        self.failure = ""
        self.cancelled = False
        self.log.clear()
        process = QtCore.QProcess(self)
        self.process = process
        process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("MPLBACKEND", "Agg")
        env.insert("QT_API", "pyside6")
        process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(self._read)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._process_error)
        args = ["--worker", str(request_path)]
        if not getattr(sys, "frozen", False):
            args = ["-m", "aerogel_studio.worker", str(request_path)]
            process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
        process.setProgram(sys.executable)
        process.setArguments(args)
        self.inputs.setEnabled(False)
        for button in (self.generate_button, self.open_button, self.demo_button, self.export_button):
            button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.status_label.setText("正在生成结构…" if self.operation == "generate" else "正在导出模型…")
        process.start()
        self.poll.start()

    def _read(self) -> None:
        if self.process is None:
            return
        extra = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if extra.strip():
            self.log.appendPlainText(extra.strip())
        if self.events_path is not None and self.events_path.exists():
            with self.events_path.open("rb") as stream:
                stream.seek(self.events_offset)
                incoming = stream.read()
                self.events_offset += len(incoming)
            self.buffer += incoming.decode("utf-8", errors="replace")
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.strip()
            try:
                event = json.loads(line)
            except ValueError:
                self.log.appendPlainText(line)
                continue
            if event["event"] == "progress":
                self.log.appendPlainText(event["message"])
                message = event["message"].lower()
                stage = "正在构建微结构"
                for words, label in ((('seed', 'attempt'), '生成周期种子'),
                                     (('voronoi',), '构造三维 Voronoi 剖分'),
                                     (('select', 'sobol', 'coverage'), '按并集体积选择骨架'),
                                     (('independent', 'validat'), '独立核验孔隙率'),
                                     (('voxel', 'occupancy'), '构建体素实体并集'),
                                     (('mesh', 'triang'), '提取并检查封闭实体网格'),
                                     (('export', 'writing', 'saving'), '保存模型与导出文件')):
                    if any(word in message for word in words):
                        stage = label
                self.status_label.setText("计算中 · " + stage)
            elif event["event"] == "error":
                self.failure = event["message"]
                self.log.appendPlainText(event.get("detail", ""))
            elif event["event"] == "success":
                if self.operation == "generate":
                    self._load_result(event["result"])
                else:
                    self.log.appendPlainText(json.dumps(event["result"], ensure_ascii=False, indent=2))

    def _finished(self, code: int, _status) -> None:
        self._read()
        self.poll.stop()
        process = self.process
        self.process = None
        if process is not None:
            process.deleteLater()
        self.inputs.setEnabled(True)
        self.generate_button.setEnabled(True)
        self.open_button.setEnabled(True)
        self.demo_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.export_button.setEnabled(self.result is not None)
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if code == 0 and not self.cancelled else 0)
        if self.cancelled:
            self.status_label.setText("计算已停止 · 未完成文件保留用于排查")
        elif code != 0 or self.failure:
            self.status_label.setText("计算未完成 · 请查看运行详情")
            self.log.show()
            W.QMessageBox.warning(self, "模型生成或导出未完成", self.failure or "计算进程异常退出，详细信息见运行日志。")
        else:
            self.status_label.setText("生成完成 · 可以旋转预览并导出文件" if self.operation == "generate" else "文件导出完成")

    def _process_error(self, error) -> None:
        if error == QtCore.QProcess.ProcessError.FailedToStart:
            self.failure = "无法启动计算进程：" + self.process.errorString()
            self._finished(1, None)

    def _cancel(self) -> None:
        if self.process is not None:
            self.cancelled = True
            self.process.kill()

    def _load_result(self, result: dict) -> None:
        self.result = result
        config = result["config"]
        self.kind.setCurrentIndex(self.kind.findData(result["model"]))
        self.porosity.setValue(config["target_porosity"] * 100)
        self.tolerance.setValue(config["porosity_tolerance"] * 100)
        self.size.setValue(config["cube_size_nm"])
        self.seed.setValue(config["random_seed"])
        self.seeds.setValue(config.get("num_seeds") or 0)
        self.voxel.setValue(config["voxel_size_nm"])
        if result["model"] == "nanowire":
            self.rod_min.setValue(config["rod_diameter_min_nm"])
            self.rod_max.setValue(config["rod_diameter_max_nm"])
        else:
            self.thickness.setValue(config["thickness_nm"])
            self.sheet_min.setValue(config["diameter_min_nm"])
            self.sheet_max.setValue(config["diameter_max_nm"])
        powers = (config["qmc_power"], config["validation_qmc_power"])
        for index in range(self.sampling.count()):
            if tuple(self.sampling.itemData(index)) == powers:
                self.sampling.setCurrentIndex(index)
                break
        metadata = result.get("metadata", {})
        porosity = metadata.get("porosity", {})
        self.stats["kind"].setText("纳米线" if result["model"] == "nanowire" else "氧化铝片层")
        for key, source in (("porosity", "independent_qmc"), ("voxel", "closed_mesh_volume")):
            value = porosity.get(source, porosity.get("voxel_count") if key == "voxel" else None)
            self.stats[key].setText(f"{100 * value:.3f}%" if value is not None else "—")
        counts = result.get("counts", {})
        self.stats["count"].setText(f"{counts.get('full_triangles', 0):,}")
        self.folder_button.setEnabled(True)
        self.export_button.setEnabled(self.process is None)
        self.image_button.setEnabled(self.view is not None)
        if self.view is not None:
            import pyvista as pv
            self.full_mesh = pv.read(result["preview_path"])
            self.view.clear()
            self.view.enable_lightkit()
            self.view.add_mesh(self.full_mesh, color="#c38e46" if result["model"] == "nanowire" else "#83a8bb", smooth_shading=True, lighting=True, ambient=0.12, diffuse=0.8, specular=0.3, specular_power=25)
            self._toggle_box(self.bounds.isChecked())
            self.view.view_isometric()
            self.view.reset_camera()
            self.view.render()
        suffix = " · 大网格使用简化预览，导出保留完整实体" if result.get("preview_simplified") else ""
        self.preview_note.setText("左键拖动旋转  ·  滚轮缩放  ·  Shift＋拖动平移" + suffix)

    def _toggle_box(self, visible: bool) -> None:
        if self.view is None or self.result is None:
            return
        self.view.remove_actor("cell_box", render=False)
        if visible:
            import pyvista as pv
            length = self.result["config"]["cube_size_nm"]
            self.view.add_mesh(pv.Box(bounds=(0, length, 0, length, 0, length)).outline(), color="#9baeb7", line_width=1.2, name="cell_box", render=False)
        self.view.render()

    def _camera(self, kind: str) -> None:
        if self.view is not None:
            {"iso": self.view.view_isometric, "xy": self.view.view_xy, "xz": self.view.view_xz, "yz": self.view.view_yz}[kind]()
            self.view.reset_camera()
            self.view.render()

    def _open_result(self) -> None:
        path, _ = W.QFileDialog.getOpenFileName(self, "打开 Aerogel Studio 结果", self.output.text(), "Studio 结果 (studio_result.json)")
        if not path:
            return
        self._open_record(path)

    def _open_demo(self) -> None:
        if getattr(sys, "frozen", False):
            path = Path(sys.executable).parent / "示例项目" / "纳米线_100nm" / "studio_result.json"
        else:
            path = Path(__file__).resolve().parents[1] / "studio_demo" / "nanowire_100nm" / "studio_result.json"
        self._open_record(str(path))
        if self.result is not None:
            self.status_label.setText("已载入示例 · 可旋转与导出；修改参数后重新生成")

    def _open_record(self, path: str) -> None:
        try:
            record = json.loads(Path(path).read_text(encoding="utf-8"))
            record["output_dir"] = str(Path(path).resolve().parent)
            record["result_path"] = str(Path(path).resolve())
            for key in ("mesh_path", "preview_path"):
                saved = Path(record[key])
                if not saved.exists():
                    record[key] = str(Path(path).resolve().parent / saved.name)
            self._load_result(record)
            self.export_button.setEnabled(True)
            self.status_label.setText("已载入保存的结构 · 参数以该结果记录为准")
        except Exception as error:
            W.QMessageBox.warning(self, "无法打开结果", str(error))

    def _export(self) -> None:
        if not self._formats():
            W.QMessageBox.information(self, "选择格式", "请先勾选至少一种输出格式。")
            return
        destination = W.QFileDialog.getExistingDirectory(self, "选择模型导出文件夹", self.result["output_dir"])
        if destination:
            self._start(dict(action="export", result_path=self.result["result_path"], destination=destination, formats=self._formats(), unit=self.unit.currentData()))

    def _save_image(self) -> None:
        path, _ = W.QFileDialog.getSaveFileName(self, "保存当前三维视图", str(Path(self.result["output_dir"]) / "studio_view.png"), "PNG 图片 (*.png)")
        if path and self.view is not None:
            self.view.screenshot(path, scale=2)

    def _open_folder(self) -> None:
        if self.result is not None:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.result["output_dir"]))

    def _toggle_log(self) -> None:
        self.log.setVisible(not self.log.isVisible())
        self.log_toggle.setText("收起运行详情" if self.log.isVisible() else "显示运行详情")

    def closeEvent(self, event) -> None:
        if self.process is not None:
            answer = W.QMessageBox.question(self, "计算正在运行", "关闭软件将停止当前计算。是否关闭？")
            if answer != W.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.process.kill()
            self.process.waitForFinished(3000)
        if self.view is not None:
            self.view.close()
        event.accept()


def run() -> int:
    app = W.QApplication.instance() or W.QApplication(sys.argv)
    app.setApplicationName("Aerogel Studio")
    app.setOrganizationName("AerogelStudio")
    app.setWindowIcon(QtGui.QIcon(str(Path(__file__).resolve().parent / "assets" / "app_icon.ico")))
    app.setFont(QtGui.QFont("Microsoft YaHei UI", 9))
    app.setStyleSheet(STYLE)
    window = StudioWindow()
    available = app.primaryScreen().availableGeometry()
    window.resize(min(1360, available.width() - 40), min(870, available.height() - 45))
    window.show()
    return app.exec()
