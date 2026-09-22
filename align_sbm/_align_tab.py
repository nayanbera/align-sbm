"""Alignment runner tab — controls, live log, and per-motor scan plots."""
import numpy as np

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QIntValidator
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QGroupBox, QCheckBox, QPushButton, QLabel, QLineEdit,
    QProgressBar, QPlainTextEdit, QListWidget, QListWidgetItem,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox, QComboBox, QDialog, QDialogButtonBox,
    QAbstractItemView,
)
from .smart_scan_functions import ScanStatus
from ._hold_widget import HoldConditionsWidget

try:
    import pyqtgraph as pg
    _PG = True
except ImportError:
    _PG = False

_MOTOR_TABS = ["BRG2", "Pitch", "Roll2", "X2"]


class _MoveToEnergyThread(QThread):
    """Background thread that moves all motors to the positions of a selected energy row."""
    log_chunk = pyqtSignal(str)
    done      = pyqtSignal()

    def __init__(self, row, kwargs, simulate, parent=None):
        super().__init__(parent)
        self._row      = row       # [MonoE, Harmonic, UndE, Roll2, X2]
        self._kwargs   = kwargs
        self._simulate = simulate

    def run(self):
        try:
            self._do_move()
        except Exception:
            import traceback
            self.log_chunk.emit(f"\n[MoveToEnergy] ERROR:\n{traceback.format_exc()}")
        finally:
            self.done.emit()

    def _apply_extra_pvs(self, pvlist, caput_fn):
        """Write pre/post energy PV tuples and optionally wait for a readback PV."""
        import time
        for entry in pvlist:
            pvname   = entry[0]
            value    = entry[1]
            wait_pv  = entry[2] if len(entry) > 2 else None
            wait_val = entry[3] if len(entry) > 3 else None
            self.log_chunk.emit(f"[MoveToEnergy] caput {pvname} → {value}\n")
            caput_fn(pvname, value, wait=False)
            if wait_pv:
                self.log_chunk.emit(
                    f"[MoveToEnergy] waiting {wait_pv} == {wait_val}\n"
                )
                try:
                    from epics import caget
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        if caget(wait_pv) == wait_val:
                            break
                        time.sleep(0.2)
                except Exception:
                    pass

    def _do_move(self):
        from .smart_scan_functions import caput
        import time
        kw  = self._kwargs
        row = self._row
        mono_e, harmonic, und_e, roll2, x2 = (
            row[0], row[1], row[2], row[3], row[4]
        )
        pre_pvs  = kw.get("pre_energy_pvs")  or []
        post_pvs = kw.get("post_energy_pvs") or []

        self.log_chunk.emit(
            f"\n[MoveToEnergy] Target: MonoE={mono_e} keV  Harmonic={harmonic}"
            f"  UndE={und_e} eV  Roll2={roll2}  X2={x2}\n"
        )

        if self._simulate:
            lines = ["[MoveToEnergy] SIMULATE — would set:"]
            for entry in pre_pvs:
                lines.append(f"  [Pre]  caput {entry[0]} → {entry[1]}")
            lines += [
                f"  {kw.get('mono_e_pv','')}  → {mono_e}",
                f"  {kw.get('harmonic_pv','')} → {harmonic}",
                f"  {kw.get('und_e_pv','')}    → {und_e}",
                f"  {kw.get('und_start_pv','')} → 1",
                f"  {kw.get('roll2_energy_pv','')} → {roll2}",
                f"  {kw.get('x2_energy_pv','')}    → {x2}",
                f"  roll2_motor ({kw.get('roll2_motor','')}) → {roll2}",
                f"  x2_motor    ({kw.get('x2_motor','')})    → {x2}",
            ]
            for entry in post_pvs:
                lines.append(f"  [Post] caput {entry[0]} → {entry[1]}")
            self.log_chunk.emit("\n".join(lines) + "\n")
            return

        # Enable energy pseudomotor (pre-energy PVs)
        if pre_pvs:
            self._apply_extra_pvs(pre_pvs, caput)

        # Set energy PVs
        for pv_key, value, label in [
            ("mono_e_pv",      mono_e,   "mono energy"),
            ("harmonic_pv",    harmonic, "undulator harmonic"),
            ("und_e_pv",       und_e,    "undulator energy"),
            ("und_start_pv",   1,        "undulator start"),
            ("roll2_energy_pv", roll2,   "Roll2 energy set"),
            ("x2_energy_pv",    x2,      "X2 energy set"),
        ]:
            pv = kw.get(pv_key, "")
            if pv:
                self.log_chunk.emit(f"[MoveToEnergy] caput {pv} → {value}\n")
                caput(pv, value)

        time.sleep(0.5)

        # Move motors
        for motor_key, value, label in [
            ("roll2_motor", roll2, "Roll2"),
            ("x2_motor",    x2,   "X2"),
        ]:
            motor = kw.get(motor_key, "")
            if motor:
                pv = motor + ".VAL"
                self.log_chunk.emit(f"[MoveToEnergy] Moving {label}: caput {pv} → {value}\n")
                caput(pv, value, wait=True)

        # Disable energy pseudomotor (post-energy PVs)
        if post_pvs:
            self._apply_extra_pvs(post_pvs, caput)

        self.log_chunk.emit("[MoveToEnergy] Done.\n")


_POINT_COLOR      = "#4fc3f7"   # coarse scan points (blue)
_FINE_POINT_COLOR = "#66bb6a"   # fine scan points (green)
_FIT_COLOR        = "#ef5350"
_PEAK_COLOR       = "#ffa726"
_BG_COLOR    = "#1a1a2e"

_RES_COLS    = ["#", "MonoE (keV)", "BRG2 ctr", "Roll2 RBV", "X2 RBV", "OK"]


def _fmt(v, decimals=6):
    try:
        f = float(v)
        return "—" if (f != f) else f"{f:.{decimals}g}"   # nan check
    except (TypeError, ValueError):
        return "—"


class _ColorRuleDialog(QDialog):
    """Dialog for editing CSV row color-coding rules."""
    _OPS = [">", "<", ">=", "<=", "=", "!=", "in range", "out of range"]

    def __init__(self, rules, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Color Coding Rules")
        self.setMinimumWidth(700)
        self.setMinimumHeight(300)

        vbox = QVBoxLayout(self)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(["Operator", "Value", "Value 2", "Label", "Color", ""])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(5, 32)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        vbox.addWidget(self._table)

        for rule in rules:
            self._add_row(rule)

        add_btn = QPushButton("+ Add Rule")
        add_btn.clicked.connect(lambda: self._add_row({}))
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        hbox = QHBoxLayout()
        hbox.addWidget(add_btn)
        hbox.addStretch()
        hbox.addWidget(btns)
        vbox.addLayout(hbox)

    def _add_row(self, rule=None):
        from PyQt6.QtGui import QColor
        if rule is None:
            rule = {}
        r = self._table.rowCount()
        self._table.insertRow(r)

        op_combo = QComboBox()
        op_combo.addItems(self._OPS)
        op = rule.get("op", ">")
        idx = op_combo.findText(op)
        if idx >= 0:
            op_combo.setCurrentIndex(idx)
        self._table.setCellWidget(r, 0, op_combo)

        val_edit = QLineEdit(str(rule.get("value", "")))
        self._table.setCellWidget(r, 1, val_edit)

        val2_edit = QLineEdit(str(rule.get("value2", "") or ""))
        val2_edit.setEnabled(op in ("in range", "out of range"))
        self._table.setCellWidget(r, 2, val2_edit)

        def _on_op(text, v2=val2_edit):
            v2.setEnabled(text in ("in range", "out of range"))
        op_combo.currentTextChanged.connect(_on_op)

        label_edit = QLineEdit(str(rule.get("label", "")))
        label_edit.setPlaceholderText("optional display text")
        self._table.setCellWidget(r, 3, label_edit)

        color_hex = rule.get("color", "#4caf50")
        color_btn = QPushButton()
        color_btn.setToolTip("Click to choose a color")
        self._set_color_btn(color_btn, color_hex)
        color_btn.clicked.connect(self._pick_color)
        self._table.setCellWidget(r, 4, color_btn)

        rm_btn = QPushButton("✕")
        rm_btn.setMaximumWidth(32)
        rm_btn.clicked.connect(self._remove_row)
        self._table.setCellWidget(r, 5, rm_btn)

    @staticmethod
    def _set_color_btn(btn, hex_color):
        from PyQt6.QtGui import QColor
        btn.setProperty("color_hex", hex_color)
        c = QColor(hex_color)
        lum = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
        fg = "#000000" if lum > 128 else "#ffffff"
        btn.setStyleSheet(f"background-color: {hex_color}; color: {fg}; border-radius: 3px;")
        btn.setText(hex_color)

    def _pick_color(self):
        from PyQt6.QtWidgets import QColorDialog
        from PyQt6.QtGui import QColor
        btn = self.sender()
        current = btn.property("color_hex") or "#4caf50"
        c = QColorDialog.getColor(QColor(current), self)
        if c.isValid():
            self._set_color_btn(btn, c.name())

    def _remove_row(self):
        btn = self.sender()
        for r in range(self._table.rowCount()):
            if self._table.cellWidget(r, 5) is btn:
                self._table.removeRow(r)
                return

    def get_rules(self):
        rules = []
        for r in range(self._table.rowCount()):
            op_w    = self._table.cellWidget(r, 0)
            val_w   = self._table.cellWidget(r, 1)
            val2_w  = self._table.cellWidget(r, 2)
            label_w = self._table.cellWidget(r, 3)
            col_w   = self._table.cellWidget(r, 4)
            if not (op_w and val_w and col_w):
                continue
            rules.append({
                "op":     op_w.currentText(),
                "value":  val_w.text().strip(),
                "value2": val2_w.text().strip() if val2_w else "",
                "label":  label_w.text().strip() if label_w else "",
                "color":  col_w.property("color_hex") or "#4caf50",
            })
        return rules


class _CrystalConfigDialog(QDialog):
    """Configure the crystal-status PV name and value → label/color mappings."""
    _PALETTE = ["#1565c0", "#2e7d32", "#e65100", "#6a1b9a", "#00838f", "#ad1457"]

    def __init__(self, pv_name, mappings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Crystal Status Configuration")
        self.setMinimumWidth(520)
        self.setMinimumHeight(300)

        vbox = QVBoxLayout(self)
        from PyQt6.QtWidgets import QFormLayout
        form = QFormLayout()
        self._pv_edit = QLineEdit(pv_name)
        self._pv_edit.setPlaceholderText("e.g. ID15:DCM:crystal")
        form.addRow("PV name:", self._pv_edit)
        vbox.addLayout(form)

        vbox.addWidget(QLabel("Value → Crystal mappings:"))
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["PV value", "Display name", "Color"])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        vbox.addWidget(self._table)

        for m in mappings:
            self._add_row(m)

        row_btns = QHBoxLayout()
        add_btn = QPushButton("+ Add")
        add_btn.clicked.connect(lambda: self._add_row({}))
        rm_btn = QPushButton("Remove")
        rm_btn.clicked.connect(self._remove_selected)
        row_btns.addWidget(add_btn)
        row_btns.addWidget(rm_btn)
        row_btns.addStretch()
        vbox.addLayout(row_btns)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        vbox.addWidget(btns)

    def _add_row(self, mapping=None):
        if mapping is None:
            mapping = {}
        r = self._table.rowCount()
        self._table.insertRow(r)
        self._table.setCellWidget(r, 0, QLineEdit(str(mapping.get("raw", ""))))
        self._table.setCellWidget(r, 1, QLineEdit(str(mapping.get("label", ""))))
        color_hex = mapping.get("color", self._PALETTE[r % len(self._PALETTE)])
        color_btn = QPushButton()
        color_btn.setToolTip("Click to choose a color")
        _ColorRuleDialog._set_color_btn(color_btn, color_hex)
        color_btn.clicked.connect(self._pick_color)
        self._table.setCellWidget(r, 2, color_btn)

    def _pick_color(self):
        from PyQt6.QtWidgets import QColorDialog
        from PyQt6.QtGui import QColor
        btn = self.sender()
        c = QColorDialog.getColor(QColor(btn.property("color_hex") or "#1565c0"), self)
        if c.isValid():
            _ColorRuleDialog._set_color_btn(btn, c.name())

    def _remove_selected(self):
        row = self._table.currentRow()
        if row >= 0:
            self._table.removeRow(row)

    def get_pv_name(self):
        return self._pv_edit.text().strip()

    def get_mappings(self):
        result = []
        for r in range(self._table.rowCount()):
            raw_w   = self._table.cellWidget(r, 0)
            label_w = self._table.cellWidget(r, 1)
            color_w = self._table.cellWidget(r, 2)
            if not (raw_w and label_w and color_w):
                continue
            result.append({
                "raw":   raw_w.text().strip(),
                "label": label_w.text().strip(),
                "color": color_w.property("color_hex") or "#1565c0",
            })
        return result


class _CrystalStatusWidget(QWidget):
    """Chip showing the current crystal, updated via CA monitor."""
    _pv_received = pyqtSignal(str)   # CA thread → Qt main thread

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._pv_name  = ""
        self._mappings = []
        self._pv_obj   = None

        self._build_ui()
        self._pv_received.connect(self._update_chip)
        self._load_settings()

    def _build_ui(self):
        hbox = QHBoxLayout(self)
        hbox.setContentsMargins(0, 2, 0, 2)
        hbox.setSpacing(6)
        hbox.addWidget(QLabel("Current Crystal:"))
        self._chip = QLabel("—")
        self._chip.setMinimumWidth(80)
        self._chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._chip.setStyleSheet(
            "background-color: #555; color: #fff; border-radius: 4px; "
            "padding: 2px 10px; font-weight: bold;"
        )
        hbox.addWidget(self._chip)
        cfg_btn = QPushButton("⚙")
        cfg_btn.setMaximumWidth(28)
        cfg_btn.setToolTip("Configure crystal status PV and value mappings")
        cfg_btn.clicked.connect(self._configure)
        hbox.addWidget(cfg_btn)
        hbox.addStretch()

    def _subscribe(self):
        self._unsubscribe()
        if not self._pv_name:
            return
        try:
            from epics import PV
            self._pv_obj = PV(
                self._pv_name,
                callback=self._on_ca_value,
                connection_callback=self._on_ca_connect,
                auto_monitor=True,
            )
        except Exception:
            pass

    def _unsubscribe(self):
        if self._pv_obj is not None:
            try:
                self._pv_obj.disconnect()
            except Exception:
                pass
            self._pv_obj = None

    def _on_ca_value(self, pvname=None, value=None, char_value=None, **kw):
        """Called from CA thread — emit signal to update UI in Qt main thread."""
        val = str(char_value).strip() if char_value is not None else str(value)
        self._pv_received.emit(val)

    def _on_ca_connect(self, pvname=None, conn=False, **kw):
        if not conn:
            self._pv_received.emit("")

    def _update_chip(self, raw_value):
        from PyQt6.QtGui import QColor
        if not raw_value:
            self._chip.setText("—")
            self._chip.setStyleSheet(
                "background-color: #555; color: #fff; border-radius: 4px; "
                "padding: 2px 10px; font-weight: bold;"
            )
            return
        for m in self._mappings:
            if m.get("raw", "").strip() == raw_value:
                label = m.get("label") or raw_value
                color = m.get("color", "#1565c0")
                c     = QColor(color)
                lum   = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
                fg    = "#000000" if lum > 128 else "#ffffff"
                self._chip.setText(label)
                self._chip.setStyleSheet(
                    f"background-color: {color}; color: {fg}; border-radius: 4px; "
                    "padding: 2px 10px; font-weight: bold;"
                )
                return
        # No mapping matched — show raw value in a neutral chip
        self._chip.setText(raw_value)
        self._chip.setStyleSheet(
            "background-color: #37474f; color: #fff; border-radius: 4px; "
            "padding: 2px 10px; font-weight: bold;"
        )

    def _configure(self):
        dlg = _CrystalConfigDialog(
            pv_name=self._pv_name,
            mappings=list(self._mappings),
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._pv_name  = dlg.get_pv_name()
            self._mappings = dlg.get_mappings()
            self.save_settings()
            self._subscribe()

    def _load_settings(self):
        import json
        if not self._settings:
            self._subscribe()
            return
        self._pv_name = self._settings.value("crystal_pv_name", "")
        raw = self._settings.value("crystal_mappings", "[]")
        try:
            self._mappings = json.loads(raw) if isinstance(raw, str) else []
        except Exception:
            self._mappings = []
        self._subscribe()

    def save_settings(self):
        import json
        if not self._settings:
            return
        self._settings.setValue("crystal_pv_name", self._pv_name)
        self._settings.setValue("crystal_mappings", json.dumps(self._mappings))

    def reload_settings(self):
        self._load_settings()


class AlignTab(QWidget):
    status_message = pyqtSignal(str)

    def __init__(self, setup_tab, energy_tab, parent=None, settings=None):
        super().__init__(parent)
        self._setup_tab    = setup_tab
        self._energy_tab   = energy_tab
        self._worker       = None
        self._settings     = settings

        # Live-point accumulation for the currently active scan
        self._current_tab  = "BRG2"
        self._live_xs: list   = []
        self._live_ys: list   = []
        self._coarse_xs: list = []
        self._coarse_ys: list = []
        self._fine_xs: list   = []
        self._fine_ys: list   = []

        # Per-motor pyqtgraph items (populated in _build_right_panel)
        self._plot_widgets:     dict = {}
        self._data_items:       dict = {}
        self._fine_data_items:  dict = {}
        self._fit_items:        dict = {}
        self._peak_lines:       dict = {}
        self._param_items:      dict = {}

        # Demo animation state
        self._demo_timer   = None
        self._demo_pts     = []
        self._demo_result  = None
        self._demo_idx     = 0

        # CSV viewer state
        self._csv_path: str = ""
        self._color_col: str = ""
        self._color_rules: list = []

        # Loop state
        self._loop_active          = False
        self._loop_abort           = False
        self._loop_count           = 0
        self._loop_max             = 1
        self._loop_rows            = []
        self._loop_kwargs          = {}
        self._loop_simulate        = False
        self._running_list_items   = []   # selected QListWidgetItems for current run

        self._build_ui()
        self._restore_last_csv()
        self._restore_color_settings()

        # Push setup-tab changes to the running worker between rows
        self._setup_tab.config_changed.connect(self._on_setup_changed)

        # Keep energy row list in sync with the energy table
        self._energy_tab.rows_changed.connect(self._refresh_row_list)

    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

    # ── Left: controls ───────────────────────────────────────────────────────

    def _build_controls(self):
        panel = QWidget()
        vbox = QVBoxLayout(panel)
        vbox.setSpacing(8)

        mode_grp = QGroupBox("Mode")
        mv = QVBoxLayout(mode_grp)
        self._sim_cb = QCheckBox("Simulation (no EPICS)")
        self._sim_cb.setChecked(True)
        self._sim_cb.setToolTip(
            "When checked, all scans run in simulation mode — no EPICS hardware needed."
        )
        mv.addWidget(self._sim_cb)
        vbox.addWidget(mode_grp)

        self._crystal_widget = _CrystalStatusWidget(settings=self._settings, parent=self)
        vbox.addWidget(self._crystal_widget)

        energy_grp = QGroupBox("Energy rows to align")
        ev = QVBoxLayout(energy_grp)
        self._row_list = QListWidget()
        self._row_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self._row_list.setMaximumHeight(180)
        ev.addWidget(self._row_list)
        row_btns = QHBoxLayout()
        for label, slot in [("All",  self._select_all_rows),
                             ("None", self._select_no_rows),
                             ("Refresh", self._refresh_row_list)]:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            if label == "Refresh":
                btn.setToolTip("Reload from Energy Table tab")
            row_btns.addWidget(btn)
        ev.addLayout(row_btns)
        vbox.addWidget(energy_grp)

        run_grp = QGroupBox("Run")
        rv = QVBoxLayout(run_grp)

        self._start_btn = QPushButton("Start Alignment")
        self._start_btn.setMinimumHeight(36)
        self._start_btn.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white;"
            " font-weight: bold; border-radius: 4px; }"
            "QPushButton:hover { background-color: #43a047; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self._start_btn.clicked.connect(self._start_alignment)
        rv.addWidget(self._start_btn)

        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setMinimumHeight(30)
        self._abort_btn.setEnabled(False)
        self._abort_btn.setStyleSheet(
            "QPushButton { background-color: #c62828; color: white; border-radius: 4px; }"
            "QPushButton:hover { background-color: #e53935; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self._abort_btn.clicked.connect(self._abort_alignment)
        rv.addWidget(self._abort_btn)

        self._move_btn = QPushButton("Move to Energy")
        self._move_btn.setToolTip(
            "Move all motors to the selected energy row positions without running any scans"
        )
        self._move_btn.clicked.connect(self._move_to_energy)
        rv.addWidget(self._move_btn)

        demo_btn = QPushButton("Demo Scan (sim)")
        demo_btn.setToolTip(
            "Run a simulated BRG2 scan and animate its data points in the BRG2 tab"
        )
        demo_btn.clicked.connect(self._demo_scan)
        rv.addWidget(demo_btn)

        loop_row = QHBoxLayout()
        self._loop_cb = QCheckBox("Loop")
        self._loop_cb.setToolTip("Repeat the full alignment sequence multiple times")
        loop_row.addWidget(self._loop_cb)
        loop_row.addWidget(QLabel("Iterations:"))
        self._loop_iter_edit = QLineEdit("0")
        self._loop_iter_edit.setFixedWidth(48)
        self._loop_iter_edit.setValidator(QIntValidator(0, 9999, self))
        self._loop_iter_edit.setEnabled(False)
        self._loop_iter_edit.setToolTip("Number of full-sequence repetitions (0 = run until Abort)")
        self._loop_cb.toggled.connect(self._loop_iter_edit.setEnabled)
        loop_row.addWidget(self._loop_iter_edit)
        loop_row.addWidget(QLabel("(0 = ∞)"))
        loop_row.addStretch()
        rv.addLayout(loop_row)

        per_e_row = QHBoxLayout()
        per_e_row.addWidget(QLabel("Repeats per energy:"))
        self._per_e_edit = QLineEdit("1")
        self._per_e_edit.setFixedWidth(48)
        self._per_e_edit.setValidator(QIntValidator(1, 999, self))
        self._per_e_edit.setToolTip(
            "How many times to align each energy before moving to the next.\n"
            "E.g. 3 → [E1, E1, E1, E2, E2, E2, …]"
        )
        per_e_row.addWidget(self._per_e_edit)
        per_e_row.addStretch()
        rv.addLayout(per_e_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        rv.addWidget(self._progress)

        self._status_lbl = QLabel("Ready")
        self._status_lbl.setWordWrap(True)
        rv.addWidget(self._status_lbl)
        vbox.addWidget(run_grp)

        self._hold_widget = HoldConditionsWidget(self._settings)
        self._hold_widget.suspend_triggered.connect(self._on_worker_suspend)
        self._hold_widget.suspend_cleared.connect(self._on_worker_resume)
        vbox.addWidget(self._hold_widget)

        vbox.addStretch()
        self._refresh_row_list()
        return panel

    def _refresh_row_list(self):
        # Preserve current selection by label so edits don't reset it.
        selected = {self._row_list.item(i).text()
                    for i in range(self._row_list.count())
                    if self._row_list.item(i).isSelected()}
        self._row_list.clear()
        for i, label in enumerate(self._energy_tab.get_row_labels()):
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, i)
            self._row_list.addItem(item)
        if selected:
            for i in range(self._row_list.count()):
                self._row_list.item(i).setSelected(
                    self._row_list.item(i).text() in selected)
        else:
            self._select_all_rows()

    def _select_all_rows(self):
        for i in range(self._row_list.count()):
            self._row_list.item(i).setSelected(True)

    def _select_no_rows(self):
        self._row_list.clearSelection()

    # ── Right: per-motor plot tabs + bottom tab (Results | Log) ─────────────

    def _build_right_panel(self):
        panel = QWidget()
        vbox = QVBoxLayout(panel)
        vbox.setSpacing(4)

        right_split = QSplitter(Qt.Orientation.Vertical)

        # ── Plot tabs ──────────────────────────────────────────────────────────
        if _PG:
            self._plot_tabs = QTabWidget()
            for name in _MOTOR_TABS:
                pw = pg.PlotWidget()
                pw.setBackground(_BG_COLOR)
                pw.setLabel("bottom", "Position")
                pw.setLabel("left", "Signal")
                pw.showGrid(x=True, y=True, alpha=0.25)
                pw.setMinimumHeight(200)

                data_item = pw.plot(
                    [], [], pen=None,
                    symbol="o", symbolSize=7,
                    symbolBrush=pg.mkBrush(_POINT_COLOR),
                    symbolPen=pg.mkPen(_POINT_COLOR),
                )
                fine_item = pw.plot(
                    [], [], pen=None,
                    symbol="o", symbolSize=7,
                    symbolBrush=pg.mkBrush(_FINE_POINT_COLOR),
                    symbolPen=pg.mkPen(_FINE_POINT_COLOR),
                )
                fit_item = pw.plot([], [], pen=pg.mkPen(_FIT_COLOR, width=2))
                peak_line = pg.InfiniteLine(
                    angle=90, movable=False,
                    pen=pg.mkPen(_PEAK_COLOR, width=1,
                                 style=Qt.PenStyle.DashLine),
                )
                peak_line.setVisible(False)   # hidden until a fit is drawn
                pw.addItem(peak_line)
                pw.getViewBox().disableAutoRange()
                pw.setXRange(-1, 1)
                pw.setYRange(0, 1)

                param_item = pg.TextItem(
                    text="", anchor=(1.0, 0.0),
                    color=(210, 210, 210),
                )
                param_item.setFont(QFont("Menlo" if "darwin" in __import__("sys").platform
                                         else "Consolas", 9))
                pw.addItem(param_item)

                self._plot_widgets[name]    = pw
                self._data_items[name]      = data_item
                self._fine_data_items[name] = fine_item
                self._fit_items[name]       = fit_item
                self._peak_lines[name]      = peak_line
                self._param_items[name]     = param_item

                self._plot_tabs.addTab(pw, name)

            right_split.addWidget(self._plot_tabs)
        else:
            lbl = QLabel("pyqtgraph not installed — install it for live scan plots")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #888;")
            right_split.addWidget(lbl)

        # ── Bottom tab widget: Results | Log ───────────────────────────────────
        self._bottom_tabs = QTabWidget()

        # Results tab
        res_frame = QWidget()
        res_v = QVBoxLayout(res_frame)
        res_v.setContentsMargins(4, 4, 4, 4)
        res_hdr = QHBoxLayout()
        res_hdr.addStretch()
        clear_res_btn = QPushButton("Clear")
        clear_res_btn.setMaximumWidth(60)
        clear_res_btn.clicked.connect(lambda: self._results_table.setRowCount(0))
        res_hdr.addWidget(clear_res_btn)
        res_v.addLayout(res_hdr)

        self._results_table = QTableWidget(0, len(_RES_COLS))
        self._results_table.setHorizontalHeaderLabels(_RES_COLS)
        hdr = self._results_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hdr.setStretchLastSection(False)
        for c, width in enumerate([30, 90, 90, 90, 90, 40]):
            self._results_table.setColumnWidth(c, width)
        self._results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        res_v.addWidget(self._results_table)
        self._bottom_tabs.addTab(res_frame, "Results")

        # Log tab
        log_frame = QWidget()
        lv = QVBoxLayout(log_frame)
        lv.setContentsMargins(4, 4, 4, 4)
        log_hdr = QHBoxLayout()
        log_hdr.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.setMaximumWidth(60)
        clear_btn.clicked.connect(self._clear_log)
        log_hdr.addWidget(clear_btn)
        lv.addLayout(log_hdr)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(10000)
        font = QFont("Menlo" if "darwin" in __import__("sys").platform else "Consolas", 10)
        self._log.setFont(font)
        lv.addWidget(self._log)
        self._log_tab = log_frame
        self._bottom_tabs.addTab(log_frame, "Log")

        # CSV tab
        csv_frame = QWidget()
        cv = QVBoxLayout(csv_frame)
        cv.setContentsMargins(4, 4, 4, 4)
        csv_hdr = QHBoxLayout()
        self._csv_path_lbl = QLabel("No file yet")
        self._csv_path_lbl.setStyleSheet("color: #888; font-size: 10px;")
        self._csv_path_lbl.setWordWrap(True)
        csv_hdr.addWidget(self._csv_path_lbl, 1)
        open_csv_btn = QPushButton("Open CSV…")
        open_csv_btn.setToolTip(
            "Open an existing CSV file to append new results to it.\n"
            "New record PV columns are added automatically (existing rows → current PV value).\n"
            "Removed record PV columns are kept (new rows → 0)."
        )
        open_csv_btn.clicked.connect(self._open_csv)
        csv_hdr.addWidget(open_csv_btn)
        add_col_btn = QPushButton("Add Column…")
        add_col_btn.setToolTip(
            "Add a new column to the CSV file.\n"
            "All existing rows are filled with the current value of the chosen PV."
        )
        add_col_btn.clicked.connect(self._add_csv_column)
        csv_hdr.addWidget(add_col_btn)
        del_btn = QPushButton("Delete Row(s)")
        del_btn.setToolTip("Permanently remove selected rows from the CSV file")
        del_btn.setStyleSheet(
            "QPushButton { color: #ef5350; }"
            "QPushButton:hover { color: #e53935; }"
        )
        del_btn.clicked.connect(self._delete_csv_rows)
        csv_hdr.addWidget(del_btn)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setMaximumWidth(65)
        refresh_btn.clicked.connect(self._refresh_csv)
        csv_hdr.addWidget(refresh_btn)
        analyze_btn = QPushButton("Analyze…")
        analyze_btn.setToolTip(
            "Open the statistical analysis dialog for the current CSV file:\n"
            "plots, correlation matrix, per-energy group stats, drift analysis."
        )
        analyze_btn.clicked.connect(self._analyze_csv)
        csv_hdr.addWidget(analyze_btn)
        cv.addLayout(csv_hdr)

        # Color coding row
        color_hdr = QHBoxLayout()
        color_hdr.addWidget(QLabel("Color by:"))
        self._color_col_combo = QComboBox()
        self._color_col_combo.setMinimumWidth(120)
        self._color_col_combo.setToolTip("Select a column whose values drive row color coding")
        self._color_col_combo.currentTextChanged.connect(self._on_color_col_changed)
        color_hdr.addWidget(self._color_col_combo)
        edit_rules_btn = QPushButton("Edit Rules…")
        edit_rules_btn.setToolTip("Define conditions and colors for row highlighting")
        edit_rules_btn.clicked.connect(self._edit_color_rules)
        color_hdr.addWidget(edit_rules_btn)
        clear_color_btn = QPushButton("Clear Coloring")
        clear_color_btn.clicked.connect(self._clear_coloring)
        color_hdr.addWidget(clear_color_btn)
        # Legend: colored chips for active rules
        self._color_legend_layout = QHBoxLayout()
        self._color_legend_layout.setContentsMargins(8, 0, 0, 0)
        self._color_legend_layout.setSpacing(4)
        color_hdr.addLayout(self._color_legend_layout, 1)
        cv.addLayout(color_hdr)

        self._csv_table = QTableWidget(0, 0)
        self._csv_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._csv_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._csv_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self._csv_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self._csv_table.setFont(
            QFont("Menlo" if "darwin" in __import__("sys").platform else "Consolas", 9)
        )
        cv.addWidget(self._csv_table)
        self._bottom_tabs.addTab(csv_frame, "CSV")

        right_split.addWidget(self._bottom_tabs)

        right_split.setStretchFactor(0, 3)   # plots — largest
        right_split.setStretchFactor(1, 2)   # bottom tabs
        vbox.addWidget(right_split)
        return panel

    def _clear_log(self):
        self._log.clear()

    def _save_csv_path(self):
        if self._settings and self._csv_path:
            self._settings.setValue("last_csv_path", self._csv_path)

    def _restore_last_csv(self):
        import os
        if not self._settings:
            return
        path = self._settings.value("last_csv_path", "")
        if path and os.path.isfile(path):
            self._csv_path = path
            self._csv_path_lbl.setText(path)
            self._refresh_csv()

    def _open_csv(self):
        """Let the user pick an existing CSV to append to."""
        import csv, os
        path, _ = QFileDialog.getOpenFileName(
            self, "Open existing CSV file", "", "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return

        # Read existing headers
        try:
            with open(path, newline="") as f:
                existing_fields = csv.DictReader(f).fieldnames or []
        except Exception as e:
            QMessageBox.critical(self, "Open CSV", f"Could not read file:\n{e}")
            return

        # Require the mandatory base columns to be present
        base = ["datetime", "MonoE", "Harmonic", "UndE", "Roll2", "X2"]
        missing = [c for c in base if c not in existing_fields]
        if missing:
            QMessageBox.critical(
                self, "Incompatible CSV",
                f"The file is missing required columns:\n{', '.join(missing)}\n\n"
                f"Choose a file produced by this application."
            )
            return

        # Build expected headers from current record_pvs
        kwargs = self._setup_tab.get_kwargs()
        rpvs   = kwargs.get("record_pvs") or {}
        expected = base + list(rpvs.keys())

        # Inform the user if columns differ — the backend will merge them automatically
        added   = [c for c in expected      if c not in existing_fields]
        removed = [c for c in existing_fields if c not in expected and c not in base]
        notes   = []
        if added:
            notes.append(f"New column(s) will be added (existing rows → 0): {', '.join(added)}")
        if removed:
            notes.append(f"Column(s) no longer in Record PVs (kept, new rows → 0): {', '.join(removed)}")

        # Adopt this file as the output target
        self._csv_path = os.path.abspath(path)
        self._csv_path_lbl.setText(self._csv_path)
        self._setup_tab.set_output_filename(self._csv_path)
        self._save_csv_path()
        self._refresh_csv()
        # Switch to CSV tab so user can see the loaded data
        self._bottom_tabs.setCurrentIndex(
            self._bottom_tabs.indexOf(self._csv_table.parent())
        )
        note_text = ("\n\n" + "\n".join(notes)) if notes else ""
        QMessageBox.information(
            self, "CSV opened",
            f"Loaded {len(existing_fields)} columns, "
            f"{self._csv_table.rowCount()} existing row(s).\n"
            f"New alignment results will be appended to:\n{self._csv_path}"
            f"{note_text}"
        )

    def _refresh_csv(self):
        """Read the current CSV file and populate the CSV table."""
        import csv, os
        path = self._csv_path
        if not path or not os.path.isfile(path):
            return
        try:
            with open(path, newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
        except Exception:
            return
        if not rows:
            return
        headers = rows[0]
        data    = rows[1:]
        self._csv_table.setColumnCount(len(headers))
        self._csv_table.setHorizontalHeaderLabels(headers)
        self._csv_table.setRowCount(len(data))
        for r, row in enumerate(data):
            for c, val in enumerate(row):
                item = QTableWidgetItem(val.strip())
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._csv_table.setItem(r, c, item)
        self._csv_table.scrollToBottom()
        self._csv_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)

        # Update color-by combobox, preserving the current selection if still valid
        if hasattr(self, '_color_col_combo'):
            prev = self._color_col_combo.currentText()
            self._color_col_combo.blockSignals(True)
            self._color_col_combo.clear()
            self._color_col_combo.addItem("")
            self._color_col_combo.addItems(headers)
            idx = self._color_col_combo.findText(prev)
            self._color_col_combo.setCurrentIndex(max(0, idx))
            self._color_col_combo.blockSignals(False)
            self._apply_csv_coloring()
            self._update_color_legend()

    def _delete_csv_rows(self):
        """Permanently remove selected rows from the CSV file."""
        import csv, os
        selected = sorted({idx.row() for idx in self._csv_table.selectedIndexes()}, reverse=True)
        if not selected:
            QMessageBox.information(self, "Delete Rows", "No rows selected.")
            return
        path = self._csv_path
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "Delete Rows", "No CSV file is currently open.")
            return
        reply = QMessageBox.question(
            self, "Delete Rows",
            f"Permanently delete {len(selected)} row(s) from:\n{path}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            with open(path, newline="") as f:
                rows = list(csv.reader(f))
        except Exception as e:
            QMessageBox.critical(self, "Delete Rows", f"Could not read file:\n{e}")
            return
        if not rows:
            return
        header = rows[0]
        data = rows[1:]
        keep_indices = set(range(len(data))) - set(selected)
        data_to_keep = [r for i, r in enumerate(data) if i in keep_indices]
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(data_to_keep)
        except Exception as e:
            QMessageBox.critical(self, "Delete Rows", f"Could not write file:\n{e}")
            return
        self._refresh_csv()

    def _analyze_csv(self):
        from ._stats_dialog import StatsDialog
        dlg = StatsDialog(csv_path=self._csv_path, energy_tab=self._energy_tab, parent=self)
        dlg.exec()

    def _add_csv_column(self):
        """Prompt for a column label + PV, caget the current value, backfill all rows."""
        import os
        from PyQt6.QtWidgets import (QDialog, QFormLayout, QDialogButtonBox,
                                     QCheckBox)
        from .smart_scan_functions import caget, add_csv_column

        path = self._csv_path
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "Add Column",
                                "No CSV file is currently open.\n"
                                "Use 'Open CSV…' first.")
            return

        # ── Dialog ────────────────────────────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle("Add Column to CSV")
        dlg.setMinimumWidth(360)
        form = QFormLayout(dlg)

        label_edit = QLineEdit()
        label_edit.setPlaceholderText("e.g. ring_current")
        form.addRow("Column label:", label_edit)

        pv_edit = QLineEdit()
        pv_edit.setPlaceholderText("e.g. S:SRcurrentAI")
        form.addRow("PV name:", pv_edit)

        also_record_cb = QCheckBox("Also add to Record PVs in Setup tab")
        also_record_cb.setToolTip(
            "If checked, this PV will also be appended to the Record PVs table\n"
            "so future alignment rows record it automatically."
        )
        form.addRow("", also_record_cb)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        label  = label_edit.text().strip()
        pv_name = pv_edit.text().strip()

        if not label:
            QMessageBox.warning(self, "Add Column", "Column label cannot be empty.")
            return
        if not pv_name:
            QMessageBox.warning(self, "Add Column", "PV name cannot be empty.")
            return

        # ── Read current PV value ─────────────────────────────────────────────
        live = caget(pv_name)
        if live is None:
            reply = QMessageBox.question(
                self, "Add Column",
                f"Could not read PV '{pv_name}' (EPICS unavailable or timeout).\n\n"
                f"Fill existing rows with 0 instead?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            fill_value = "0"
        else:
            fill_value = str(live)

        # ── Write column ──────────────────────────────────────────────────────
        err = add_csv_column(path, label, fill_value)
        if err:
            QMessageBox.critical(self, "Add Column", f"Failed to add column:\n{err}")
            return

        # ── Optionally add to Record PVs table ────────────────────────────────
        if also_record_cb.isChecked():
            self._setup_tab._add_record_row(label, pv_name)

        self._refresh_csv()
        QMessageBox.information(
            self, "Add Column",
            f"Column '{label}' added.\n"
            f"All {self._csv_table.rowCount()} existing rows filled with {fill_value}."
        )

    # ── CSV color coding ─────────────────────────────────────────────────────

    def _on_color_col_changed(self, col):
        self._color_col = col
        self._apply_csv_coloring()
        self._save_color_settings()

    def _apply_csv_coloring(self):
        from PyQt6.QtGui import QBrush, QColor
        # Clear all row backgrounds
        for r in range(self._csv_table.rowCount()):
            for c in range(self._csv_table.columnCount()):
                item = self._csv_table.item(r, c)
                if item:
                    item.setBackground(QBrush())

        col = self._color_col_combo.currentText() if hasattr(self, '_color_col_combo') else ""
        if not col or not self._color_rules:
            return

        col_idx = next(
            (c for c in range(self._csv_table.columnCount())
             if self._csv_table.horizontalHeaderItem(c) and
                self._csv_table.horizontalHeaderItem(c).text() == col),
            -1,
        )
        if col_idx < 0:
            return

        for r in range(self._csv_table.rowCount()):
            val_item = self._csv_table.item(r, col_idx)
            val_str  = val_item.text() if val_item else ""
            for rule in self._color_rules:
                if self._eval_rule(val_str, rule):
                    brush = QBrush(QColor(rule.get("color", "#ffffff")))
                    for c in range(self._csv_table.columnCount()):
                        item = self._csv_table.item(r, c)
                        if item:
                            item.setBackground(brush)
                    break

    @staticmethod
    def _eval_rule(val_str, rule):
        try:
            v = float(val_str)
        except (TypeError, ValueError):
            return False
        op = rule.get("op", ">")
        try:
            t = float(rule.get("value", 0))
        except (TypeError, ValueError):
            return False
        if op in ("in range", "out of range"):
            try:
                t2 = float(rule.get("value2", 0))
            except (TypeError, ValueError):
                return False
            return (t <= v <= t2) if op == "in range" else (v < t or v > t2)
        if op == ">":  return v > t
        if op == "<":  return v < t
        if op == ">=": return v >= t
        if op == "<=": return v <= t
        if op == "=":  return v == t
        if op == "!=": return v != t
        return False

    def _update_color_legend(self):
        from PyQt6.QtGui import QColor
        while self._color_legend_layout.count():
            item = self._color_legend_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        col = self._color_col_combo.currentText() if hasattr(self, '_color_col_combo') else ""
        if not col or not self._color_rules:
            return

        _OP_SYMS = {
            ">": ">", "<": "<", ">=": "≥", "<=": "≤",
            "=": "=", "!=": "≠",
            "in range": "in", "out of range": "out",
        }
        for rule in self._color_rules:
            op        = rule.get("op", ">")
            val       = rule.get("value", "")
            val2      = rule.get("value2", "")
            hex_color = rule.get("color", "#ffffff")
            user_lbl  = rule.get("label", "")
            sym       = _OP_SYMS.get(op, op)
            cond_text = f"{val} {sym} {val2}" if op in ("in range", "out of range") else f"{sym} {val}"
            chip_text = user_lbl if user_lbl else cond_text
            c         = QColor(hex_color)
            lum       = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
            fg        = "#000000" if lum > 128 else "#ffffff"
            chip = QLabel(f"  {chip_text}  ")
            chip.setStyleSheet(
                f"background-color: {hex_color}; color: {fg}; "
                "border-radius: 3px; padding: 1px 4px; font-size: 10px;"
            )
            chip.setToolTip(f"value {op} {val}" + (f" .. {val2}" if val2 else ""))
            self._color_legend_layout.addWidget(chip)

        self._color_legend_layout.addStretch()

    def _clear_coloring(self):
        if hasattr(self, '_color_col_combo'):
            self._color_col_combo.blockSignals(True)
            self._color_col_combo.setCurrentIndex(0)
            self._color_col_combo.blockSignals(False)
        self._color_col   = ""
        self._color_rules = []
        self._apply_csv_coloring()
        self._update_color_legend()
        self._save_color_settings()

    def _edit_color_rules(self):
        col = self._color_col_combo.currentText() if hasattr(self, '_color_col_combo') else ""
        if not col:
            QMessageBox.information(self, "Edit Rules",
                                    "Please select a 'Color by' column first.")
            return
        dlg = _ColorRuleDialog(rules=list(self._color_rules), parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._color_rules = dlg.get_rules()
            self._apply_csv_coloring()
            self._update_color_legend()
            self._save_color_settings()

    def _save_color_settings(self):
        import json
        if not self._settings:
            return
        self._settings.setValue("csv_color_col", self._color_col)
        self._settings.setValue("csv_color_rules", json.dumps(self._color_rules))

    def _restore_color_settings(self):
        import json
        if not self._settings:
            return
        self._color_col  = self._settings.value("csv_color_col", "")
        rules_raw        = self._settings.value("csv_color_rules", "[]")
        try:
            self._color_rules = json.loads(rules_raw) if isinstance(rules_raw, str) else []
        except Exception:
            self._color_rules = []
        if hasattr(self, '_color_col_combo') and self._color_col:
            idx = self._color_col_combo.findText(self._color_col)
            if idx >= 0:
                self._color_col_combo.blockSignals(True)
                self._color_col_combo.setCurrentIndex(idx)
                self._color_col_combo.blockSignals(False)
        self._apply_csv_coloring()
        self._update_color_legend()

    # ── Alignment control ────────────────────────────────────────────────────

    def _get_selected_rows(self):
        all_rows = self._energy_tab.get_table()
        selected_indices = [
            self._row_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self._row_list.count())
            if self._row_list.item(i).isSelected()
        ]
        return [all_rows[i] for i in selected_indices if i < len(all_rows)]

    def _start_alignment(self):
        rows = self._get_selected_rows()
        if not rows:
            self._log.appendPlainText("No energy rows selected.")
            return

        simulate = self._sim_cb.isChecked()
        kwargs   = self._setup_tab.get_kwargs()

        # Track CSV path for the viewer
        import os
        raw_path = kwargs.get("filename", "alignment_results.csv")
        self._csv_path = os.path.abspath(raw_path)
        self._csv_path_lbl.setText(self._csv_path)
        self._save_csv_path()

        # Per-energy repeat: expand row list before passing to worker
        try:
            per_e = max(1, int(self._per_e_edit.text()))
        except ValueError:
            per_e = 1
        self._per_e_repeat = per_e
        expanded = [row for row in rows for _ in range(per_e)]

        # Loop state
        self._loop_active   = self._loop_cb.isChecked()
        self._loop_abort    = False
        self._loop_count    = 0
        self._loop_rows     = expanded
        self._loop_kwargs   = kwargs
        self._loop_simulate = simulate
        try:
            self._loop_max = int(self._loop_iter_edit.text())
        except ValueError:
            self._loop_max = 0

        # Snapshot selected list items for row highlighting during the run
        self._running_list_items = [
            self._row_list.item(i)
            for i in range(self._row_list.count())
            if self._row_list.item(i).isSelected()
        ]

        # Clear results table and switch to Log tab for the new run
        self._results_table.setRowCount(0)
        self._bottom_tabs.setCurrentWidget(self._log_tab)

        self._log_loop_header(iteration=1)
        self._launch_worker()

    def _log_loop_header(self, iteration: int):
        simulate   = self._loop_simulate
        n_unique   = len(self._loop_rows) // max(self._per_e_repeat, 1)
        total      = str(self._loop_max) if self._loop_max > 0 else "∞"
        iter_str   = f"  Loop {iteration}/{total}\n" if self._loop_active else ""
        repeat_str = (f"  {self._per_e_repeat} repeat(s) per energy\n"
                      if self._per_e_repeat > 1 else "")
        self._log.appendPlainText(
            f"\n{'═'*60}\n"
            f"  Starting alignment  {'[SIMULATION]' if simulate else '[EPICS]'}\n"
            f"{iter_str}"
            f"  {n_unique} energy row(s) selected\n"
            f"{repeat_str}"
            f"{'═'*60}"
        )

    def _launch_worker(self):
        from ._worker import AlignWorker
        self._worker = AlignWorker(
            self._loop_rows, self._loop_kwargs, self._loop_simulate,
            parent=self,
        )
        self._worker.log_chunk.connect(self._on_log)
        self._worker.scan_started.connect(self._on_scan_started)
        self._worker.point_measured.connect(self._on_point_measured)
        self._worker.fine_point_measured.connect(self._on_fine_point_measured)
        self._worker.scan_finished.connect(self._on_scan_finished)
        self._worker.step_update.connect(self._on_step_update)
        self._worker.row_done.connect(self._on_row_done)
        self._worker.row_started.connect(self._on_row_started)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.hold_triggered.connect(self._on_hold_triggered)
        self._worker.hold_cleared.connect(self._on_hold_cleared)

        self._start_btn.setEnabled(False)
        self._abort_btn.setEnabled(True)
        self._move_btn.setEnabled(False)
        self._per_e_edit.setEnabled(False)
        self._progress.setRange(0, 0)
        self._progress.setVisible(True)
        iter_str = ""
        if self._loop_active:
            total = str(self._loop_max) if self._loop_max > 0 else "∞"
            iter_str = f" (loop {self._loop_count + 1}/{total})"
        self._status_lbl.setText(f"Running…{iter_str}")
        self.status_message.emit(f"Alignment running…{iter_str}")
        self._worker.start()
        # Pre-sync: if conditions were already active before alignment started,
        # immediately suspend the worker without waiting for the next 6s poll.
        self._hold_widget.sync_worker(self._worker)

    def _abort_alignment(self):
        self._loop_abort = True
        if self._worker and self._worker.isRunning():
            self._worker.abort()
            if not self._worker.wait(4000):
                self._worker.terminate()
            self._log.appendPlainText("\n⚠ Alignment aborted by user.")
            self._status_lbl.setText("Aborted")
            self.status_message.emit("Alignment aborted")
        self._loop_active = False
        self._reset_buttons()

    def _reset_buttons(self):
        self._start_btn.setEnabled(True)
        self._abort_btn.setEnabled(False)
        self._move_btn.setEnabled(True)
        self._per_e_edit.setEnabled(True)
        self._progress.setVisible(False)
        from PyQt6.QtGui import QBrush
        for item in self._running_list_items:
            f = item.font()
            f.setBold(False)
            item.setFont(f)
            item.setForeground(QBrush())
        self._running_list_items = []

    def _move_to_energy(self):
        """Move all motors to the first selected energy row without running alignment scans."""
        selected = self._row_list.selectedItems()
        if not selected:
            self._log.appendPlainText("[MoveToEnergy] No energy row selected.")
            return

        idx = selected[0].data(Qt.ItemDataRole.UserRole)
        all_rows = self._energy_tab.get_table()
        if idx >= len(all_rows):
            self._log.appendPlainText("[MoveToEnergy] Selected row index out of range.")
            return
        row    = all_rows[idx]
        kwargs = self._setup_tab.get_kwargs()
        simulate = self._sim_cb.isChecked()

        self._move_btn.setEnabled(False)
        self._bottom_tabs.setCurrentWidget(self._log_tab)

        thread = _MoveToEnergyThread(row, kwargs, simulate, parent=self)
        thread.log_chunk.connect(self._on_log)
        thread.done.connect(lambda: self._move_btn.setEnabled(True))
        thread.done.connect(thread.deleteLater)
        thread.start()

    # ── Worker signal handlers ────────────────────────────────────────────────

    def _on_log(self, text):
        cursor = self._log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self._log.setTextCursor(cursor)
        self._log.ensureCursorVisible()

    def _on_step_update(self, label: str, current: int, total: int):
        self._progress.setRange(0, total)
        self._progress.setValue(current)
        self._status_lbl.setText(f"[{current}/{total}] {label}")
        self.status_message.emit(f"{label}…")

    def _on_row_started(self, row_idx: int):
        per_e    = max(1, getattr(self, '_per_e_repeat', 1))
        list_idx = row_idx // per_e
        from PyQt6.QtGui import QBrush, QColor
        active_color = QColor("#ffa726")   # amber — visible over selection blue
        for i, item in enumerate(self._running_list_items):
            bold = (i == list_idx)
            f = item.font()
            f.setBold(bold)
            item.setFont(f)
            item.setForeground(QBrush(active_color) if bold else QBrush())

    def _on_row_done(self, record: dict):
        r = self._results_table.rowCount()
        self._results_table.insertRow(r)
        ok = record.get("_row_ok", True)
        for c, val in enumerate([
            str(r + 1),
            _fmt(record.get("MonoE"), 6),
            _fmt(record.get("_brg2_center")),
            _fmt(record.get("Roll2")),
            _fmt(record.get("X2")),
            "✓" if ok else "✗",
        ]):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if c == 5:
                item.setForeground(
                    __import__("PyQt6.QtGui", fromlist=["QColor"]).QColor(
                        "#66bb6a" if ok else "#ef5350"
                    )
                )
            self._results_table.setItem(r, c, item)
        self._results_table.scrollToBottom()
        self._refresh_csv()

        # Update Roll2 and X2 in the energy table — skip entirely in simulation mode.
        mono_e = record.get("MonoE")
        roll2  = record.get("Roll2")
        x2     = record.get("X2")
        if (not self._loop_simulate
                and mono_e is not None and roll2 is not None and x2 is not None):
            from datetime import datetime
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._energy_tab.update_row_after_alignment(float(mono_e), float(roll2), float(x2), ts)

    def _on_scan_started(self, tab_name: str):
        """Switch to the named motor tab and clear its plot for a new scan."""
        self._current_tab = tab_name
        self._live_xs   = []
        self._live_ys   = []
        self._coarse_xs = []
        self._coarse_ys = []
        self._fine_xs   = []
        self._fine_ys   = []
        if not _PG or tab_name not in self._data_items:
            return
        idx = _MOTOR_TABS.index(tab_name) if tab_name in _MOTOR_TABS else 0
        self._plot_tabs.setCurrentIndex(idx)
        self._data_items[tab_name].setData([], [])
        if tab_name in self._fine_data_items:
            self._fine_data_items[tab_name].setData([], [])
        self._fit_items[tab_name].setData([], [])
        self._peak_lines[tab_name].setVisible(False)
        if tab_name in self._param_items:
            self._param_items[tab_name].setText("")
        self._plot_widgets[tab_name].setTitle(f"{tab_name} — scanning…")

    def _on_point_measured(self, x: float, y: float):
        """Append one coarse-scan data point to the currently active plot tab."""
        self._live_xs.append(x)
        self._live_ys.append(y)
        self._coarse_xs.append(x)
        self._coarse_ys.append(y)
        if not _PG or self._current_tab not in self._data_items:
            return
        self._data_items[self._current_tab].setData(
            np.array(self._coarse_xs, dtype=float),
            np.array(self._coarse_ys, dtype=float),
        )
        self._fit_to_data(
            self._plot_widgets[self._current_tab],
            np.array(self._live_xs, dtype=float),
            np.array(self._live_ys, dtype=float),
        )

    def _on_fine_point_measured(self, x: float, y: float):
        """Append one fine-scan data point to the currently active plot tab."""
        self._live_xs.append(x)
        self._live_ys.append(y)
        self._fine_xs.append(x)
        self._fine_ys.append(y)
        if not _PG or self._current_tab not in self._fine_data_items:
            return
        self._fine_data_items[self._current_tab].setData(
            np.array(self._fine_xs, dtype=float),
            np.array(self._fine_ys, dtype=float),
        )
        self._fit_to_data(
            self._plot_widgets[self._current_tab],
            np.array(self._live_xs, dtype=float),
            np.array(self._live_ys, dtype=float),
        )

    def _on_scan_finished(self, result):
        """Overlay the fit curve on the current tab after a scan completes."""
        if not _PG:
            return
        tab = self._current_tab
        status_str = result.status.value if hasattr(result.status, "value") else str(result.status)
        if result.status == ScanStatus.SUCCESS:
            self._draw_fit(tab, result)
        if tab in self._plot_widgets:
            self._plot_widgets[tab].setTitle(f"{tab} — {status_str}")

    def _draw_fit(self, tab_name: str, result):
        if tab_name not in self._fit_items:
            return
        if self._live_xs:
            pos = np.array(self._live_xs, dtype=float)
            sig = np.array(self._live_ys, dtype=float)
            self._data_items[tab_name].setData(
                np.array(self._coarse_xs, dtype=float),
                np.array(self._coarse_ys, dtype=float),
            )
            if tab_name in self._fine_data_items:
                self._fine_data_items[tab_name].setData(
                    np.array(self._fine_xs, dtype=float),
                    np.array(self._fine_ys, dtype=float),
                )
        else:
            pos = np.asarray(result.positions, dtype=float)
            sig = np.asarray(result.signals,   dtype=float)
            self._data_items[tab_name].setData(pos, sig)

        if (result.center is not None
                and result.sigma is not None
                and result.amplitude is not None):
            fit_xs = np.linspace(pos.min(), pos.max(), 400)
            offset = result.offset or 0.0
            sigma  = result.sigma
            amp    = result.amplitude
            cen    = result.center
            prof   = result.profile or "gaussian"

            if prof == "lorentzian":
                fit_ys = amp / (1.0 + ((fit_xs - cen) / sigma) ** 2) + offset
                fwhm   = 2.0 * sigma
            elif prof == "supergaussian":
                p = (result.stats.get("supergaussian_p", 2.0)
                     if result.stats else 2.0)
                fit_ys = amp * np.exp(-np.abs((fit_xs - cen) / sigma) ** p) + offset
                fwhm   = 2.0 * sigma * (np.log(2.0) ** (1.0 / p))
            else:  # gaussian
                fit_ys = amp * np.exp(-0.5 * ((fit_xs - cen) / sigma) ** 2) + offset
                fwhm   = 2.3548 * sigma

            self._fit_items[tab_name].setData(fit_xs, fit_ys)
            self._peak_lines[tab_name].setValue(cen)
            self._peak_lines[tab_name].setVisible(True)

            # Build parameter annotation
            lines = [f"Profile : {prof}",
                     f"Center  : {cen:.6g}",
                     f"FWHM    : {fwhm:.5g}",
                     f"Sigma   : {sigma:.5g}",
                     f"Ampl    : {amp:.5g}",
                     f"Offset  : {offset:.5g}"]
            if prof == "supergaussian":
                p_val = (result.stats.get("supergaussian_p", 2.0)
                         if result.stats else 2.0)
                lines.insert(3, f"p (SG)  : {p_val:.3g}")
            if tab_name in self._param_items:
                pi = self._param_items[tab_name]
                pi.setText("\n".join(lines))
                pi.setPos(float(pos.max()), float(sig.max()))
        else:
            self._fit_items[tab_name].setData([], [])
            if tab_name in self._param_items:
                self._param_items[tab_name].setText("")

        if tab_name in self._plot_widgets:
            self._fit_to_data(self._plot_widgets[tab_name], pos, sig)

    @staticmethod
    def _fit_to_data(pw, xs, ys):
        """Set x/y range to exactly the data extent with a small padding."""
        if len(xs) < 1:
            return
        xlo, xhi = float(xs.min()), float(xs.max())
        ylo, yhi = float(ys.min()), float(ys.max())
        xpad = (xhi - xlo) * 0.12 if xhi > xlo else abs(xlo) * 0.1 + 0.01
        ypad = (yhi - ylo) * 0.15 if yhi > ylo else abs(ylo) * 0.1 + 0.1
        pw.setXRange(xlo - xpad, xhi + xpad, padding=0)
        pw.setYRange(ylo - ypad, yhi + ypad, padding=0)

    # ── Demo scan ────────────────────────────────────────────────────────────

    def _demo_scan(self):
        """Run a simulated BRG2 scan, then animate its points in the BRG2 tab."""
        if self._demo_timer and self._demo_timer.isActive():
            return  # already animating

        from .smart_scan_functions import smart_scan
        kwargs = self._setup_tab.get_kwargs()
        center = (kwargs.get("brg2_start", -0.005) + kwargs.get("brg2_stop", 0.005)) / 2
        half   = abs(kwargs.get("brg2_stop", 0.005) - kwargs.get("brg2_start", -0.005)) / 2

        result = smart_scan(
            "demo_motor", "demo_det",
            start=center - half, stop=center + half,
            nsteps=kwargs.get("brg2_nsteps", 21),
            simulate=True,
            sim_center=center, sim_sigma=half * 0.3,
            sim_amplitude=1000, sim_noise=15,
            plot=False, fine_scan=True, move_to_peak=False,
        )
        if result.status != ScanStatus.SUCCESS:
            self._log.appendPlainText(
                f"[Demo] scan failed: {result.status.value}"
            )
            return

        # Clear the BRG2 tab and animate the points one by one
        self._on_scan_started("BRG2")
        self._demo_pts    = list(zip(result.positions, result.signals))
        self._demo_result = result
        self._demo_idx    = 0

        self._demo_timer = QTimer(self)
        self._demo_timer.timeout.connect(self._demo_tick)
        self._demo_timer.start(60)   # ~60 ms per point → visually smooth

    def _demo_tick(self):
        if self._demo_idx < len(self._demo_pts):
            x, y = self._demo_pts[self._demo_idx]
            self._on_point_measured(x, y)
            self._demo_idx += 1
        else:
            self._demo_timer.stop()
            self._draw_fit("BRG2", self._demo_result)
            r = self._demo_result
            if _PG and "BRG2" in self._plot_widgets:
                self._plot_widgets["BRG2"].setTitle(
                    f"BRG2 — Demo  center={r.center:.5g}  σ={r.sigma:.5g}"
                )
            self._log.appendPlainText(
                f"[Demo] center={r.center:.5g}, sigma={r.sigma:.5g}, "
                f"amplitude={r.amplitude:.5g}"
            )

    # ── Completion / error ────────────────────────────────────────────────────

    def _on_done(self, results):
        # Abort was already handled — don't overwrite "Aborted" status.
        if self._loop_abort:
            self._loop_active = False
            return

        self._loop_count += 1
        total   = str(self._loop_max) if self._loop_max > 0 else "∞"
        loop_str = f"  Loop {self._loop_count}/{total}\n" if self._loop_active else ""

        keep_looping = (
            self._loop_active
            and self._loop_cb.isChecked()
            and not self._loop_abort
            and (self._loop_max == 0 or self._loop_count < self._loop_max)
        )

        if keep_looping:
            self._log.appendPlainText(
                f"\n{'─'*60}\n"
                f"{loop_str}"
                f"  {len(results)} row(s) processed. Starting next loop…\n"
                f"{'─'*60}"
            )
            self._status_lbl.setText(f"Loop {self._loop_count}/{total} done — restarting…")
            self.status_message.emit(f"Loop {self._loop_count}/{total} complete, restarting…")
            QTimer.singleShot(400, self._next_loop_iteration)
        else:
            if self._loop_active:
                summary = (f"{self._loop_count} loop(s)"
                           if self._loop_max == 0 or self._loop_count == self._loop_max
                           else f"{self._loop_count}/{total} loop(s)")
                self._log.appendPlainText(
                    f"\n{'═'*60}\n"
                    f"  All done — {summary}, {len(results)} row(s)/loop\n"
                    f"{'═'*60}"
                )
                self._status_lbl.setText(f"Done ({summary})")
                self.status_message.emit(f"Alignment complete: {summary}")
            else:
                self._log.appendPlainText(
                    f"\n{'═'*60}\n  Alignment complete — {len(results)} row(s) processed.\n{'═'*60}"
                )
                self._status_lbl.setText(f"Done ({len(results)} rows)")
                self.status_message.emit(f"Alignment complete: {len(results)} rows")
            self._loop_active = False
            self._reset_buttons()

    def _next_loop_iteration(self):
        if self._loop_abort or not self._loop_cb.isChecked():
            self._loop_active = False
            self._reset_buttons()
            return
        # Re-read scan parameters for each loop iteration so changes made in
        # the Setup tab during a run take effect at the next full iteration.
        # Filename and record_pvs are NOT re-read here — the worker preserves
        # them internally via update_kwargs() to keep data in one CSV file.
        new_kw = self._setup_tab.get_kwargs()
        self._loop_kwargs.update({k: v for k, v in new_kw.items()
                                  if k not in ("filename", "record_pvs")})
        self._log_loop_header(iteration=self._loop_count + 1)
        self._launch_worker()

    def _on_setup_changed(self):
        """Push updated scan parameters to a running worker between rows."""
        if self._worker and self._worker.isRunning():
            new_kw = self._setup_tab.get_kwargs()
            self._worker.update_kwargs(new_kw)
            self._loop_kwargs.update({k: v for k, v in new_kw.items()
                                      if k not in ("filename", "record_pvs")})
            self._status_lbl.setText(
                f"Settings updated — will apply at the next energy row"
            )

    def _on_error(self, tb):
        self._log.appendPlainText(f"\n✗ ERROR:\n{tb}")
        self._status_lbl.setText("Error — see log")
        self.status_message.emit("Alignment error — see log")
        self._reset_buttons()

    def _on_worker_suspend(self, msg: str):
        """Called from Qt main thread when HoldConditionsWidget detects conditions active."""
        if self._worker and self._worker.isRunning():
            self._worker.suspend(msg)

    def _on_worker_resume(self):
        """Called from Qt main thread when HoldConditionsWidget detects conditions cleared."""
        if self._worker and self._worker.isRunning():
            self._worker.resume()

    def _on_hold_triggered(self, msg: str):
        """Called from worker signal when it has actually entered the paused state."""
        self._status_lbl.setText(f"⏸ ON HOLD — {msg}")
        self._hold_widget.set_hold_active(msg)
        self.status_message.emit(f"ON HOLD: {msg}")

    def _on_hold_cleared(self):
        """Called from worker signal when it has actually resumed."""
        self._status_lbl.setText("Hold cleared — restarting row…")
        self._hold_widget.set_hold_cleared()
        self.status_message.emit("Hold cleared — restarting")

    def reload_settings(self):
        self._hold_widget.reload_settings()
        self._crystal_widget.reload_settings()
        self._restore_color_settings()

    def save_settings(self):
        self._hold_widget.save_settings()
        self._crystal_widget.save_settings()
        self._save_color_settings()
