"""Setup tab — Motors/PVs and scan parameters."""
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QWidget, QScrollArea, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLineEdit, QSpinBox, QCheckBox,
    QComboBox, QPushButton, QLabel, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView,
)


class _NoScrollSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class _NoScrollComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class _PVBridge(QObject):
    """Thread-safe bridge: CA monitor callbacks (CA thread) → Qt signals (main thread)."""
    value_changed = pyqtSignal(str, object)   # (key, value)

# ── Default PV names (ID15A2 prefix) ────────────────────────────────────────
_PV_DEFAULTS = {
    "pv_prefix":      "ID15A2:",
    "detector":       "ID15A2:det:Signal",
    "monitor_pv":     "",
    "brg2":           "ID15A2:BRG2",
    "roll1_motor":    "",
    "roll2_motor":    "ID15A2:Roll2",
    "x2_motor":       "ID15A2:X2",
    "pitch_pv":       "ID15A2:PitchPiezo:SP",
    "slit_v_pv":      "ID15A2:SlitV:SP",
    "slit_h_pv":      "ID15A2:SlitH:SP",
    "mono_e_pv":      "ID15A2:mono:Energy",
    "harmonic_pv":    "ID15A2:und:Harmonic",
    "und_e_pv":       "ID15A2:und:Energy",
    "und_start_pv":   "ID15A2:und:Start",
    "roll2_energy_pv":"ID15A2:Roll2:EnergySet",
    "x2_energy_pv":   "ID15A2:X2:EnergySet",
    "bpm_x_pv":           "",
    "bpm_y_pv":           "",
    "slit_v_center_pv":   "",
    "slit_v_top_pv":      "",
    "slit_v_bot_pv":      "",
}

_SCAN_DEFAULTS = {
    # Monitor normalization per scan
    "monitor_brg2":   True,
    "monitor_pitch":  True,
    "monitor_roll2":  True,
    "monitor_x2":     True,
    # Slits
    "slit_open_v":    0.5,
    "slit_open_h":    0.5,
    "slit_close_v":   0.05,
    "slit_close_h":   0.05,
    # BRG2
    "brg2_start":          -0.005,
    "brg2_stop":            0.005,
    "brg2_nsteps":          21,
    "brg2_min_prominence":  3.0,
    # Pitch
    "pitch_home":      5.0,
    "pitch_start":    -1.0,
    "pitch_stop":      1.0,
    "pitch_nsteps":    21,
    "pitch_settle":    0.1,
    # Roll2
    "roll2_start":    -0.005,
    "roll2_stop":      0.005,
    "roll2_nsteps":    21,
    # X2
    "x2_start":       -0.5,
    "x2_stop":         0.5,
    "x2_nsteps":       21,
    # Fine scan
    "fine_scan":       True,
    "fine_sigma_range": 3.0,
    "fine_nsteps":     21,
    "fine_scan_iter":   2,
    "dmov_delay":       0.25,
    # BPM alignment phase
    "bpm_slit_open":        10.0,
    "bpm_x_search_step":    10.0,
    "bpm_y_search_step":    0.001,
    "bpm_max_steps":        20,
    "bpm_x_tolerance":      10.0,
    "bpm_y_tolerance":      10.0,
    "bpm_refine_iter":      3,
    "bpm_slit_v_gap":       0.1,
    "bpm_slit_v_start":     -2.0,
    "bpm_slit_v_stop":       2.0,
    "bpm_slit_v_nsteps":    21,
    # Other
    "settle":           0.3,
    "energy_settle":    2.0,
    "record_settle":    2.0,
    "do_pitch_scan":        True,
    "backlash_correction":  False,
    "peak_method":          "stats",
    "stats_centre":    "centroid",
    "fit_profile":     "auto",
    "filename":        "alignment_results.csv",
}


def _dbl(val, lo=-1e6, hi=1e6, decimals=6, step=0.001):
    w = QLineEdit(str(val))
    w.setValidator(QDoubleValidator(lo, hi, decimals))
    w.setMaximumWidth(130)
    return w


def _int(val, lo=1, hi=999):
    w = _NoScrollSpinBox()
    w.setRange(lo, hi)
    w.setValue(val)
    w.setMaximumWidth(80)
    return w


def _le(text=""):
    w = QLineEdit(text)
    return w


class SetupTab(QWidget):
    config_changed = pyqtSignal()   # emitted when any scan parameter changes

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._pv_widgets = {}
        self._scan_widgets = {}
        self._rbk_labels: dict = {}
        self._rbk_pvs: dict = {}          # key → epics.PV handle
        self._bridge = _PVBridge(self)
        self._bridge.value_changed.connect(self._on_pv_value)
        self._build_ui()
        self._load_settings()
        self._subscribe_pvs()

    def _build_ui(self):
        outer = QVBoxLayout(self)

        inner_tabs = QTabWidget()
        inner_tabs.addTab(self._build_pv_page(), "Motors && PVs")
        inner_tabs.addTab(self._build_scan_page(), "Scan Parameters")
        outer.addWidget(inner_tabs)
        self._connect_scan_signals()

    def _connect_scan_signals(self):
        """Emit config_changed whenever any scan parameter widget changes."""
        for w in self._scan_widgets.values():
            if isinstance(w, QLineEdit):
                w.textChanged.connect(self.config_changed)
            elif isinstance(w, QSpinBox):
                w.valueChanged.connect(self.config_changed)
            elif isinstance(w, QCheckBox):
                w.toggled.connect(self.config_changed)
            elif isinstance(w, QComboBox):
                w.currentIndexChanged.connect(self.config_changed)

    # ── Motors & PVs page ───────────────────────────────────────────────────

    def _build_pv_page(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setSpacing(8)

        # Helper: build a [QLineEdit | live-value label] row widget and register both.
        def _pv_row(key, default, tip=""):
            w = _le(default)
            if tip:
                w.setToolTip(tip)
            self._pv_widgets[key] = w
            rbk = QLabel("—")
            rbk.setFixedWidth(110)
            rbk.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            rbk.setStyleSheet("color: #888; font-family: monospace;")
            self._rbk_labels[key] = rbk
            row_w = QWidget()
            rh = QHBoxLayout(row_w)
            rh.setContentsMargins(0, 0, 0, 0)
            rh.setSpacing(6)
            rh.addWidget(w, 1)
            rh.addWidget(rbk)
            return row_w

        # Prefix apply row (no live readback — not a PV)
        prefix_grp = QGroupBox("PV Prefix")
        pf = QHBoxLayout(prefix_grp)
        self._pv_widgets["pv_prefix"] = _le(_PV_DEFAULTS["pv_prefix"])
        self._pv_widgets["pv_prefix"].setPlaceholderText("e.g. ID15A2:")
        pf.addWidget(QLabel("Prefix:"))
        pf.addWidget(self._pv_widgets["pv_prefix"], 1)
        apply_btn = QPushButton("Auto-fill all PVs")
        apply_btn.setToolTip("Replaces all PV name prefixes with the value above")
        apply_btn.clicked.connect(self._apply_prefix)
        pf.addWidget(apply_btn)
        reconnect_btn = QPushButton("Reconnect")
        reconnect_btn.setToolTip("Re-subscribe CA monitors after changing PV names")
        reconnect_btn.clicked.connect(self._subscribe_pvs)
        pf.addWidget(reconnect_btn)
        vbox.addWidget(prefix_grp)

        # Motors — live value shows .RBV
        motor_grp = QGroupBox("EPICS Motors  ·  RBV →")
        mf = QFormLayout(motor_grp)
        for key, label, tip in [
            ("brg2",        "BRG2",   "Bragg 2 motor record base PV"),
            ("roll1_motor", "Roll1",  "Roll1 motor record base PV (used to read current position for energy table)"),
            ("roll2_motor", "Roll2",  "Roll2 motor record base PV"),
            ("x2_motor",    "X2",     "X2 motor record base PV"),
        ]:
            mf.addRow(label + ":", _pv_row(key, _PV_DEFAULTS[key], tip))
        vbox.addWidget(motor_grp)

        # Process Variables — live value shows the PV itself
        pv_grp = QGroupBox("Process Variables  ·  current value →")
        pvf = QFormLayout(pv_grp)
        for key, label, tip in [
            ("detector",        "Detector",           "Scalar detector readback PV"),
            ("monitor_pv",      "Monitor (normalize)", "Monitor detector PV — when set, all scan signals are divided by this value before fitting and plotting (optional)"),
            ("pitch_pv",        "Pitch piezo SP",     "Pitch piezo setpoint PV (PVAxis)"),
            ("slit_v_pv",       "Vertical slit SP",   "Vertical slit setpoint PV"),
            ("slit_h_pv",       "Horizontal slit SP", "Horizontal slit setpoint PV"),
            ("mono_e_pv",       "Mono energy",        "Monochromator energy setpoint PV"),
            ("harmonic_pv",     "Undulator harmonic", "Undulator harmonic PV"),
            ("und_e_pv",        "Undulator energy",   "Undulator energy setpoint PV"),
            ("und_start_pv",    "Undulator start",    "Undulator start/trigger PV"),
            ("roll2_energy_pv", "Roll2 energy set",   "Roll2 nominal energy setpoint PV"),
            ("x2_energy_pv",    "X2 energy set",      "X2 nominal energy setpoint PV"),
        ]:
            pvf.addRow(label + ":", _pv_row(key, _PV_DEFAULTS[key], tip))
        vbox.addWidget(pv_grp)

        # BPM position readbacks
        bpm_grp = QGroupBox("BPM Position Readbacks  ·  current value →")
        bpmf = QFormLayout(bpm_grp)
        for key, label, tip in [
            ("bpm_x_pv", "BPM X (BPMX)",
             "EPICS PV for beam position along X — used by the optional BPM alignment phase.\n"
             "BPMX > 0 → X2 moves negative; BPMX < 0 → X2 moves positive."),
            ("bpm_y_pv", "BPM Y (BPMY)",
             "EPICS PV for beam position along Y — used by the optional BPM alignment phase.\n"
             "BPMY > 0 → Roll2 moves positive; BPMY < 0 → Roll2 moves negative."),
        ]:
            bpmf.addRow(label + ":", _pv_row(key, _PV_DEFAULTS[key], tip))
        vbox.addWidget(bpm_grp)

        # Slit V center motors (for BPM-phase slit scan)
        slitv_grp = QGroupBox("Slit V Center Motors  ·  current value →")
        slitv_grp.setToolTip(
            "Motor PVs for the vertical slit center scan — the final step of the BPM alignment phase.\n"
            "The center PV is the scan axis; top and bottom blade RBVs are recorded in the CSV."
        )
        slitvf = QFormLayout(slitv_grp)
        for key, label, tip in [
            ("slit_v_center_pv", "Slit V center",
             "Virtual center motor PV — this is the axis that smart_scan moves during the slit scan."),
            ("slit_v_top_pv",    "Slit V top blade",
             "Upper blade motor PV — RBV is recorded in the CSV after the slit scan."),
            ("slit_v_bot_pv",    "Slit V bottom blade",
             "Lower blade motor PV — RBV is recorded in the CSV after the slit scan."),
        ]:
            slitvf.addRow(label + ":", _pv_row(key, _PV_DEFAULTS[key], tip))
        vbox.addWidget(slitv_grp)

        # ── Pre / Post energy change PVs ────────────────────────────────────
        energy_pvs_grp = QGroupBox("Energy Change PVs")
        energy_pvs_grp.setToolTip(
            "Optional PVs to write before and after the monochromator energy change.\n"
            "Rows marked 'Pre' are written before the mono moves; "
            "'Post' rows are written after the settle time."
        )
        epv = QVBoxLayout(energy_pvs_grp)

        info_epv = QLabel(
            "PVs written around the monochromator move — e.g. open/close shutters, "
            "set attenuators, or trigger beamline interlocks.\n"
            "<b>PV Name</b>: EPICS PV to write.  "
            "<b>Value</b>: value to write.  "
            "<b>When</b>: Pre = before mono moves; Post = after settle.  "
            "<b>Wait PV</b> (optional): poll this PV after writing.  "
            "<b>Wait Value</b>: target value to wait for (timeout 30 s)."
        )
        info_epv.setWordWrap(True)
        epv.addWidget(info_epv)

        self._energy_pv_table = QTableWidget(0, 5)
        self._energy_pv_table.setHorizontalHeaderLabels(["PV Name", "Value", "When", "Wait PV", "Wait Value"])
        self._energy_pv_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._energy_pv_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._energy_pv_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._energy_pv_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._energy_pv_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._energy_pv_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._energy_pv_table.setMaximumHeight(160)
        epv.addWidget(self._energy_pv_table)

        epv_btns = QHBoxLayout()
        add_pre_btn = QPushButton("Add Pre")
        add_pre_btn.setToolTip("Add a PV to write before the mono energy change")
        add_pre_btn.clicked.connect(lambda: self._add_energy_pv_row("Pre"))
        add_post_btn = QPushButton("Add Post")
        add_post_btn.setToolTip("Add a PV to write after the mono energy change (and settle)")
        add_post_btn.clicked.connect(lambda: self._add_energy_pv_row("Post"))
        rem_epv_btn = QPushButton("Remove")
        rem_epv_btn.clicked.connect(self._remove_energy_pv_row)
        epv_btns.addWidget(add_pre_btn)
        epv_btns.addWidget(add_post_btn)
        epv_btns.addWidget(rem_epv_btn)
        epv_btns.addStretch()
        epv.addLayout(epv_btns)

        vbox.addWidget(energy_pvs_grp)

        vbox.addStretch()
        scroll.setWidget(container)
        return scroll

    def _browse_csv(self):
        from PyQt6.QtWidgets import QFileDialog
        import os
        current = self._scan_widgets["filename"].text().strip()
        start_dir = os.path.dirname(os.path.abspath(current)) if current else ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Choose output CSV file", start_dir or "",
            "CSV files (*.csv);;All files (*)"
        )
        if path:
            self._scan_widgets["filename"].setText(path)

    def _apply_prefix(self):
        prefix = self._pv_widgets["pv_prefix"].text().strip()
        if not prefix:
            return
        for key, w in self._pv_widgets.items():
            if key == "pv_prefix":
                continue
            current = w.text().strip()
            if ":" in current:
                # replace up to (and including) the first ':' segment
                parts = current.split(":", 1)
                if len(parts) == 2:
                    new_val = prefix + parts[1]
                else:
                    new_val = prefix + current
            else:
                new_val = prefix + current
            w.setText(new_val)
        self._subscribe_pvs()

    # ── Scan Parameters page ────────────────────────────────────────────────

    def _build_scan_page(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setSpacing(8)

        _stay = QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint

        # Slits
        slit_grp = QGroupBox("Slit Positions (mm)")
        sf = QFormLayout(slit_grp)
        sf.setFieldGrowthPolicy(_stay)
        for key, label in [
            ("slit_open_v",  "Open vertical"),
            ("slit_open_h",  "Open horizontal"),
            ("slit_close_v", "Close vertical"),
            ("slit_close_h", "Close horizontal"),
        ]:
            w = _dbl(_SCAN_DEFAULTS[key], lo=0.0, hi=100.0, decimals=3, step=0.01)
            self._scan_widgets[key] = w
            sf.addRow(label + ":", w)
        vbox.addWidget(slit_grp)

        # BRG2
        brg2_grp = QGroupBox("BRG2 Scan")
        bf = QFormLayout(brg2_grp)
        bf.setFieldGrowthPolicy(_stay)
        for key, label, lo, hi, dec, step in [
            ("brg2_start",         "Start",           -1.0, 1.0, 5, 0.001),
            ("brg2_stop",          "Stop",            -1.0, 1.0, 5, 0.001),
            ("brg2_nsteps",        "N steps",          3, 201, 0, 1),
            ("brg2_min_prominence","Min prominence",  0.0, 100.0, 2, 0.5),
        ]:
            if key == "brg2_nsteps":
                w = _int(_SCAN_DEFAULTS[key], lo=3, hi=201)
            else:
                w = _dbl(_SCAN_DEFAULTS[key], lo=lo, hi=hi, decimals=dec, step=step)
            self._scan_widgets[key] = w
            bf.addRow(label + ":", w)
        mon_brg2 = QCheckBox("Normalize with monitor PV")
        mon_brg2.setChecked(_SCAN_DEFAULTS["monitor_brg2"])
        mon_brg2.setToolTip("Divide detector by Monitor PV at each point (set Monitor PV in Motors & PVs tab)")
        self._scan_widgets["monitor_brg2"] = mon_brg2
        bf.addRow("", mon_brg2)
        vbox.addWidget(brg2_grp)

        # Pitch
        pitch_grp = QGroupBox("Pitch Piezo Scan")
        pf = QFormLayout(pitch_grp)
        pf.setFieldGrowthPolicy(_stay)
        do_pitch = QCheckBox("Enable pitch scan")
        do_pitch.setChecked(_SCAN_DEFAULTS["do_pitch_scan"])
        self._scan_widgets["do_pitch_scan"] = do_pitch
        pf.addRow("", do_pitch)
        for key, label, lo, hi, dec, step in [
            ("pitch_home",   "Home position",  -50.0, 50.0, 3, 0.1),
            ("pitch_start",  "Start",         -50.0, 50.0, 3, 0.1),
            ("pitch_stop",   "Stop",          -50.0, 50.0, 3, 0.1),
            ("pitch_nsteps", "N steps",         3, 201, 0, 1),
            ("pitch_settle", "Settle time (s)", 0.0, 10.0, 3, 0.01),
        ]:
            if key == "pitch_nsteps":
                w = _int(_SCAN_DEFAULTS[key], lo=3, hi=201)
            else:
                w = _dbl(_SCAN_DEFAULTS[key], lo=lo, hi=hi, decimals=dec, step=step)
            self._scan_widgets[key] = w
            pf.addRow(label + ":", w)
        mon_pitch = QCheckBox("Normalize with monitor PV")
        mon_pitch.setChecked(_SCAN_DEFAULTS["monitor_pitch"])
        mon_pitch.setToolTip("Divide detector by Monitor PV at each point (set Monitor PV in Motors & PVs tab)")
        self._scan_widgets["monitor_pitch"] = mon_pitch
        pf.addRow("", mon_pitch)
        vbox.addWidget(pitch_grp)

        # Roll2
        roll2_grp = QGroupBox("Roll2 Scan")
        rf = QFormLayout(roll2_grp)
        rf.setFieldGrowthPolicy(_stay)
        for key, label, lo, hi, dec, step in [
            ("roll2_start",  "Start",  -1.0, 1.0, 5, 0.001),
            ("roll2_stop",   "Stop",   -1.0, 1.0, 5, 0.001),
            ("roll2_nsteps", "N steps",   3, 201, 0, 1),
        ]:
            if key == "roll2_nsteps":
                w = _int(_SCAN_DEFAULTS[key], lo=3, hi=201)
            else:
                w = _dbl(_SCAN_DEFAULTS[key], lo=lo, hi=hi, decimals=dec, step=step)
            self._scan_widgets[key] = w
            rf.addRow(label + ":", w)
        mon_roll2 = QCheckBox("Normalize with monitor PV")
        mon_roll2.setChecked(_SCAN_DEFAULTS["monitor_roll2"])
        mon_roll2.setToolTip("Divide detector by Monitor PV at each point (set Monitor PV in Motors & PVs tab)")
        self._scan_widgets["monitor_roll2"] = mon_roll2
        rf.addRow("", mon_roll2)
        vbox.addWidget(roll2_grp)

        # X2
        x2_grp = QGroupBox("X2 Scan")
        xf = QFormLayout(x2_grp)
        xf.setFieldGrowthPolicy(_stay)
        for key, label, lo, hi, dec, step in [
            ("x2_start",  "Start",  -10.0, 10.0, 3, 0.1),
            ("x2_stop",   "Stop",   -10.0, 10.0, 3, 0.1),
            ("x2_nsteps", "N steps",    3, 201, 0, 1),
        ]:
            if key == "x2_nsteps":
                w = _int(_SCAN_DEFAULTS[key], lo=3, hi=201)
            else:
                w = _dbl(_SCAN_DEFAULTS[key], lo=lo, hi=hi, decimals=dec, step=step)
            self._scan_widgets[key] = w
            xf.addRow(label + ":", w)
        mon_x2 = QCheckBox("Normalize with monitor PV")
        mon_x2.setChecked(_SCAN_DEFAULTS["monitor_x2"])
        mon_x2.setToolTip("Divide detector by Monitor PV at each point (set Monitor PV in Motors & PVs tab)")
        self._scan_widgets["monitor_x2"] = mon_x2
        xf.addRow("", mon_x2)
        vbox.addWidget(x2_grp)

        # Fine scan
        fine_grp = QGroupBox("Fine Scan")
        ff = QFormLayout(fine_grp)
        ff.setFieldGrowthPolicy(_stay)
        fine_en = QCheckBox("Enable fine scan")
        fine_en.setChecked(_SCAN_DEFAULTS["fine_scan"])
        self._scan_widgets["fine_scan"] = fine_en
        ff.addRow("", fine_en)
        for key, label, lo, hi, dec, step in [
            ("fine_sigma_range", "Sigma range",   0.5, 20.0, 1, 0.5),
            ("fine_nsteps",      "N steps",         3, 201, 0, 1),
            ("fine_scan_iter",   "Max iterations",  1,  10, 0, 1),
            ("dmov_delay",       "DMOV delay (s)", 0.0,  5.0, 2, 0.05),
        ]:
            if key in ("fine_nsteps", "fine_scan_iter"):
                w = _int(_SCAN_DEFAULTS[key], lo=1, hi=201)
            else:
                w = _dbl(_SCAN_DEFAULTS[key], lo=lo, hi=hi, decimals=dec, step=step)
            self._scan_widgets[key] = w
            ff.addRow(label + ":", w)
        self._scan_widgets["dmov_delay"].setToolTip(
            "Wait this many seconds after issuing a motor move before polling DMOV.\n"
            "Increase if the motor record takes time to clear DMOV (typical: 0.2–0.5 s)."
        )
        vbox.addWidget(fine_grp)

        # BPM Alignment parameters
        bpm_scan_grp = QGroupBox("BPM Alignment Parameters")
        bpm_scan_grp.setToolTip(
            "Parameters for the optional BPM position alignment phase.\n"
            "Enable it per-run via the 'Enable BPM alignment' checkbox on the Alignment tab.\n"
            "BPM X and BPM Y readback PVs are set in the Motors & PVs tab."
        )
        bpmsf = QFormLayout(bpm_scan_grp)
        bpmsf.setFieldGrowthPolicy(_stay)

        w = _dbl(_SCAN_DEFAULTS["bpm_slit_open"], lo=0.001, hi=100.0, decimals=3, step=1.0)
        w.setToolTip("Slit opening (mm) for BPM phase — both V and H")
        self._scan_widgets["bpm_slit_open"] = w
        bpmsf.addRow("Slit open (mm):", w)

        w = _dbl(_SCAN_DEFAULTS["bpm_x_search_step"], lo=0.001, hi=10000.0, decimals=3, step=1.0)
        w.setToolTip("X2 step size (μm) for the BPMX zero-crossing walk\n"
                     "BPMX > 0 → X2 moves negative; BPMX < 0 → X2 moves positive")
        self._scan_widgets["bpm_x_search_step"] = w
        bpmsf.addRow("X2 step (μm):", w)

        w = _dbl(_SCAN_DEFAULTS["bpm_y_search_step"], lo=1e-6, hi=1.0, decimals=5, step=0.0001)
        w.setToolTip("Roll2 step size (mdeg) for the BPMY zero-crossing walk\n"
                     "BPMY > 0 → Roll2 moves positive; BPMY < 0 → Roll2 moves negative")
        self._scan_widgets["bpm_y_search_step"] = w
        bpmsf.addRow("Roll2 step (mdeg):", w)

        w = _int(_SCAN_DEFAULTS["bpm_max_steps"], lo=2, hi=200)
        w.setToolTip("Maximum steps to walk before giving up on the zero-crossing search")
        self._scan_widgets["bpm_max_steps"] = w
        bpmsf.addRow("Max steps:", w)

        w = _dbl(_SCAN_DEFAULTS["bpm_x_tolerance"], lo=0.1, hi=10000.0, decimals=1, step=1.0)
        w.setToolTip("Target |BPMX| (μm): refinement passes stop when the X beam position is within this distance of zero")
        self._scan_widgets["bpm_x_tolerance"] = w
        bpmsf.addRow("BPMX tolerance (μm):", w)

        w = _dbl(_SCAN_DEFAULTS["bpm_y_tolerance"], lo=0.1, hi=10000.0, decimals=1, step=1.0)
        w.setToolTip("Target |BPMY| (μm): refinement passes stop when the Y beam position is within this distance of zero")
        self._scan_widgets["bpm_y_tolerance"] = w
        bpmsf.addRow("BPMY tolerance (μm):", w)

        w = _int(_SCAN_DEFAULTS["bpm_refine_iter"], lo=0, hi=10)
        w.setToolTip("Maximum number of extra refinement passes after the initial zero-crossing walk\n"
                     "Set to 0 to disable refinement (single pass only)")
        self._scan_widgets["bpm_refine_iter"] = w
        bpmsf.addRow("Max refine passes:", w)

        bpmsf.addRow(QLabel(""))
        _slitv_hdr = QLabel("<b>Slit V center scan</b>")
        bpmsf.addRow(_slitv_hdr)

        w = _dbl(_SCAN_DEFAULTS["bpm_slit_v_gap"], lo=0.001, hi=100.0, decimals=3, step=0.01)
        w.setToolTip("Fixed vertical slit gap (mm) held during the slit center scan")
        self._scan_widgets["bpm_slit_v_gap"] = w
        bpmsf.addRow("Slit V gap (mm):", w)

        w = _dbl(_SCAN_DEFAULTS["bpm_slit_v_start"], lo=-100.0, hi=0.0, decimals=3, step=0.1)
        w.setToolTip("Scan start offset (mm) relative to current slit center position")
        self._scan_widgets["bpm_slit_v_start"] = w
        bpmsf.addRow("Slit V start (mm):", w)

        w = _dbl(_SCAN_DEFAULTS["bpm_slit_v_stop"], lo=0.0, hi=100.0, decimals=3, step=0.1)
        w.setToolTip("Scan stop offset (mm) relative to current slit center position")
        self._scan_widgets["bpm_slit_v_stop"] = w
        bpmsf.addRow("Slit V stop (mm):", w)

        w = _int(_SCAN_DEFAULTS["bpm_slit_v_nsteps"], lo=3, hi=500)
        w.setToolTip("Number of scan points for the slit V center scan")
        self._scan_widgets["bpm_slit_v_nsteps"] = w
        bpmsf.addRow("Slit V steps:", w)

        vbox.addWidget(bpm_scan_grp)

        # Other
        other_grp = QGroupBox("Other Parameters")
        of = QFormLayout(other_grp)
        of.setFieldGrowthPolicy(_stay)

        backlash_w = QCheckBox("Enable backlash correction")
        backlash_w.setToolTip(
            "After each scan, overshoot by one FWHM then return to the peak "
            "to eliminate motor backlash before recording the position."
        )
        backlash_w.setChecked(_SCAN_DEFAULTS["backlash_correction"])
        self._scan_widgets["backlash_correction"] = backlash_w
        of.addRow("Backlash:", backlash_w)

        settle_w = _dbl(_SCAN_DEFAULTS["settle"], lo=0.0, hi=60.0, decimals=2, step=0.05)
        self._scan_widgets["settle"] = settle_w
        of.addRow("Motor settle (s):", settle_w)

        e_settle_w = _dbl(_SCAN_DEFAULTS["energy_settle"], lo=0.0, hi=60.0, decimals=1, step=0.5)
        self._scan_widgets["energy_settle"] = e_settle_w
        of.addRow("Energy settle (s):", e_settle_w)

        peak_cb = _NoScrollComboBox()
        peak_cb.addItems(["stats", "fit"])
        peak_cb.setCurrentText(_SCAN_DEFAULTS["peak_method"])
        self._scan_widgets["peak_method"] = peak_cb
        of.addRow("Peak method:", peak_cb)

        centre_cb = _NoScrollComboBox()
        centre_cb.addItems(["centroid", "peak", "weighted_median"])
        centre_cb.setCurrentText(_SCAN_DEFAULTS["stats_centre"])
        self._scan_widgets["stats_centre"] = centre_cb
        of.addRow("Stats centre:", centre_cb)

        profile_cb = _NoScrollComboBox()
        profile_cb.addItems(["auto", "gaussian", "lorentzian", "super_gaussian"])
        profile_cb.setCurrentText(_SCAN_DEFAULTS["fit_profile"])
        self._scan_widgets["fit_profile"] = profile_cb
        of.addRow("Fit profile:", profile_cb)

        filename_w = _le(_SCAN_DEFAULTS["filename"])
        self._scan_widgets["filename"] = filename_w
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_csv)
        autosave_cb = QCheckBox("Autosave")
        autosave_cb.setChecked(True)
        autosave_cb.setToolTip(
            "Append results to the CSV after every alignment.\n"
            "Uncheck to run without saving. Always disabled in Simulation mode."
        )
        self._scan_widgets["autosave"] = autosave_cb
        fn_row = QWidget()
        fn_h = QHBoxLayout(fn_row)
        fn_h.setContentsMargins(0, 0, 0, 0)
        fn_h.setSpacing(4)
        fn_h.addWidget(filename_w, 1)
        fn_h.addWidget(browse_btn)
        fn_h.addWidget(autosave_cb)
        of.addRow("Output CSV:", fn_row)

        r_settle_w = _dbl(_SCAN_DEFAULTS["record_settle"], lo=0.0, hi=60.0, decimals=1, step=0.5)
        self._scan_widgets["record_settle"] = r_settle_w
        of.addRow("Record settle (s):", r_settle_w)

        vbox.addWidget(other_grp)

        # Post-Alignment Recording
        rec_grp = QGroupBox("Post-Alignment Recording")
        rec_grp.setToolTip(
            "PVs read after each energy row's alignment and appended as extra CSV columns."
        )
        rv = QVBoxLayout(rec_grp)

        info = QLabel(
            "Extra PVs to read after each energy alignment and save to the CSV output.\n"
            "Each row adds one column: <b>Label</b> becomes the CSV column header, "
            "<b>PV Name</b> is the EPICS PV to read."
        )
        info.setWordWrap(True)
        rv.addWidget(info)

        self._record_table = QTableWidget(0, 2)
        self._record_table.setHorizontalHeaderLabels(["Label (CSV column)", "PV Name"])
        self._record_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._record_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._record_table.setMaximumHeight(160)
        rv.addWidget(self._record_table)

        rec_btns = QHBoxLayout()
        add_btn = QPushButton("Add PV")
        add_btn.clicked.connect(self._add_record_row)
        rem_btn = QPushButton("Remove")
        rem_btn.clicked.connect(self._remove_record_row)
        rec_btns.addWidget(add_btn)
        rec_btns.addWidget(rem_btn)
        rec_btns.addStretch()
        rv.addLayout(rec_btns)

        vbox.addWidget(rec_grp)
        vbox.addStretch()
        scroll.setWidget(container)
        return scroll

    def _add_record_row(self, label="", pv=""):
        r = self._record_table.rowCount()
        self._record_table.insertRow(r)
        self._record_table.setItem(r, 0, QTableWidgetItem(label))
        self._record_table.setItem(r, 1, QTableWidgetItem(pv))

    def _remove_record_row(self):
        rows = sorted(
            {idx.row() for idx in self._record_table.selectedIndexes()}, reverse=True
        )
        if not rows:
            rows = [self._record_table.rowCount() - 1]
        for r in rows:
            if r >= 0:
                self._record_table.removeRow(r)

    def _add_energy_pv_row(self, when="Pre", pv="", value="", wait_pv="", wait_value=""):
        r = self._energy_pv_table.rowCount()
        self._energy_pv_table.insertRow(r)
        self._energy_pv_table.setItem(r, 0, QTableWidgetItem(pv))
        self._energy_pv_table.setItem(r, 1, QTableWidgetItem(str(value)))
        when_cb = _NoScrollComboBox()
        when_cb.addItems(["Pre", "Post"])
        when_cb.setCurrentText(when)
        self._energy_pv_table.setCellWidget(r, 2, when_cb)
        self._energy_pv_table.setItem(r, 3, QTableWidgetItem(str(wait_pv)))
        self._energy_pv_table.setItem(r, 4, QTableWidgetItem(str(wait_value)))

    def _remove_energy_pv_row(self):
        rows = sorted(
            {idx.row() for idx in self._energy_pv_table.selectedIndexes()}, reverse=True
        )
        if not rows:
            rows = [self._energy_pv_table.rowCount() - 1]
        for r in rows:
            if r >= 0:
                self._energy_pv_table.removeRow(r)

    def _subscribe_pvs(self):
        """Create CA monitors for all configured motor / PV names."""
        self._unsubscribe_pvs()
        from .smart_scan_functions import create_pv_monitor
        bridge = self._bridge
        pv_map = {}
        for key in ("brg2", "roll1_motor", "roll2_motor", "x2_motor",
                    "slit_v_center_pv", "slit_v_top_pv", "slit_v_bot_pv"):
            pv = self._pv_widgets[key].text().strip()
            if pv:
                pv_map[key] = pv + ".RBV"
        for key in ("detector", "monitor_pv", "pitch_pv", "slit_v_pv", "slit_h_pv",
                    "mono_e_pv", "harmonic_pv", "und_e_pv", "und_start_pv",
                    "roll2_energy_pv", "x2_energy_pv",
                    "bpm_x_pv", "bpm_y_pv"):
            pv = self._pv_widgets[key].text().strip()
            if pv:
                pv_map[key] = pv
        for key, pv_name in pv_map.items():
            lbl = self._rbk_labels.get(key)
            if lbl is None:
                continue
            def _make_cb(k, br):
                def _cb(val):
                    try:
                        br.value_changed.emit(k, val)
                    except RuntimeError:
                        pass
                return _cb
            handle = create_pv_monitor(pv_name, _make_cb(key, bridge))
            if handle is None:
                lbl.setText("—")
                lbl.setStyleSheet("color: #888; font-family: monospace;")
            else:
                lbl.setText("…")
                lbl.setStyleSheet("color: #888; font-family: monospace;")
                self._rbk_pvs[key] = handle

    def _unsubscribe_pvs(self):
        for pv in self._rbk_pvs.values():
            try:
                pv.clear_callbacks()
            except Exception:
                pass
        self._rbk_pvs.clear()

    def _on_pv_value(self, key: str, value):
        lbl = self._rbk_labels.get(key)
        if lbl is None:
            return
        if value is None:
            lbl.setText("—")
            lbl.setStyleSheet("font-family: monospace; color: #888;")
        else:
            try:
                lbl.setText(f"{float(value):.6g}")
                lbl.setStyleSheet("font-family: monospace;")
            except (TypeError, ValueError):
                lbl.setText(str(value))
                lbl.setStyleSheet("font-family: monospace;")

    def set_output_filename(self, path: str):
        self._scan_widgets["filename"].setText(path)

    def get_output_filename(self) -> str:
        return self._scan_widgets["filename"].text().strip()

    def get_roll1_rbv(self) -> str:
        """Return the current Roll1 RBV as a string, or '' if not configured/connected."""
        lbl = self._rbk_labels.get("roll1_motor")
        if lbl is None:
            return ""
        text = lbl.text().strip()
        return "" if text in ("—", "…") else text

    # ── Public API ───────────────────────────────────────────────────────────

    def get_kwargs(self):
        """Return dict suitable for passing directly to align_beamline()."""
        kwargs = {}
        for key, w in self._pv_widgets.items():
            if key == "pv_prefix":
                continue
            kwargs[key] = w.text().strip()

        for key, w in self._scan_widgets.items():
            if isinstance(w, QCheckBox):
                kwargs[key] = w.isChecked()
            elif isinstance(w, QComboBox):
                kwargs[key] = w.currentText()
            elif isinstance(w, QSpinBox):
                kwargs[key] = w.value()
            elif isinstance(w, QLineEdit):
                text = w.text().strip()
                try:
                    kwargs[key] = float(text)
                except ValueError:
                    kwargs[key] = text

        # Build record_pvs dict from the recording table
        record_pvs = {}
        for r in range(self._record_table.rowCount()):
            lbl_item = self._record_table.item(r, 0)
            pv_item  = self._record_table.item(r, 1)
            lbl = lbl_item.text().strip() if lbl_item else ""
            pv  = pv_item.text().strip()  if pv_item  else ""
            if lbl and pv:
                record_pvs[lbl] = pv
        kwargs["record_pvs"] = record_pvs if record_pvs else None

        # Build pre/post energy PV lists from the energy change table
        pre_energy_pvs, post_energy_pvs = [], []
        for r in range(self._energy_pv_table.rowCount()):
            pv_item  = self._energy_pv_table.item(r, 0)
            val_item = self._energy_pv_table.item(r, 1)
            cb       = self._energy_pv_table.cellWidget(r, 2)
            wpv_item = self._energy_pv_table.item(r, 3)
            wval_item= self._energy_pv_table.item(r, 4)
            pv       = pv_item.text().strip()   if pv_item   else ""
            raw_v    = val_item.text().strip()  if val_item  else ""
            when     = cb.currentText() if cb else "Pre"
            wait_pv  = wpv_item.text().strip()  if wpv_item  else ""
            raw_wv   = wval_item.text().strip() if wval_item else ""
            if not pv:
                continue
            try:
                value = float(raw_v)
            except ValueError:
                value = raw_v
            entry = (pv, value)
            if wait_pv:
                try:
                    wait_value = float(raw_wv)
                except ValueError:
                    wait_value = raw_wv
                entry = (pv, value, wait_pv, wait_value)
            if when == "Pre":
                pre_energy_pvs.append(entry)
            else:
                post_energy_pvs.append(entry)
        kwargs["pre_energy_pvs"]  = pre_energy_pvs  or None
        kwargs["post_energy_pvs"] = post_energy_pvs or None

        return kwargs

    def reload_settings(self):
        self._load_settings()

    def save_settings(self):
        for key, w in self._pv_widgets.items():
            self._settings.setValue(f"pv/{key}", w.text())
        for key, w in self._scan_widgets.items():
            if isinstance(w, QCheckBox):
                self._settings.setValue(f"scan/{key}", w.isChecked())
            elif isinstance(w, QComboBox):
                self._settings.setValue(f"scan/{key}", w.currentText())
            elif isinstance(w, QSpinBox):
                self._settings.setValue(f"scan/{key}", w.value())
            elif isinstance(w, QLineEdit):
                self._settings.setValue(f"scan/{key}", w.text())

        # Save record_pvs table as a list of (label, pv) tuples
        rows = []
        for r in range(self._record_table.rowCount()):
            lbl = (self._record_table.item(r, 0) or QTableWidgetItem()).text().strip()
            pv  = (self._record_table.item(r, 1) or QTableWidgetItem()).text().strip()
            if lbl or pv:
                rows.append((lbl, pv))
        self._settings.setValue("record_pvs", repr(rows))

        # Save energy change PV table as list of (pv, value, when, wait_pv, wait_value) tuples
        epv_rows = []
        for r in range(self._energy_pv_table.rowCount()):
            pv       = (self._energy_pv_table.item(r, 0) or QTableWidgetItem()).text().strip()
            value    = (self._energy_pv_table.item(r, 1) or QTableWidgetItem()).text().strip()
            cb       = self._energy_pv_table.cellWidget(r, 2)
            when     = cb.currentText() if cb else "Pre"
            wait_pv  = (self._energy_pv_table.item(r, 3) or QTableWidgetItem()).text().strip()
            wait_val = (self._energy_pv_table.item(r, 4) or QTableWidgetItem()).text().strip()
            if pv:
                epv_rows.append((pv, value, when, wait_pv, wait_val))
        self._settings.setValue("energy_pvs", repr(epv_rows))

    def _load_settings(self):
        for key, w in self._pv_widgets.items():
            v = self._settings.value(f"pv/{key}")
            if v is not None:
                w.setText(str(v))
        for key, w in self._scan_widgets.items():
            v = self._settings.value(f"scan/{key}")
            if v is None:
                continue
            if isinstance(w, QCheckBox):
                w.setChecked(str(v).lower() in ("true", "1"))
            elif isinstance(w, QComboBox):
                idx = w.findText(str(v))
                if idx >= 0:
                    w.setCurrentIndex(idx)
            elif isinstance(w, QSpinBox):
                try:
                    w.setValue(int(float(v)))
                except (ValueError, TypeError):
                    pass
            elif isinstance(w, QLineEdit):
                w.setText(str(v))

        # Restore record_pvs table
        raw = self._settings.value("record_pvs")
        if raw:
            try:
                rows = eval(raw)  # noqa: S307  (trusted local QSettings)
                self._record_table.setRowCount(0)
                for lbl, pv in rows:
                    self._add_record_row(lbl, pv)
            except Exception:
                pass

        # Restore energy change PV table
        raw_epv = self._settings.value("energy_pvs")
        if raw_epv:
            try:
                epv_rows = eval(raw_epv)  # noqa: S307
                self._energy_pv_table.setRowCount(0)
                for row in epv_rows:
                    # support old 3-tuple (pv, value, when) and new 5-tuple
                    if len(row) == 5:
                        pv, value, when, wait_pv, wait_val = row
                    else:
                        pv, value, when = row[:3]
                        wait_pv, wait_val = "", ""
                    self._add_energy_pv_row(when, pv, value, wait_pv, wait_val)
            except Exception:
                pass
