"""Setup tab — Motors/PVs and scan parameters."""
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QWidget, QScrollArea, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLineEdit, QSpinBox, QCheckBox,
    QComboBox, QPushButton, QLabel, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView,
)


class _PVBridge(QObject):
    """Thread-safe bridge: CA monitor callbacks (CA thread) → Qt signals (main thread)."""
    value_changed = pyqtSignal(str, object)   # (key, value)

# ── Default PV names (ID15A2 prefix) ────────────────────────────────────────
_PV_DEFAULTS = {
    "pv_prefix":      "ID15A2:",
    "detector":       "ID15A2:det:Signal",
    "brg2":           "ID15A2:BRG2",
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
}

_SCAN_DEFAULTS = {
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
    # Other
    "settle":           0.3,
    "energy_settle":    2.0,
    "record_settle":    2.0,
    "do_pitch_scan":   True,
    "peak_method":     "stats",
    "stats_centre":    "centroid",
    "fit_profile":     "auto",
    "filename":        "alignment_results.csv",
}


def _dbl(val, lo=-1e6, hi=1e6, decimals=6, step=0.001):
    w = QLineEdit(str(val))
    w.setValidator(QDoubleValidator(lo, hi, decimals))
    return w


def _int(val, lo=1, hi=999):
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setValue(val)
    return w


def _le(text=""):
    w = QLineEdit(text)
    return w


class SetupTab(QWidget):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._pv_widgets = {}
        self._scan_widgets = {}
        self._rbk_labels: dict = {}
        self._rbk_pvs: dict = {}          # key → epics.PV handle
        self._bridge = _PVBridge()
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

        vbox.addStretch()
        scroll.setWidget(container)
        return scroll

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

        # Slits
        slit_grp = QGroupBox("Slit Positions (mm)")
        sf = QFormLayout(slit_grp)
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
        vbox.addWidget(brg2_grp)

        # Pitch
        pitch_grp = QGroupBox("Pitch Piezo Scan")
        pf = QFormLayout(pitch_grp)
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
        vbox.addWidget(pitch_grp)

        # Roll2
        roll2_grp = QGroupBox("Roll2 Scan")
        rf = QFormLayout(roll2_grp)
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
        vbox.addWidget(roll2_grp)

        # X2
        x2_grp = QGroupBox("X2 Scan")
        xf = QFormLayout(x2_grp)
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
        vbox.addWidget(x2_grp)

        # Fine scan
        fine_grp = QGroupBox("Fine Scan")
        ff = QFormLayout(fine_grp)
        fine_en = QCheckBox("Enable fine scan")
        fine_en.setChecked(_SCAN_DEFAULTS["fine_scan"])
        self._scan_widgets["fine_scan"] = fine_en
        ff.addRow("", fine_en)
        for key, label, lo, hi, dec, step in [
            ("fine_sigma_range", "Sigma range",   0.5, 20.0, 1, 0.5),
            ("fine_nsteps",      "N steps",         3, 201, 0, 1),
            ("fine_scan_iter",   "Max iterations",  1,  10, 0, 1),
        ]:
            if key in ("fine_nsteps", "fine_scan_iter"):
                w = _int(_SCAN_DEFAULTS[key], lo=1, hi=201)
            else:
                w = _dbl(_SCAN_DEFAULTS[key], lo=lo, hi=hi, decimals=dec, step=step)
            self._scan_widgets[key] = w
            ff.addRow(label + ":", w)
        vbox.addWidget(fine_grp)

        # Other
        other_grp = QGroupBox("Other Parameters")
        of = QFormLayout(other_grp)

        settle_w = _dbl(_SCAN_DEFAULTS["settle"], lo=0.0, hi=60.0, decimals=2, step=0.05)
        self._scan_widgets["settle"] = settle_w
        of.addRow("Motor settle (s):", settle_w)

        e_settle_w = _dbl(_SCAN_DEFAULTS["energy_settle"], lo=0.0, hi=60.0, decimals=1, step=0.5)
        self._scan_widgets["energy_settle"] = e_settle_w
        of.addRow("Energy settle (s):", e_settle_w)

        peak_cb = QComboBox()
        peak_cb.addItems(["stats", "fit"])
        peak_cb.setCurrentText(_SCAN_DEFAULTS["peak_method"])
        self._scan_widgets["peak_method"] = peak_cb
        of.addRow("Peak method:", peak_cb)

        centre_cb = QComboBox()
        centre_cb.addItems(["centroid", "peak", "weighted_median"])
        centre_cb.setCurrentText(_SCAN_DEFAULTS["stats_centre"])
        self._scan_widgets["stats_centre"] = centre_cb
        of.addRow("Stats centre:", centre_cb)

        profile_cb = QComboBox()
        profile_cb.addItems(["auto", "gaussian", "lorentzian", "super_gaussian"])
        profile_cb.setCurrentText(_SCAN_DEFAULTS["fit_profile"])
        self._scan_widgets["fit_profile"] = profile_cb
        of.addRow("Fit profile:", profile_cb)

        filename_w = _le(_SCAN_DEFAULTS["filename"])
        self._scan_widgets["filename"] = filename_w
        of.addRow("Output CSV:", filename_w)

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

    def _subscribe_pvs(self):
        """Create CA monitors for all configured motor / PV names."""
        self._unsubscribe_pvs()
        from .smart_scan_functions import create_pv_monitor
        bridge = self._bridge
        pv_map = {}
        for key in ("brg2", "roll2_motor", "x2_motor"):
            pv = self._pv_widgets[key].text().strip()
            if pv:
                pv_map[key] = pv + ".RBV"
        for key in ("detector", "pitch_pv", "slit_v_pv", "slit_h_pv",
                    "mono_e_pv", "harmonic_pv", "und_e_pv", "und_start_pv",
                    "roll2_energy_pv", "x2_energy_pv"):
            pv = self._pv_widgets[key].text().strip()
            if pv:
                pv_map[key] = pv
        for key, pv_name in pv_map.items():
            lbl = self._rbk_labels.get(key)
            if lbl is None:
                continue
            handle = create_pv_monitor(
                pv_name,
                lambda val, k=key: bridge.value_changed.emit(k, val),
            )
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

        return kwargs

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
