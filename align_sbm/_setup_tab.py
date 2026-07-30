"""Setup tab — Motors/PVs and scan parameters."""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QScrollArea, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLineEdit, QDoubleSpinBox, QSpinBox, QCheckBox,
    QComboBox, QPushButton, QLabel, QTabWidget,
)

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
    "do_pitch_scan":   True,
    "peak_method":     "stats",
    "stats_centre":    "centroid",
    "fit_profile":     "auto",
    "filename":        "alignment_results.csv",
}


def _dbl(val, lo=-1e6, hi=1e6, decimals=6, step=0.001):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(val)
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
        self._build_ui()
        self._load_settings()

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

        # Prefix apply row
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
        vbox.addWidget(prefix_grp)

        # Motors
        motor_grp = QGroupBox("EPICS Motors (motor record base PV)")
        mf = QFormLayout(motor_grp)
        for key, label, tip in [
            ("brg2",        "BRG2 motor",   "Bragg 2 motor record base PV"),
            ("roll2_motor", "Roll2 motor",  "Roll2 motor record base PV"),
            ("x2_motor",    "X2 motor",     "X2 motor record base PV"),
        ]:
            w = _le(_PV_DEFAULTS[key])
            w.setToolTip(tip)
            self._pv_widgets[key] = w
            mf.addRow(label + ":", w)
        vbox.addWidget(motor_grp)

        # PVs
        pv_grp = QGroupBox("Process Variables")
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
            w = _le(_PV_DEFAULTS[key])
            w.setToolTip(tip)
            self._pv_widgets[key] = w
            pvf.addRow(label + ":", w)
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

        vbox.addWidget(other_grp)
        vbox.addStretch()
        scroll.setWidget(container)
        return scroll

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
            elif isinstance(w, (QDoubleSpinBox, QSpinBox)):
                kwargs[key] = w.value()
            elif isinstance(w, QLineEdit):
                kwargs[key] = w.text().strip()

        return kwargs

    def save_settings(self):
        for key, w in self._pv_widgets.items():
            self._settings.setValue(f"pv/{key}", w.text())
        for key, w in self._scan_widgets.items():
            if isinstance(w, QCheckBox):
                self._settings.setValue(f"scan/{key}", w.isChecked())
            elif isinstance(w, QComboBox):
                self._settings.setValue(f"scan/{key}", w.currentText())
            elif isinstance(w, QDoubleSpinBox):
                self._settings.setValue(f"scan/{key}", w.value())
            elif isinstance(w, QSpinBox):
                self._settings.setValue(f"scan/{key}", w.value())
            elif isinstance(w, QLineEdit):
                self._settings.setValue(f"scan/{key}", w.text())

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
            elif isinstance(w, QDoubleSpinBox):
                try:
                    w.setValue(float(v))
                except (ValueError, TypeError):
                    pass
            elif isinstance(w, QSpinBox):
                try:
                    w.setValue(int(float(v)))
                except (ValueError, TypeError):
                    pass
            elif isinstance(w, QLineEdit):
                w.setText(str(v))
