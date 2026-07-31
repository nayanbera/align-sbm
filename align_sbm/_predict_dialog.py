"""Predict Roll2 / X2 for new MonoE values by fitting the alignment CSV history."""
import csv
import os

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem, QFileDialog, QGroupBox,
    QRadioButton, QButtonGroup, QHeaderView,
)

from ._ml import NumpyGPR

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    _MPL = True
except ImportError:
    _MPL = False

try:
    from scipy.interpolate import UnivariateSpline as _USpline
    _SCIPY = True
except ImportError:
    _SCIPY = False


class PredictDialog(QDialog):
    """Fit Roll2/X2 vs MonoE from alignment history and predict for new energies."""

    _AMBER_BG  = QColor("#4a3800")
    _AMBER_FG  = QColor("#ffc107")
    _EXTRAP_BG = QColor("#5a3a00")
    _EXTRAP_FG = QColor("#ffb74d")

    # Column indices in the prediction table
    _COL_MONO = 0
    _COL_HARM = 1
    _COL_UNDE = 2
    _COL_R2   = 3   # Roll2 mean
    _COL_RS   = 4   # Roll2 ±σ
    _COL_X2   = 5   # X2 mean
    _COL_XS   = 6   # X2 ±σ

    def __init__(self, csv_path="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Predict Roll2 & X2 from Alignment History")
        self.resize(980, 720)

        self._csv_path    = csv_path
        self._mono_e      = np.array([])
        self._roll2       = np.array([])
        self._x2          = np.array([])
        self._model_roll2 = None   # callable: MonoE → float (mean)
        self._model_x2    = None
        self._std_roll2   = None   # callable: MonoE → float (±σ) or None
        self._std_x2      = None
        self._use_gp      = False

        self._build_ui()
        if csv_path and os.path.isfile(csv_path):
            self._load_csv(csv_path)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # CSV source row
        src = QHBoxLayout()
        src.addWidget(QLabel("CSV:"))
        self._path_edit = QLineEdit(self._csv_path)
        self._path_edit.setReadOnly(True)
        src.addWidget(self._path_edit, 1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        src.addWidget(browse_btn)
        layout.addLayout(src)

        self._info_lbl = QLabel("No data loaded.")
        self._info_lbl.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._info_lbl)

        # Model selection
        model_grp = QGroupBox("Regression model")
        mg = QHBoxLayout(model_grp)

        self._gp_rb = QRadioButton("Gaussian Process (numpy)")
        self._gp_rb.setChecked(True)
        self._gp_rb.setToolTip(
            "Fits a GP with an RBF kernel; optimises hyperparameters via "
            "log-marginal-likelihood.\nProvides calibrated ±σ uncertainty bands — "
            "best choice for small beamline datasets."
        )
        mg.addWidget(self._gp_rb)

        self._poly_rb = QRadioButton("Polynomial  degree:")
        mg.addWidget(self._poly_rb)

        self._degree_cb = QComboBox()
        self._degree_cb.addItems(["1 (linear)", "2 (quadratic)", "3 (cubic)", "4"])
        self._degree_cb.setCurrentIndex(2)
        self._degree_cb.setEnabled(False)
        self._degree_cb.currentIndexChanged.connect(self._fit_and_update)
        mg.addWidget(self._degree_cb)

        self._spline_rb = QRadioButton("Cubic spline (scipy)")
        if not _SCIPY:
            self._spline_rb.setEnabled(False)
            self._spline_rb.setToolTip("scipy not installed")
        mg.addWidget(self._spline_rb)

        bg = QButtonGroup(self)
        for rb in (self._gp_rb, self._poly_rb, self._spline_rb):
            bg.addButton(rb)
        bg.buttonClicked.connect(self._on_model_changed)

        self._r2_lbl = QLabel("")
        self._r2_lbl.setStyleSheet("font-size: 11px; color: #aaa;")
        mg.addWidget(self._r2_lbl)
        mg.addStretch()
        layout.addWidget(model_grp)

        # Plot area
        if _MPL:
            self._fig = Figure(figsize=(9, 3.0), tight_layout=True)
            self._ax_r, self._ax_x = self._fig.subplots(1, 2)
            self._canvas = FigureCanvasQTAgg(self._fig)
            self._canvas.setMinimumHeight(210)
            layout.addWidget(self._canvas)
        else:
            lbl = QLabel("(matplotlib not installed — no plot)")
            lbl.setStyleSheet("color: #888; font-style: italic;")
            layout.addWidget(lbl)

        # Prediction table
        pred_grp = QGroupBox("Predict new energies")
        pv = QVBoxLayout(pred_grp)

        hint = QHBoxLayout()
        hint.addWidget(QLabel(
            "Enter MonoE → Roll2 / X2 predicted automatically.  "
            "Amber = fill Harmonic & UndE manually.  "
            "Orange background = extrapolation outside training range."
        ))
        hint.addStretch()
        for label, slot in [("Add Row", self._add_pred_row),
                             ("Remove Row", self._remove_pred_row)]:
            b = QPushButton(label)
            b.clicked.connect(slot)
            hint.addWidget(b)
        pv.addLayout(hint)

        self._pred_table = QTableWidget(0, 7)
        self._pred_table.setHorizontalHeaderLabels([
            "MonoE (keV)", "Harmonic", "UndE (eV)",
            "Roll2 (mdeg)", "±σ Roll2",
            "X2 (μm)",     "±σ X2",
        ])
        hdr = self._pred_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hdr.setStretchLastSection(False)
        for c, w in enumerate([90, 70, 80, 100, 70, 90, 70]):
            self._pred_table.setColumnWidth(c, w)
        self._pred_table.setMinimumHeight(130)
        self._pred_table.cellChanged.connect(self._on_pred_cell_changed)
        pv.addWidget(self._pred_table)
        layout.addWidget(pred_grp)

        # Dialog buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        add_btn = QPushButton("Add to Energy Table")
        add_btn.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white;"
            " font-weight: bold; border-radius: 4px; padding: 6px 16px; }"
            "QPushButton:hover { background-color: #43a047; }"
        )
        add_btn.clicked.connect(self.accept)
        btn_row.addWidget(add_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self._add_pred_row()

    # ── CSV loading ───────────────────────────────────────────────────────────

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open alignment CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if path:
            self._path_edit.setText(path)
            self._load_csv(path)

    def _load_csv(self, path):
        try:
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception as e:
            self._info_lbl.setText(f"Error reading file: {e}")
            return

        mono_e, roll2, x2 = [], [], []
        for r in rows:
            try:
                mono_e.append(float(r["MonoE"]))
                roll2.append(float(r["Roll2"]))
                x2.append(float(r["X2"]))
            except (KeyError, ValueError):
                pass

        n = len(mono_e)
        if n < 2:
            self._info_lbl.setText(
                f"Need ≥ 2 valid rows with MonoE, Roll2, X2. Found {n}."
            )
            return

        # Collapse repeated MonoE values to their mean (spline requires unique X)
        mono_arr  = np.array(mono_e)
        roll2_arr = np.array(roll2)
        x2_arr    = np.array(x2)
        unique_e  = np.unique(mono_arr)
        self._mono_e = unique_e
        self._roll2  = np.array([roll2_arr[mono_arr == e].mean() for e in unique_e])
        self._x2     = np.array([x2_arr[mono_arr == e].mean()    for e in unique_e])
        n_u = len(unique_e)

        repeats = n - n_u
        repeat_str = f"  •  {repeats} repeated point(s) averaged" if repeats else ""
        self._info_lbl.setText(
            f"{n} measurements  •  {n_u} unique energies"
            + repeat_str +
            f"  •  MonoE: {self._mono_e.min():.5g}–{self._mono_e.max():.5g} keV"
        )

        # Cap polynomial degree at n_u - 1
        max_deg = min(4, n_u - 1)
        self._degree_cb.blockSignals(True)
        for i in range(4):
            self._degree_cb.model().item(i).setEnabled(i < max_deg)
        if self._degree_cb.currentIndex() >= max_deg:
            self._degree_cb.setCurrentIndex(max_deg - 1)
        self._degree_cb.blockSignals(False)

        self._fit_and_update()

    # ── Model fitting ─────────────────────────────────────────────────────────

    def _on_model_changed(self, btn):
        self._degree_cb.setEnabled(self._poly_rb.isChecked())
        self._fit_and_update()

    def _fit_and_update(self, *_):
        if len(self._mono_e) < 2:
            return

        n = len(self._mono_e)
        self._std_roll2 = None
        self._std_x2    = None

        if self._gp_rb.isChecked():
            self._use_gp = True
            gpr_r = NumpyGPR().fit(self._mono_e, self._roll2)
            gpr_x = NumpyGPR().fit(self._mono_e, self._x2)

            def mean_r(e, _g=gpr_r): return float(_g.predict(np.array([e]))[0])
            def mean_x(e, _g=gpr_x): return float(_g.predict(np.array([e]))[0])
            def std_r(e,  _g=gpr_r): return float(_g.predict(np.array([e]), return_std=True)[1][0])
            def std_x(e,  _g=gpr_x): return float(_g.predict(np.array([e]), return_std=True)[1][0])

            self._model_roll2 = mean_r
            self._model_x2    = mean_x
            self._std_roll2   = std_r
            self._std_x2      = std_x

            r2r = gpr_r.score(self._mono_e, self._roll2)
            r2x = gpr_x.score(self._mono_e, self._x2)
            ls  = gpr_r.length_scale_
            self._r2_lbl.setText(
                f"  GP  R²(Roll2)={r2r:.4f}  R²(X2)={r2x:.4f}"
                f"  ℓ={ls*gpr_r._Xstd:.4g} keV"
            )

        elif self._spline_rb.isChecked() and _SCIPY and n >= 4:
            self._use_gp = False
            k  = min(3, n - 1)
            sr = _USpline(self._mono_e, self._roll2, k=k, s=0)
            sx = _USpline(self._mono_e, self._x2,    k=k, s=0)
            self._model_roll2 = lambda e, _f=sr: float(_f(e))
            self._model_x2    = lambda e, _f=sx: float(_f(e))
            self._r2_lbl.setText(f"  Cubic spline (k={k}) — interpolating")

        else:
            self._use_gp = False
            deg = min(self._degree_cb.currentIndex() + 1, n - 1)
            pr  = np.poly1d(np.polyfit(self._mono_e, self._roll2, deg))
            px  = np.poly1d(np.polyfit(self._mono_e, self._x2,    deg))
            # constant ±σ band = residual RMSE
            rmse_r = float(np.sqrt(np.mean((self._roll2 - pr(self._mono_e)) ** 2)))
            rmse_x = float(np.sqrt(np.mean((self._x2    - px(self._mono_e)) ** 2)))
            self._model_roll2 = lambda e, _p=pr: float(_p(e))
            self._model_x2    = lambda e, _p=px: float(_p(e))
            self._std_roll2   = lambda e, _s=rmse_r: _s
            self._std_x2      = lambda e, _s=rmse_x: _s
            r2r = self._r2(self._roll2, pr(self._mono_e))
            r2x = self._r2(self._x2,   px(self._mono_e))
            self._r2_lbl.setText(
                f"  Poly deg {deg}  R²(Roll2)={r2r:.4f}  R²(X2)={r2x:.4f}"
            )

        self._update_plot()
        self._recompute_predictions()

    @staticmethod
    def _r2(y_true, y_pred):
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    # ── Plot ──────────────────────────────────────────────────────────────────

    def _update_plot(self):
        if not _MPL or self._model_roll2 is None:
            return

        e_min, e_max = self._mono_e.min(), self._mono_e.max()
        margin = (e_max - e_min) * 0.12 or 1.0
        xs = np.linspace(e_min - margin, e_max + margin, 400)

        for ax, data, mean_fn, std_fn, ylabel in [
            (self._ax_r, self._roll2, self._model_roll2, self._std_roll2, "Roll2 (mdeg)"),
            (self._ax_x, self._x2,   self._model_x2,    self._std_x2,    "X2 (μm)"),
        ]:
            ax.clear()

            mu  = np.array([mean_fn(x) for x in xs])
            if std_fn is not None:
                sd  = np.array([std_fn(x) for x in xs])
                ax.fill_between(xs, mu - 2 * sd, mu + 2 * sd,
                                alpha=0.15, color="#1976d2", label="±2σ")
                ax.fill_between(xs, mu - sd, mu + sd,
                                alpha=0.32, color="#1976d2", label="±1σ")

            ax.plot(xs, mu, "-", lw=2, color="#1976d2", label="Predicted mean")
            ax.plot(self._mono_e, data, "o", ms=6, color="#e65100",
                    zorder=5, label="Training (mean/energy)")

            # Training range markers
            for xv in [e_min, e_max]:
                ax.axvline(xv, color="#666", lw=1, ls="--", alpha=0.45)

            # Prediction point markers
            for r in range(self._pred_table.rowCount()):
                it = self._pred_table.item(r, self._COL_MONO)
                try:
                    ax.axvline(float(it.text().strip()),
                               color="#ffa726", lw=1.5, ls=":", alpha=0.8)
                except (ValueError, AttributeError):
                    pass

            ax.set_xlabel("MonoE (keV)", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.tick_params(labelsize=8)
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.18)

        self._canvas.draw()

    # ── Prediction table ──────────────────────────────────────────────────────

    def _add_pred_row(self):
        self._pred_table.blockSignals(True)
        r = self._pred_table.rowCount()
        self._pred_table.insertRow(r)
        for c in range(7):
            item = QTableWidgetItem("")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if c in (self._COL_HARM, self._COL_UNDE):
                item.setBackground(self._AMBER_BG)
                item.setForeground(self._AMBER_FG)
            elif c in (self._COL_R2, self._COL_RS, self._COL_X2, self._COL_XS):
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setForeground(QColor("#888888"))
            self._pred_table.setItem(r, c, item)
        self._pred_table.blockSignals(False)

    def _remove_pred_row(self):
        rows = sorted({idx.row() for idx in self._pred_table.selectedIndexes()},
                      reverse=True)
        if not rows:
            rows = [self._pred_table.rowCount() - 1]
        for r in rows:
            if r >= 0:
                self._pred_table.removeRow(r)

    def _on_pred_cell_changed(self, row, col):
        if col == self._COL_MONO:
            self._recompute_one_row(row)
            if _MPL:
                self._update_plot()

    def _recompute_predictions(self):
        for r in range(self._pred_table.rowCount()):
            self._recompute_one_row(r)

    def _recompute_one_row(self, row):
        if self._model_roll2 is None:
            return
        it = self._pred_table.item(row, self._COL_MONO)
        try:
            e = float(it.text().strip())
        except (ValueError, AttributeError):
            return

        outside = (len(self._mono_e) > 0
                   and (e < self._mono_e.min() or e > self._mono_e.max()))

        self._pred_table.blockSignals(True)
        for col, mean_fn, std_fn, std_col in [
            (self._COL_R2, self._model_roll2, self._std_roll2, self._COL_RS),
            (self._COL_X2, self._model_x2,    self._std_x2,    self._COL_XS),
        ]:
            for c_idx, val in [
                (col,     f"{mean_fn(e):.6g}"),
                (std_col, f"{std_fn(e):.4g}" if std_fn else "—"),
            ]:
                it2 = self._pred_table.item(row, c_idx)
                if it2 is None:
                    continue
                it2.setText(val)
                if outside:
                    it2.setBackground(self._EXTRAP_BG)
                    it2.setForeground(self._EXTRAP_FG)
                else:
                    it2.setBackground(QTableWidgetItem().background())
                    it2.setForeground(QColor("#888888"))
        self._pred_table.blockSignals(False)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_predicted_rows(self):
        """Return [[MonoE, Harmonic, UndE, Roll2, X2], ...] for rows with valid MonoE."""
        rows = []
        for r in range(self._pred_table.rowCount()):
            try:
                def _cell(c):
                    it = self._pred_table.item(r, c)
                    text = (it.text().strip() if it else "") or "0"
                    return float(text)
                mono_e = _cell(self._COL_MONO)
                if mono_e == 0.0:
                    continue
                rows.append([
                    mono_e,
                    _cell(self._COL_HARM),
                    _cell(self._COL_UNDE),
                    _cell(self._COL_R2),
                    _cell(self._COL_X2),
                ])
            except ValueError:
                pass
        return rows
