"""Predict Roll2 / X2 for new MonoE values by fitting the alignment CSV history."""
import csv
import os

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem, QFileDialog, QGroupBox,
    QRadioButton, QButtonGroup, QHeaderView,
)

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

    def __init__(self, csv_path="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Predict Roll2 & X2 from Alignment History")
        self.resize(880, 700)

        self._csv_path    = csv_path
        self._mono_e      = np.array([])
        self._roll2       = np.array([])
        self._x2          = np.array([])
        self._model_roll2 = None   # callable MonoE → Roll2
        self._model_x2    = None   # callable MonoE → X2

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

        self._poly_rb = QRadioButton("Polynomial  degree:")
        self._poly_rb.setChecked(True)
        mg.addWidget(self._poly_rb)

        self._degree_cb = QComboBox()
        self._degree_cb.addItems(["1 (linear)", "2 (quadratic)", "3 (cubic)", "4"])
        self._degree_cb.setCurrentIndex(2)
        self._degree_cb.currentIndexChanged.connect(self._fit_and_update)
        mg.addWidget(self._degree_cb)

        self._spline_rb = QRadioButton("Cubic spline (scipy)")
        if not _SCIPY:
            self._spline_rb.setEnabled(False)
            self._spline_rb.setToolTip("scipy not installed")
        mg.addWidget(self._spline_rb)

        bg = QButtonGroup(self)
        bg.addButton(self._poly_rb)
        bg.addButton(self._spline_rb)
        bg.buttonClicked.connect(self._on_model_changed)

        self._r2_lbl = QLabel("")
        self._r2_lbl.setStyleSheet("font-size: 11px; color: #aaa;")
        mg.addWidget(self._r2_lbl)
        mg.addStretch()
        layout.addWidget(model_grp)

        # Plot area
        if _MPL:
            self._fig = Figure(figsize=(8.5, 2.8), tight_layout=True)
            self._ax_r, self._ax_x = self._fig.subplots(1, 2)
            self._canvas = FigureCanvasQTAgg(self._fig)
            self._canvas.setMinimumHeight(200)
            layout.addWidget(self._canvas)
        else:
            no_plot = QLabel("(matplotlib not installed — no plot)")
            no_plot.setStyleSheet("color: #888; font-style: italic;")
            layout.addWidget(no_plot)

        # Prediction table
        pred_grp = QGroupBox("Predict new energies")
        pv = QVBoxLayout(pred_grp)

        hint = QHBoxLayout()
        hint.addWidget(QLabel(
            "Enter MonoE (keV); Roll2 and X2 are filled automatically.  "
            "Orange background = extrapolation outside training range."
        ))
        hint.addStretch()
        for label, slot in [("Add Row", self._add_pred_row),
                             ("Remove Row", self._remove_pred_row)]:
            b = QPushButton(label)
            b.clicked.connect(slot)
            hint.addWidget(b)
        pv.addLayout(hint)

        self._pred_table = QTableWidget(0, 5)
        self._pred_table.setHorizontalHeaderLabels(
            ["MonoE (keV)", "Harmonic", "UndE (eV)", "Roll2 (mdeg)", "X2 (μm)"])
        self._pred_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
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

        order = np.argsort(mono_e)
        self._mono_e = np.array(mono_e)[order]
        self._roll2  = np.array(roll2)[order]
        self._x2     = np.array(x2)[order]

        self._info_lbl.setText(
            f"{n} data point{'s' if n != 1 else ''}  •  "
            f"MonoE: {self._mono_e.min():.5g} – {self._mono_e.max():.5g} keV"
        )

        # Cap polynomial degree at n - 1
        max_deg = min(4, n - 1)
        self._degree_cb.blockSignals(True)
        for i in range(4):
            self._degree_cb.model().item(i).setEnabled(i < max_deg)
        if self._degree_cb.currentIndex() >= max_deg:
            self._degree_cb.setCurrentIndex(max_deg - 1)
        self._degree_cb.blockSignals(False)

        self._fit_and_update()

    # ── Model fitting ─────────────────────────────────────────────────────────

    def _on_model_changed(self, *_):
        self._degree_cb.setEnabled(self._poly_rb.isChecked())
        self._fit_and_update()

    def _fit_and_update(self, *_):
        if len(self._mono_e) < 2:
            return

        use_spline = (self._spline_rb.isChecked()
                      and _SCIPY
                      and len(self._mono_e) >= 4)

        if use_spline:
            k = min(3, len(self._mono_e) - 1)
            sr = _USpline(self._mono_e, self._roll2, k=k, s=0)
            sx = _USpline(self._mono_e, self._x2,    k=k, s=0)
            # capture loop variables explicitly to avoid closure aliasing
            self._model_roll2 = lambda e, _f=sr: float(_f(e))
            self._model_x2    = lambda e, _f=sx: float(_f(e))
            self._r2_lbl.setText(f"  Interpolating cubic spline (k={k})")
        else:
            deg = min(self._degree_cb.currentIndex() + 1, len(self._mono_e) - 1)
            pr = np.poly1d(np.polyfit(self._mono_e, self._roll2, deg))
            px = np.poly1d(np.polyfit(self._mono_e, self._x2,    deg))
            self._model_roll2 = lambda e, _p=pr: float(_p(e))
            self._model_x2    = lambda e, _p=px: float(_p(e))
            r2r = self._r2(self._roll2, pr(self._mono_e))
            r2x = self._r2(self._x2,   px(self._mono_e))
            self._r2_lbl.setText(f"  R²(Roll2)={r2r:.4f}   R²(X2)={r2x:.4f}")

        self._update_plot()
        self._recompute_predictions()

    @staticmethod
    def _r2(y_true, y_pred):
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    def _update_plot(self):
        if not _MPL or self._model_roll2 is None:
            return
        xs = np.linspace(self._mono_e.min(), self._mono_e.max(), 400)
        for ax, data, fn, ylabel in [
            (self._ax_r, self._roll2, self._model_roll2, "Roll2 (mdeg)"),
            (self._ax_x, self._x2,   self._model_x2,    "X2 (μm)"),
        ]:
            ax.clear()
            ax.plot(self._mono_e, data, "o", ms=6, color="#4C72B0",
                    label="Measured", zorder=3)
            ax.plot(xs, [fn(x) for x in xs], "-", lw=2, color="#DD8452",
                    label="Fit", alpha=0.9)
            ax.set_xlabel("MonoE (keV)", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.tick_params(labelsize=8)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)
        self._canvas.draw()

    # ── Prediction table ──────────────────────────────────────────────────────

    def _add_pred_row(self):
        self._pred_table.blockSignals(True)
        r = self._pred_table.rowCount()
        self._pred_table.insertRow(r)
        for c in range(5):
            item = QTableWidgetItem("")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if c in (3, 4):
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
        if col == 0:
            self._recompute_one_row(row)

    def _recompute_predictions(self):
        for r in range(self._pred_table.rowCount()):
            self._recompute_one_row(r)

    def _recompute_one_row(self, row):
        if self._model_roll2 is None:
            return
        item = self._pred_table.item(row, 0)
        try:
            e = float(item.text().strip())
        except (ValueError, AttributeError):
            return

        outside = (len(self._mono_e) > 0
                   and (e < self._mono_e.min() or e > self._mono_e.max()))

        self._pred_table.blockSignals(True)
        for col, fn in [(3, self._model_roll2), (4, self._model_x2)]:
            it = self._pred_table.item(row, col)
            if it is None:
                continue
            it.setText(f"{fn(e):.6g}")
            if outside:
                it.setBackground(QColor("#5a3a00"))
                it.setForeground(QColor("#ffb74d"))
            else:
                it.setBackground(QTableWidgetItem().background())
                it.setForeground(QColor("#888888"))
        self._pred_table.blockSignals(False)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_predicted_rows(self):
        """Return [[MonoE, Harmonic, UndE, Roll2, X2], ...] for rows with a valid MonoE."""
        rows = []
        for r in range(self._pred_table.rowCount()):
            try:
                vals = []
                for c in range(5):
                    item = self._pred_table.item(r, c)
                    text = (item.text().strip() if item else "") or "0"
                    vals.append(float(text))
                if vals[0] != 0.0:
                    rows.append(vals)
            except ValueError:
                pass
        return rows
