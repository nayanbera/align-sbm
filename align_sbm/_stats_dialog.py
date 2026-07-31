"""Statistical analysis dialog for alignment CSV data."""
import csv
import os

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTabWidget, QWidget, QTextBrowser, QFileDialog, QMessageBox,
    QListWidget, QListWidgetItem, QGroupBox, QComboBox, QSizePolicy,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure
    import matplotlib.cm as _cm
    _MPL = True
except ImportError:
    _MPL = False

try:
    from scipy import stats as _sstats
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

from ._ml import NumpyGPR

_GROUPING_COLS  = {"MonoE", "Harmonic"}
_DEFAULT_CHECKED = {"Roll2", "X2"}

_AMBER_BG = QColor("#4a3800")
_AMBER_FG = QColor("#ffc107")
_EXTRAP_BG = QColor("#5a3a00")
_EXTRAP_FG = QColor("#ffb74d")


class StatsDialog(QDialog):
    """Load an alignment CSV, show statistics, plots, and ML-based predictions."""

    def __init__(self, csv_path="", energy_tab=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CSV Statistical Analysis & Prediction")
        self.resize(1200, 920)

        self._csv_path    = csv_path
        self._energy_tab  = energy_tab  # for "Send to Energy Table"
        self._data        = {}
        self._num_cols    = []
        self._n_rows      = 0
        self._report_html = ""
        # Trained models: col → {"mean_fn": callable, "std_fn": callable,
        #                         "train_r2": float, "cv_r2": float, "rmse": float}
        self._trained     = {}
        self._train_mono  = np.array([])   # unique MonoE values used for training
        self._train_data  = {}             # col → mean values at _train_mono

        self._build_ui()
        if csv_path and os.path.isfile(csv_path):
            self._load_csv(csv_path)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # CSV source row
        src = QHBoxLayout()
        src.addWidget(QLabel("CSV:"))
        self._path_edit = QLineEdit(self._csv_path)
        self._path_edit.setReadOnly(True)
        src.addWidget(self._path_edit, 1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        src.addWidget(browse_btn)
        analyze_btn = QPushButton("Analyze")
        analyze_btn.clicked.connect(lambda: self._load_csv(self._path_edit.text()))
        src.addWidget(analyze_btn)
        layout.addLayout(src)

        self._info_lbl = QLabel("No data loaded.")
        self._info_lbl.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._info_lbl)

        # Column selector
        col_grp = QGroupBox("Columns to analyze")
        col_grp.setMaximumHeight(145)
        col_h = QHBoxLayout(col_grp)
        self._col_list = QListWidget()
        self._col_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self._col_list.setFlow(QListWidget.Flow.LeftToRight)
        self._col_list.setWrapping(True)
        self._col_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._col_list.setSpacing(2)
        self._col_list.setStyleSheet("QListWidget { border: none; }")
        col_h.addWidget(self._col_list, 1)

        scatter_v = QVBoxLayout()
        scatter_v.setSpacing(4)
        scatter_v.addWidget(QLabel("Scatter X:"))
        self._scatter_x = QComboBox()
        scatter_v.addWidget(self._scatter_x)
        scatter_v.addWidget(QLabel("Scatter Y:"))
        self._scatter_y = QComboBox()
        scatter_v.addWidget(self._scatter_y)
        scatter_v.addStretch()
        refresh_btn = QPushButton("Refresh Plots")
        refresh_btn.clicked.connect(self._refresh)
        scatter_v.addWidget(refresh_btn)
        col_h.addLayout(scatter_v)
        layout.addWidget(col_grp)

        # Main tabs
        self._tabs = QTabWidget()

        # Plots tab
        if _MPL:
            self._plot_container = QWidget()
            pv = QVBoxLayout(self._plot_container)
            pv.setContentsMargins(0, 0, 0, 0)
            pv.setSpacing(0)
            self._canvas  = None
            self._toolbar = None
            self._tabs.addTab(self._plot_container, "Plots")
        else:
            lbl = QLabel("matplotlib not installed — plots unavailable")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #888; font-style: italic;")
            self._tabs.addTab(lbl, "Plots")

        # Report tab
        self._report = QTextBrowser()
        self._report.setOpenExternalLinks(False)
        self._tabs.addTab(self._report, "Report")

        # Predict tab
        self._tabs.addTab(self._build_predict_tab(), "Predict")

        layout.addWidget(self._tabs, 1)

        # Bottom buttons
        btn_row = QHBoxLayout()
        self._export_btn = QPushButton("Export Report…")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_report)
        btn_row.addWidget(self._export_btn)
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    # ── Predict tab ───────────────────────────────────────────────────────────

    def _build_predict_tab(self):
        widget = QWidget()
        vbox = QVBoxLayout(widget)
        vbox.setSpacing(6)

        # ── Model controls row ────────────────────────────────────────────────
        ctrl_grp = QGroupBox("Model")
        ctrl_h = QHBoxLayout(ctrl_grp)

        ctrl_h.addWidget(QLabel("Algorithm:"))
        self._model_cb = QComboBox()
        self._model_cb.setMinimumWidth(220)
        models = [
            "Gaussian Process (numpy/scipy)",   # always available
            "Polynomial deg 1 (linear)",
            "Polynomial deg 2 (quadratic)",
            "Polynomial deg 3 (cubic)",
            "Polynomial deg 4",
        ]
        if _SKLEARN:
            models += [
                "Gaussian Process (sklearn RBF)",
                "Random Forest (200 trees)",
                "Gradient Boosting",
            ]
        self._model_cb.addItems(models)
        self._model_cb.setCurrentIndex(0)   # numpy GP is the default
        self._model_cb.setToolTip(
            "Gaussian Process (numpy/scipy): always available, calibrated ±σ bands,\n"
            "best for small datasets (5–30 energies).\n"
            "sklearn models available after: pip install scikit-learn"
        )
        ctrl_h.addWidget(self._model_cb)

        ctrl_h.addWidget(QLabel("CV:"))
        self._cv_cb = QComboBox()
        self._cv_cb.addItems(["LOO", "3-fold", "5-fold", "10-fold"])
        self._cv_cb.setCurrentText("LOO")
        ctrl_h.addWidget(self._cv_cb)

        self._use_means_cb = QCheckBox("Mean per energy")
        self._use_means_cb.setChecked(True)
        self._use_means_cb.setToolTip(
            "If checked, train on the per-energy mean rather than raw measurements.\n"
            "Reduces noise from repeated alignments at the same energy."
        )
        ctrl_h.addWidget(self._use_means_cb)

        ctrl_h.addStretch()
        self._train_btn = QPushButton("Train Models")
        self._train_btn.setEnabled(False)
        self._train_btn.setStyleSheet(
            "QPushButton { background-color: #1565c0; color: white; font-weight: bold;"
            " border-radius: 4px; padding: 5px 14px; }"
            "QPushButton:hover { background-color: #1976d2; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self._train_btn.clicked.connect(self._train_models)
        ctrl_h.addWidget(self._train_btn)
        vbox.addWidget(ctrl_grp)

        # ── Performance table ─────────────────────────────────────────────────
        perf_grp = QGroupBox("Model performance")
        perf_v = QVBoxLayout(perf_grp)
        self._perf_table = QTableWidget(0, 4)
        self._perf_table.setHorizontalHeaderLabels(
            ["Target column", "Train R²", "CV R²", "RMSE"]
        )
        self._perf_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._perf_table.setMaximumHeight(130)
        self._perf_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._perf_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        perf_v.addWidget(self._perf_table)
        vbox.addWidget(perf_grp)

        # ── Fit plot ──────────────────────────────────────────────────────────
        if _MPL:
            plot_grp = QGroupBox("Fitted curves")
            plot_v = QVBoxLayout(plot_grp)
            plot_v.setContentsMargins(0, 2, 0, 2)

            plot_ctrl = QHBoxLayout()
            plot_ctrl.addWidget(QLabel("Plot column:"))
            self._pred_plot_col = QComboBox()
            self._pred_plot_col.currentIndexChanged.connect(self._update_pred_plot)
            plot_ctrl.addWidget(self._pred_plot_col)
            plot_ctrl.addStretch()
            plot_v.addLayout(plot_ctrl)

            self._pred_fig    = Figure(figsize=(10, 2.8), constrained_layout=True)
            self._pred_canvas = FigureCanvasQTAgg(self._pred_fig)
            self._pred_canvas.setMinimumHeight(200)
            self._pred_toolbar = NavigationToolbar2QT(self._pred_canvas, plot_grp)
            plot_v.addWidget(self._pred_toolbar)
            plot_v.addWidget(self._pred_canvas)
            vbox.addWidget(plot_grp)
        else:
            self._pred_plot_col = QComboBox()   # dummy, not shown

        # ── Prediction input table ────────────────────────────────────────────
        pred_grp = QGroupBox("Predict for new energies")
        pred_v = QVBoxLayout(pred_grp)
        pred_v.setSpacing(4)

        hint_row = QHBoxLayout()
        self._pred_hint = QLabel(
            "Train a model first, then enter MonoE values to get predictions."
        )
        self._pred_hint.setStyleSheet("color: #888; font-size: 11px;")
        self._pred_hint.setWordWrap(True)
        hint_row.addWidget(self._pred_hint, 1)
        for label, slot in [("+ Row", self._add_pred_row),
                             ("Remove", self._remove_pred_row)]:
            b = QPushButton(label)
            b.setMaximumWidth(65)
            b.clicked.connect(slot)
            hint_row.addWidget(b)
        pred_v.addLayout(hint_row)

        self._pred_table = QTableWidget(0, 1)   # columns rebuilt after training
        self._pred_table.setHorizontalHeaderLabels(["MonoE (keV)"])
        self._pred_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self._pred_table.setMinimumHeight(110)
        self._pred_table.cellChanged.connect(self._on_pred_cell_changed)
        pred_v.addWidget(self._pred_table)

        send_row = QHBoxLayout()
        send_row.addStretch()
        self._send_btn = QPushButton("Send selected to Energy Table")
        self._send_btn.setEnabled(False)
        self._send_btn.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white;"
            " font-weight: bold; border-radius: 4px; padding: 6px 16px; }"
            "QPushButton:hover { background-color: #43a047; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self._send_btn.setToolTip(
            "Add predicted rows to the Energy Table.\n"
            "Roll2 and X2 (and any extra PVs) are filled from the model.\n"
            "Harmonic and UndE must be entered manually (amber cells)."
        )
        self._send_btn.clicked.connect(self._send_to_energy_table)
        send_row.addWidget(self._send_btn)
        pred_v.addLayout(send_row)
        vbox.addWidget(pred_grp)

        return widget

    # ── Data loading ──────────────────────────────────────────────────────────

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open alignment CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if path:
            self._path_edit.setText(path)
            self._load_csv(path)

    def _load_csv(self, path):
        if not path or not os.path.isfile(path):
            self._info_lbl.setText("File not found.")
            return
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = list(reader.fieldnames or [])
        except Exception as e:
            self._info_lbl.setText(f"Error reading file: {e}")
            return
        if not rows:
            self._info_lbl.setText("File is empty.")
            return

        self._n_rows   = len(rows)
        self._csv_path = path
        self._trained  = {}

        raw = {k: [r.get(k, "").strip() for r in rows] for k in fieldnames}
        self._data    = {}
        self._num_cols = []
        for k, vals in raw.items():
            if k.lower() == "datetime":
                continue
            try:
                self._data[k] = np.array([float(v) for v in vals])
                self._num_cols.append(k)
            except (ValueError, TypeError):
                pass

        mono     = self._data.get("MonoE", np.array([]))
        unique_e = np.unique(mono) if len(mono) else np.array([])
        e_str    = ", ".join(f"{e:.4g}" for e in unique_e[:8])
        if len(unique_e) > 8:
            e_str += f" … ({len(unique_e)} total)"
        loop_str = ""
        if len(unique_e):
            cnts     = [int(np.sum(mono == e)) for e in unique_e]
            loop_str = (
                f" • {cnts[0]} measurement(s) per energy"
                if len(set(cnts)) == 1
                else f" • {min(cnts)}–{max(cnts)} measurements per energy"
            )
        self._info_lbl.setText(
            f"{self._n_rows} rows  •  {len(self._num_cols)} numeric columns"
            + (f"  •  MonoE: {e_str} keV" if e_str else "") + loop_str
        )

        self._rebuild_col_selector()
        self._train_btn.setEnabled(True)
        self._send_btn.setEnabled(False)
        self._perf_table.setRowCount(0)
        self._refresh()
        self._export_btn.setEnabled(True)

    # ── Column selector ───────────────────────────────────────────────────────

    def _rebuild_col_selector(self):
        prev_checked = {
            self._col_list.item(i).text()
            for i in range(self._col_list.count())
            if self._col_list.item(i).checkState() == Qt.CheckState.Checked
        }
        self._col_list.clear()
        analysis_cols = [c for c in self._num_cols if c not in _GROUPING_COLS]
        for col in analysis_cols:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            if col in prev_checked:
                checked = True
            elif prev_checked:
                checked = False
            else:
                checked = col in _DEFAULT_CHECKED
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            self._col_list.addItem(item)

        self._scatter_x.blockSignals(True)
        self._scatter_y.blockSignals(True)
        prev_x = self._scatter_x.currentText()
        prev_y = self._scatter_y.currentText()
        self._scatter_x.clear()
        self._scatter_y.clear()
        self._scatter_x.addItems(analysis_cols)
        self._scatter_y.addItems(analysis_cols)
        xi = self._scatter_x.findText(prev_x or "Roll2")
        yi = self._scatter_y.findText(prev_y or "X2")
        self._scatter_x.setCurrentIndex(max(0, xi))
        self._scatter_y.setCurrentIndex(max(0, yi if yi >= 0 else min(1, self._scatter_y.count() - 1)))
        self._scatter_x.blockSignals(False)
        self._scatter_y.blockSignals(False)

    def _selected_cols(self):
        return [
            self._col_list.item(i).text()
            for i in range(self._col_list.count())
            if self._col_list.item(i).checkState() == Qt.CheckState.Checked
        ]

    # ── Statistics refresh ────────────────────────────────────────────────────

    def _refresh(self):
        if not self._data:
            return
        sel = self._selected_cols()
        if _MPL:
            self._make_plots(sel)
        self._make_report(sel)

    # ── Plots (statistics tab) ────────────────────────────────────────────────

    def _make_plots(self, sel_cols):
        layout = self._plot_container.layout()
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()

        d    = self._data
        n    = self._n_rows
        idx  = np.arange(n)
        mono = d.get("MonoE", np.zeros(n))
        ue   = np.unique(mono)
        cmap_fn  = _cm.tab10
        color_of = {e: cmap_fn(i % 10) for i, e in enumerate(ue)}

        n_sel  = len(sel_cols)
        n_rows = max(n_sel, 1) + 1
        fig_h  = max(6.0, 2.8 * n_rows)

        fig    = Figure(figsize=(12, fig_h), constrained_layout=True)
        canvas = FigureCanvasQTAgg(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        toolbar = NavigationToolbar2QT(canvas, self._plot_container)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        self._canvas  = canvas
        self._toolbar = toolbar

        axes = fig.subplots(n_rows, 2)
        if n_rows == 1:
            axes = np.array([axes])

        def _setup(ax, title, xlabel, ylabel):
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.18)

        for row_i, col in enumerate(sel_cols):
            ax_ts  = axes[row_i, 0]
            ax_bar = axes[row_i, 1]
            if col in d:
                for e in ue:
                    mask = mono == e
                    ax_ts.plot(idx[mask], d[col][mask], "o-", ms=4, lw=1,
                               color=color_of[e], label=f"{e:.4g} keV")
                if len(ue) <= 8:
                    ax_ts.legend(fontsize=6, ncol=2, loc="best")
                _setup(ax_ts, f"{col} vs measurement index", "Index", col)
                if len(ue):
                    xs    = np.arange(len(ue))
                    means = [d[col][mono == e].mean() for e in ue]
                    stds  = [d[col][mono == e].std()  for e in ue]
                    ax_bar.bar(xs, means, yerr=stds,
                               color=[color_of[e] for e in ue],
                               capsize=4, alpha=0.82,
                               error_kw={"elinewidth": 1.5, "capthick": 1.5})
                    ax_bar.set_xticks(xs)
                    ax_bar.set_xticklabels([f"{e:.4g}" for e in ue],
                                           rotation=30, ha="right", fontsize=7)
                _setup(ax_bar, f"{col}: mean ± std per energy", "MonoE (keV)", col)
            else:
                ax_ts.set_visible(False)
                ax_bar.set_visible(False)

        ax_corr   = axes[n_rows - 1, 0]
        ax_scatter = axes[n_rows - 1, 1]

        corr_keys = [k for k in self._num_cols if k not in ("Harmonic",)]
        if len(corr_keys) >= 2:
            mat = np.corrcoef(np.array([d[k] for k in corr_keys]))
            fig.colorbar(
                ax_corr.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto"),
                ax=ax_corr, shrink=0.75, label="r"
            )
            ticks = range(len(corr_keys))
            ax_corr.set_xticks(ticks)
            ax_corr.set_xticklabels(corr_keys, rotation=45, ha="right", fontsize=7)
            ax_corr.set_yticks(ticks)
            ax_corr.set_yticklabels(corr_keys, fontsize=7)
            for i in range(len(corr_keys)):
                for j in range(len(corr_keys)):
                    ax_corr.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                                 fontsize=6.5,
                                 color="white" if abs(mat[i, j]) > 0.65 else "black")
        ax_corr.set_title("Pearson correlation matrix", fontsize=9, fontweight="bold")
        ax_corr.tick_params(labelsize=7)

        sx = self._scatter_x.currentText()
        sy = self._scatter_y.currentText()
        if sx in d and sy in d and sx != sy:
            for e in ue:
                mask = mono == e
                ax_scatter.scatter(d[sx][mask], d[sy][mask], s=28,
                                   color=color_of[e], alpha=0.85,
                                   edgecolors="none", label=f"{e:.4g} keV")
            if len(ue) <= 8:
                ax_scatter.legend(fontsize=6, loc="best")
        elif sx in d:
            ax_scatter.hist(d[sx], bins=20, alpha=0.75)
            sy = f"Histogram of {sx}"
        _setup(ax_scatter, f"{sx} vs {sy}", sx, sy)

        canvas.draw()

    # ── ML training ──────────────────────────────────────────────────────────

    def _train_models(self):
        if not self._data:
            return

        # Determine training targets
        sel_cols = [c for c in self._selected_cols() if c in self._data and c != "MonoE"]
        if not sel_cols:
            QMessageBox.warning(self, "No columns",
                                "Select at least one target column in 'Columns to analyze'.")
            return

        mono = self._data.get("MonoE", np.array([]))
        if len(mono) == 0:
            QMessageBox.warning(self, "No MonoE", "CSV must have a MonoE column.")
            return

        ue = np.unique(mono)
        n_unique = len(ue)

        if n_unique < 2:
            QMessageBox.warning(self, "Insufficient data",
                                "Need at least 2 unique MonoE values to train a model.")
            return

        # Training data: mean per unique MonoE, or all raw points
        use_means = self._use_means_cb.isChecked()
        if use_means:
            X_train = ue.reshape(-1, 1)
            Y_trains = {col: np.array([self._data[col][mono == e].mean() for e in ue])
                        for col in sel_cols}
        else:
            order    = np.argsort(mono)
            X_train  = mono[order].reshape(-1, 1)
            Y_trains = {col: self._data[col][order] for col in sel_cols}
            n_unique  = len(X_train)   # for CV

        self._train_mono = X_train.ravel()
        self._train_data = Y_trains

        model_name = self._model_cb.currentText()
        cv_str     = self._cv_cb.currentText()
        cv_folds   = n_unique if cv_str == "LOO" else int(cv_str.split("-")[0])
        cv_folds   = min(cv_folds, len(X_train))

        self._trained = {}
        self._perf_table.setRowCount(0)

        for col in sel_cols:
            y = Y_trains[col]
            try:
                result = self._fit_one(X_train, y, model_name, cv_folds)
                result["col"] = col
                self._trained[col] = result
            except Exception as exc:
                QMessageBox.warning(self, "Training error",
                                    f"Failed to train model for {col}:\n{exc}")
                continue

            # Add row to performance table
            r = self._perf_table.rowCount()
            self._perf_table.insertRow(r)
            for c_idx, val, fmt in [
                (0, col,                 None),
                (1, result["train_r2"],  ".4f"),
                (2, result["cv_r2"],     ".4f"),
                (3, result["rmse"],      ".4g"),
            ]:
                text = val if fmt is None else f"{val:{fmt}}"
                item = QTableWidgetItem(str(text))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if c_idx in (1, 2) and isinstance(val, float):
                    if val > 0.95:
                        item.setForeground(QColor("#66bb6a"))
                    elif val < 0.8:
                        item.setForeground(QColor("#ef5350"))
                self._perf_table.setItem(r, c_idx, item)

        if not self._trained:
            return

        # Rebuild prediction table columns
        self._rebuild_pred_table_cols(sel_cols)

        # Update fit-plot column combo
        self._pred_plot_col.blockSignals(True)
        prev = self._pred_plot_col.currentText()
        self._pred_plot_col.clear()
        self._pred_plot_col.addItems(list(self._trained.keys()))
        idx = self._pred_plot_col.findText(prev)
        self._pred_plot_col.setCurrentIndex(max(0, idx))
        self._pred_plot_col.blockSignals(False)

        # Show warning if few training points
        if len(X_train) < 4:
            n = len(X_train)
            self._pred_hint.setText(
                f"⚠ Only {n} training point{'s' if n!=1 else ''} — predictions may be unreliable outside the trained range."
            )
        else:
            self._pred_hint.setText(
                "Enter MonoE values below. Amber cells = enter Harmonic / UndE manually. "
                "Orange = extrapolation beyond training range."
            )

        if _MPL:
            self._update_pred_plot()
        self._send_btn.setEnabled(self._energy_tab is not None)
        self._recompute_predictions()

    def _fit_one(self, X_train, y_train, model_name, cv_folds):
        """Train one model; return dict with mean_fn, std_fn, train_r2, cv_r2, rmse."""
        n = len(y_train)

        # ── Polynomial ────────────────────────────────────────────────────────
        if model_name.startswith("Polynomial"):
            deg = int(model_name.split("deg")[1].split()[0])
            deg = min(deg, n - 1)
            coeffs = np.polyfit(X_train.ravel(), y_train, deg)
            poly   = np.poly1d(coeffs)
            y_pred = poly(X_train.ravel())
            resid  = y_train - y_pred
            rmse   = float(np.sqrt(np.mean(resid ** 2)))
            train_r2 = float(self._r2(y_train, y_pred))

            # LOO or k-fold CV
            cv_r2s = []
            for i in range(n):
                mask = np.ones(n, bool)
                mask[i] = False
                if cv_folds < n:
                    # k-fold: group indices
                    fold_size = n // cv_folds
                    start = i * fold_size
                    end   = min(start + fold_size, n)
                    mask  = np.ones(n, bool)
                    mask[start:end] = False
                    if i >= cv_folds:
                        break
                X_tr, y_tr = X_train[mask], y_train[mask]
                X_va, y_va = X_train[~mask], y_train[~mask]
                if len(X_tr) < deg + 1 or len(X_va) == 0:
                    continue
                c2 = np.polyfit(X_tr.ravel(), y_tr, min(deg, len(X_tr) - 1))
                p2 = np.poly1d(c2)
                cv_r2s.append(self._r2(y_va, p2(X_va.ravel())))
            cv_r2 = float(np.mean(cv_r2s)) if cv_r2s else train_r2

            sigma = rmse   # constant uncertainty band for polynomial

            def mean_fn(Xnew, _p=poly):
                return _p(np.asarray(Xnew).ravel())

            def std_fn(Xnew, _s=sigma):
                return np.full(len(np.asarray(Xnew).ravel()), _s)

            return {"mean_fn": mean_fn, "std_fn": std_fn,
                    "train_r2": train_r2, "cv_r2": cv_r2, "rmse": rmse}

        # ── Gaussian Process (numpy/scipy) ────────────────────────────────────
        if model_name == "Gaussian Process (numpy/scipy)":
            gpr = NumpyGPR().fit(X_train.ravel(), y_train)
            y_pred   = gpr.predict(X_train.ravel())
            train_r2 = float(self._r2(y_train, y_pred))
            rmse     = float(np.sqrt(np.mean((y_train - y_pred) ** 2)))
            cv_r2    = float(gpr.loo_r2(X_train.ravel(), y_train))

            def mean_fn(Xnew, _g=gpr):
                return _g.predict(np.asarray(Xnew).ravel())

            def std_fn(Xnew, _g=gpr):
                _, s = _g.predict(np.asarray(Xnew).ravel(), return_std=True)
                return s

            return {"mean_fn": mean_fn, "std_fn": std_fn,
                    "train_r2": train_r2, "cv_r2": cv_r2, "rmse": rmse}

        # ── sklearn models ────────────────────────────────────────────────────
        if not _SKLEARN:
            raise RuntimeError("sklearn not installed")

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)

        if "sklearn RBF" in model_name:
            kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                      * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0))
                      + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-10, 1.0)))
            model  = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                               n_restarts_optimizer=8, alpha=0.0)
            model.fit(Xs, y_train)
            y_pred  = model.predict(Xs)
            train_r2 = float(self._r2(y_train, y_pred))
            rmse    = float(np.sqrt(np.mean((y_train - y_pred) ** 2)))
            cv_scores = cross_val_score(
                GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3),
                Xs, y_train, cv=min(cv_folds, n), scoring="r2"
            )
            cv_r2 = float(np.mean(cv_scores))

            def mean_fn(Xnew, _m=model, _sc=scaler):
                Xs2 = _sc.transform(np.asarray(Xnew).reshape(-1, 1))
                return _m.predict(Xs2)

            def std_fn(Xnew, _m=model, _sc=scaler):
                Xs2 = _sc.transform(np.asarray(Xnew).reshape(-1, 1))
                _, s = _m.predict(Xs2, return_std=True)
                return s

        elif "Random Forest" in model_name:
            model = RandomForestRegressor(n_estimators=200, random_state=42,
                                          min_samples_leaf=1)
            model.fit(Xs, y_train)
            y_pred    = model.predict(Xs)
            train_r2  = float(self._r2(y_train, y_pred))
            rmse      = float(np.sqrt(np.mean((y_train - y_pred) ** 2)))
            cv_scores = cross_val_score(model, Xs, y_train,
                                        cv=min(cv_folds, n), scoring="r2")
            cv_r2 = float(np.mean(cv_scores))

            def mean_fn(Xnew, _m=model, _sc=scaler):
                Xs2 = _sc.transform(np.asarray(Xnew).reshape(-1, 1))
                return _m.predict(Xs2)

            def std_fn(Xnew, _m=model, _sc=scaler):
                Xs2 = _sc.transform(np.asarray(Xnew).reshape(-1, 1))
                preds = np.array([t.predict(Xs2) for t in _m.estimators_])
                return preds.std(axis=0)

        elif "Gradient Boosting" in model_name:
            model = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                              learning_rate=0.1, random_state=42)
            model.fit(Xs, y_train)
            y_pred    = model.predict(Xs)
            train_r2  = float(self._r2(y_train, y_pred))
            rmse      = float(np.sqrt(np.mean((y_train - y_pred) ** 2)))
            cv_scores = cross_val_score(model, Xs, y_train,
                                        cv=min(cv_folds, n), scoring="r2")
            cv_r2     = float(np.mean(cv_scores))
            sigma     = float(np.std(y_train - y_pred)) or rmse

            def mean_fn(Xnew, _m=model, _sc=scaler):
                Xs2 = _sc.transform(np.asarray(Xnew).reshape(-1, 1))
                return _m.predict(Xs2)

            def std_fn(Xnew, _s=sigma):
                return np.full(len(np.asarray(Xnew).ravel()), _s)

        else:
            raise ValueError(f"Unknown model: {model_name}")

        return {"mean_fn": mean_fn, "std_fn": std_fn,
                "train_r2": train_r2, "cv_r2": cv_r2, "rmse": rmse}

    @staticmethod
    def _r2(y_true, y_pred):
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    # ── Fit plot (Predict tab) ────────────────────────────────────────────────

    def _update_pred_plot(self):
        if not _MPL or not self._trained:
            return
        col = self._pred_plot_col.currentText()
        if col not in self._trained:
            return

        self._pred_fig.clear()
        ax = self._pred_fig.subplots(1, 1)

        t    = self._trained[col]
        mono = self._train_mono
        y    = self._train_data.get(col, np.array([]))

        e_min, e_max = mono.min(), mono.max()
        margin = (e_max - e_min) * 0.15 or 1.0
        xs = np.linspace(e_min - margin, e_max + margin, 400)

        y_mean = t["mean_fn"](xs)
        y_std  = t["std_fn"](xs)

        ax.fill_between(xs, y_mean - 2 * y_std, y_mean + 2 * y_std,
                        alpha=0.18, color="#1976d2", label="±2σ")
        ax.fill_between(xs, y_mean - y_std, y_mean + y_std,
                        alpha=0.35, color="#1976d2", label="±1σ")
        ax.plot(xs, y_mean, "-", lw=2, color="#1976d2", label="Predicted mean")
        ax.plot(mono, y, "o", ms=7, color="#e65100",
                zorder=5, label="Training points")

        # Mark extrapolation zone
        for xv in [e_min, e_max]:
            ax.axvline(xv, color="#888", lw=1, ls="--", alpha=0.5)

        # Mark user-entered prediction energies
        for r in range(self._pred_table.rowCount()):
            it = self._pred_table.item(r, 0)
            try:
                e_new = float(it.text().strip())
                ax.axvline(e_new, color="#ffa726", lw=1.5, ls=":", alpha=0.8)
            except (ValueError, AttributeError):
                pass

        ax.set_xlabel("MonoE (keV)", fontsize=9)
        ax.set_ylabel(col, fontsize=9)
        ax.set_title(f"{col} — {self._model_cb.currentText()}", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.18)
        ax.legend(fontsize=8, loc="best")

        self._pred_fig.tight_layout()
        self._pred_canvas.draw()

    # ── Prediction table ──────────────────────────────────────────────────────

    def _rebuild_pred_table_cols(self, sel_cols):
        """Rebuild prediction table columns after training."""
        self._pred_table.setColumnCount(0)
        self._pred_table.setRowCount(0)

        # Columns: MonoE | (col_pred, col_±σ) for each target | Harmonic | UndE
        headers = ["MonoE (keV)"]
        for col in sel_cols:
            headers += [f"{col}", f"±σ ({col})"]
        headers += ["Harmonic", "UndE (eV)"]
        self._pred_table.setColumnCount(len(headers))
        self._pred_table.setHorizontalHeaderLabels(headers)
        self._pred_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)

        # Add one blank row to get started
        self._add_pred_row()

    def _col_indices(self):
        """Return {col_name: (pred_col_idx, std_col_idx), ..., 'Harmonic': idx, 'UndE': idx}."""
        h = self._pred_table.columnCount()
        if h == 0:
            return {}
        headers = [self._pred_table.horizontalHeaderItem(c).text()
                   for c in range(h)]
        result = {}
        for i, hdr in enumerate(headers):
            if hdr == "Harmonic":
                result["__harmonic__"] = i
            elif hdr == "UndE (eV)":
                result["__und_e__"] = i
            elif hdr.startswith("±σ ("):
                col = hdr[4:-1]
                if col in result:
                    result[col] = (result[col][0], i)
            elif hdr != "MonoE (keV)" and not hdr.startswith("±"):
                result[hdr] = (i, None)
        return result

    def _add_pred_row(self):
        self._pred_table.blockSignals(True)
        r = self._pred_table.rowCount()
        self._pred_table.insertRow(r)
        h = self._pred_table.columnCount()
        headers = [self._pred_table.horizontalHeaderItem(c).text()
                   for c in range(h)] if h > 0 else ["MonoE (keV)"]
        for c in range(max(1, h)):
            item = QTableWidgetItem("")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            hdr = headers[c] if c < len(headers) else ""
            if hdr in ("Harmonic", "UndE (eV)"):
                item.setBackground(_AMBER_BG)
                item.setForeground(_AMBER_FG)
            elif hdr.startswith("±σ") or (c > 0 and not hdr.startswith("MonoE")
                                           and hdr not in ("Harmonic", "UndE (eV)")):
                # predicted value cells — read-only
                if not hdr.startswith("MonoE") and hdr not in ("Harmonic", "UndE (eV)"):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    item.setForeground(QColor("#aaaaaa"))
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
            self._predict_row(row)
            if _MPL and self._trained:
                self._update_pred_plot()

    def _recompute_predictions(self):
        for r in range(self._pred_table.rowCount()):
            self._predict_row(r)

    def _predict_row(self, row):
        if not self._trained:
            return
        it = self._pred_table.item(row, 0)
        try:
            e = float(it.text().strip())
        except (ValueError, AttributeError):
            return

        outside = (len(self._train_mono) > 0
                   and (e < self._train_mono.min() or e > self._train_mono.max()))

        col_idx = self._col_indices()
        self._pred_table.blockSignals(True)
        for col, info in col_idx.items():
            if col.startswith("__"):
                continue
            if col not in self._trained:
                continue
            t = self._trained[col]
            pred = float(t["mean_fn"](np.array([e]))[0])
            std  = float(t["std_fn"](np.array([e]))[0])
            pred_idx = info[0]
            std_idx  = info[1]
            for c_idx, val in [(pred_idx, f"{pred:.6g}"), (std_idx, f"{std:.4g}" if std_idx else None)]:
                if c_idx is None or val is None:
                    continue
                it2 = self._pred_table.item(row, c_idx)
                if it2 is None:
                    it2 = QTableWidgetItem()
                    it2.setFlags(it2.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    it2.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self._pred_table.setItem(row, c_idx, it2)
                it2.setText(val)
                if outside:
                    it2.setBackground(_EXTRAP_BG)
                    it2.setForeground(_EXTRAP_FG)
                else:
                    it2.setBackground(QTableWidgetItem().background())
                    it2.setForeground(QColor("#aaaaaa"))
        self._pred_table.blockSignals(False)

    # ── Send to energy table ──────────────────────────────────────────────────

    def _send_to_energy_table(self):
        if self._energy_tab is None:
            QMessageBox.warning(self, "Not connected",
                                "Open this dialog from the Alignment tab to enable "
                                "sending predictions to the Energy Table.")
            return

        col_idx = self._col_indices()
        roll2_col = col_idx.get("Roll2", (None, None))[0] if "Roll2" in col_idx else None
        x2_col    = col_idx.get("X2",    (None, None))[0] if "X2"    in col_idx else None
        harm_col  = col_idx.get("__harmonic__")
        und_e_col = col_idx.get("__und_e__")

        added = 0
        for r in range(self._pred_table.rowCount()):
            def _cell(c):
                it = self._pred_table.item(r, c) if c is not None else None
                try:
                    return float(it.text().strip()) if it and it.text().strip() else None
                except ValueError:
                    return None

            mono_e   = _cell(0)
            if mono_e is None:
                continue
            harmonic = _cell(harm_col)
            und_e    = _cell(und_e_col)
            roll2    = _cell(roll2_col)
            x2       = _cell(x2_col)

            # [MonoE, Harmonic, UndE, Roll2, X2]
            row_vals = [
                mono_e,
                harmonic if harmonic is not None else 0.0,
                und_e    if und_e    is not None else 0.0,
                roll2    if roll2    is not None else 0.0,
                x2       if x2       is not None else 0.0,
            ]
            self._energy_tab._append_row(row_vals)
            added += 1

        if added:
            QMessageBox.information(
                self, "Sent",
                f"{added} row(s) added to the Energy Table.\n"
                "Check Harmonic and UndE values (amber cells were entered manually)."
            )
        else:
            QMessageBox.warning(self, "Nothing to send",
                                "No rows with a valid MonoE value found.")

    # ── Report ────────────────────────────────────────────────────────────────

    _CSS = """
    body { font-family: Arial, sans-serif; font-size: 13px;
           margin: 16px 24px; color: #212121; }
    h2   { color: #1565c0; border-bottom: 2px solid #1565c0;
           padding-bottom: 4px; margin-top: 8px; }
    h3   { color: #37474f; margin-top: 22px; margin-bottom: 6px; }
    p    { margin: 4px 0 10px 0; }
    small { color: #777; }
    table { border-collapse: collapse; margin: 6px 0 12px 0; }
    th    { background: #e3f2fd; color: #1565c0; padding: 6px 14px;
            border: 1px solid #90caf9; text-align: center; white-space: nowrap; }
    td    { padding: 5px 12px; border: 1px solid #ccc;
            text-align: right; white-space: nowrap; }
    td.lbl { text-align: left; font-weight: 600; }
    tr:nth-child(even) td { background: #fafafa; }
    .ok   { color: #2e7d32; font-weight: 600; }
    .warn { color: #e65100; font-weight: 600; }
    """

    def _make_report(self, sel_cols):
        d   = self._data
        n   = self._n_rows
        nc  = self._num_cols
        mono = d.get("MonoE", np.array([]))
        ue   = np.unique(mono) if len(mono) else np.array([])
        grp_cols = [c for c in sel_cols if c in d]

        def _fmt(v, p=6):
            try:
                return f"{float(v):.{p}g}"
            except Exception:
                return "n/a"

        parts = [f"<html><head><style>{self._CSS}</style></head><body>"]
        parts.append("<h2>Statistical Analysis Report</h2>")
        parts.append(
            f"<p><b>File:</b> {self._csv_path}<br>"
            f"<b>Rows:</b> {n} &nbsp;&nbsp; <b>Numeric columns:</b> {len(nc)}<br>"
            f"<b>Selected for analysis:</b> {', '.join(grp_cols) or '(none)'}</p>"
        )

        # 1. Descriptive stats
        parts.append("<h3>1. Descriptive Statistics (all numeric columns)</h3>")
        hdr = ("<tr><th>Column</th><th>N</th><th>Mean</th><th>Std Dev</th>"
               "<th>Min</th><th>25 %</th><th>Median</th><th>75 %</th><th>Max</th></tr>")
        rows = []
        for k in nc:
            v = d[k]
            p25, p50, p75 = np.percentile(v, [25, 50, 75])
            rows.append(
                f"<tr><td class='lbl'>{k}</td><td>{len(v)}</td>"
                f"<td>{_fmt(v.mean())}</td><td>{_fmt(v.std(),4)}</td>"
                f"<td>{_fmt(v.min())}</td><td>{_fmt(p25)}</td>"
                f"<td>{_fmt(p50)}</td><td>{_fmt(p75)}</td>"
                f"<td>{_fmt(v.max())}</td></tr>"
            )
        parts.append(f"<table>{hdr}{''.join(rows)}</table>")

        # 2. Per-energy group stats
        if len(ue) and len(mono) == n and grp_cols:
            parts.append(
                f"<h3>2. Per-Energy Group Statistics "
                f"<small>({', '.join(grp_cols)})</small></h3>"
            )
            hdr2 = "<tr><th>MonoE (keV)</th><th>N</th>"
            for c in grp_cols:
                hdr2 += f"<th>{c} mean</th><th>{c} std</th><th>{c} CV %</th>"
            hdr2 += "</tr>"
            rows2 = []
            for e in ue:
                mask = mono == e
                row = f"<tr><td class='lbl'>{e:.4g}</td><td>{mask.sum()}</td>"
                for c in grp_cols:
                    m  = d[c][mask].mean()
                    s  = d[c][mask].std()
                    cv = abs(s / m * 100) if m != 0 else float("nan")
                    cv_cls = "ok" if cv < 0.5 else ("warn" if cv < 2 else "")
                    cv_s   = f"{cv:.3f}" if not np.isnan(cv) else "n/a"
                    row += (f"<td>{_fmt(m)}</td><td>{_fmt(s,4)}</td>"
                            f"<td class='{cv_cls}'>{cv_s}</td>")
                row += "</tr>"
                rows2.append(row)
            parts.append(f"<table>{hdr2}{''.join(rows2)}</table>")
            parts.append(
                "<p><small>CV = coefficient of variation. "
                "<span class='ok'>Green</span> &lt; 0.5 %, "
                "<span class='warn'>orange</span> &lt; 2 %.</small></p>"
            )

        # 3. Pearson correlation
        corr_keys = [k for k in nc if k not in ("Harmonic",)]
        if len(corr_keys) >= 2:
            parts.append("<h3>3. Pearson Correlation Matrix</h3>")
            mat  = np.corrcoef(np.array([d[k] for k in corr_keys]))
            hdr3 = "<tr><th></th>" + "".join(f"<th>{k}</th>" for k in corr_keys) + "</tr>"
            rows3 = []
            for i, ki in enumerate(corr_keys):
                row = f"<tr><td class='lbl'>{ki}</td>"
                for j in range(len(corr_keys)):
                    r = mat[i, j]
                    if i == j:
                        row += "<td>1.000</td>"
                    else:
                        cls = "ok" if abs(r) > 0.9 else ("warn" if abs(r) > 0.7 else "")
                        row += f"<td class='{cls}'>{r:+.3f}</td>"
                row += "</tr>"
                rows3.append(row)
            parts.append(f"<table>{hdr3}{''.join(rows3)}</table>")

        # 4. Spearman correlation
        if _SCIPY and len(corr_keys) >= 2:
            parts.append("<h3>4. Spearman Rank Correlation Matrix</h3>")
            hdr4 = "<tr><th></th>" + "".join(f"<th>{k}</th>" for k in corr_keys) + "</tr>"
            rows4 = []
            for ki in corr_keys:
                row = f"<tr><td class='lbl'>{ki}</td>"
                for kj in corr_keys:
                    if ki == kj:
                        row += "<td>1.000</td>"
                    else:
                        r, _ = _sstats.spearmanr(d[ki], d[kj])
                        cls  = "ok" if abs(r) > 0.9 else ("warn" if abs(r) > 0.7 else "")
                        row += f"<td class='{cls}'>{r:+.3f}</td>"
                row += "</tr>"
                rows4.append(row)
            parts.append(f"<table>{hdr4}{''.join(rows4)}</table>")

        # 5. Drift analysis
        if len(ue) and len(mono) == n and grp_cols:
            parts.append(
                f"<h3>5. Drift Analysis <small>({', '.join(grp_cols)})</small></h3>"
            )
            hdr5 = ("<tr><th>MonoE (keV)</th><th>Column</th><th>N</th>"
                    "<th>Slope (unit/step)</th><th>Intercept</th><th>R²</th></tr>")
            rows5 = []
            for e in ue:
                mask   = mono == e
                ix     = np.where(mask)[0]
                if len(ix) < 3:
                    continue
                ix_rel = ix - ix[0]
                for c in grp_cols:
                    y = d[c][mask]
                    slope, intercept = np.polyfit(ix_rel, y, 1)
                    fitted = slope * ix_rel + intercept
                    ss_res = np.sum((y - fitted) ** 2)
                    ss_tot = np.sum((y - y.mean()) ** 2)
                    r2  = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
                    rel = abs(slope / y.mean()) if y.mean() != 0 else float("nan")
                    cls = "ok" if np.isnan(rel) or rel < 1e-4 else "warn"
                    rows5.append(
                        f"<tr><td class='lbl'>{e:.4g}</td><td>{c}</td>"
                        f"<td>{mask.sum()}</td>"
                        f"<td class='{cls}'>{slope:+.4g}</td>"
                        f"<td>{_fmt(intercept)}</td>"
                        f"<td>{r2:.4f}</td></tr>"
                    )
            parts.append(f"<table>{hdr5}{''.join(rows5)}</table>")
            parts.append(
                "<p><small>Slope = change per step within one energy group. "
                "<span class='ok'>Green</span> = negligible drift, "
                "<span class='warn'>orange</span> = possible drift.</small></p>"
            )

        # 6. Strongest pairwise correlations
        if len(corr_keys) >= 2:
            parts.append("<h3>6. Strongest Pairwise Correlations</h3>")
            mat   = np.corrcoef(np.array([d[k] for k in corr_keys]))
            pairs = []
            for i in range(len(corr_keys)):
                for j in range(i + 1, len(corr_keys)):
                    pairs.append((abs(mat[i, j]), mat[i, j], corr_keys[i], corr_keys[j]))
            pairs.sort(reverse=True)
            hdr6 = ("<tr><th>Column A</th><th>Column B</th>"
                    "<th>Pearson r</th><th>Interpretation</th></tr>")
            rows6 = []
            for _, r, ka, kb in pairs[:10]:
                if abs(r) > 0.9:   interp = "Very strong"
                elif abs(r) > 0.7: interp = "Strong"
                elif abs(r) > 0.5: interp = "Moderate"
                elif abs(r) > 0.3: interp = "Weak"
                else:              interp = "Negligible"
                direction = "positive" if r > 0 else "negative"
                cls = "ok" if abs(r) > 0.9 else ("warn" if abs(r) > 0.7 else "")
                rows6.append(
                    f"<tr><td class='lbl'>{ka}</td><td class='lbl'>{kb}</td>"
                    f"<td class='{cls}'>{r:+.3f}</td>"
                    f"<td>{interp} {direction}</td></tr>"
                )
            parts.append(f"<table>{hdr6}{''.join(rows6)}</table>")

        parts.append("</body></html>")
        self._report_html = "\n".join(parts)
        self._report.setHtml(self._report_html)

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_report(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Report", "alignment_statistics.html",
            "HTML files (*.html);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._report_html)
            QMessageBox.information(self, "Export", f"Report saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export", f"Could not save:\n{e}")
