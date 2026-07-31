"""Statistical analysis dialog for alignment CSV data."""
import csv
import os

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTabWidget, QWidget, QTextBrowser, QFileDialog, QMessageBox,
    QListWidget, QListWidgetItem, QGroupBox, QComboBox, QSizePolicy,
    QScrollArea,
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

# Columns that are grouping variables, not analysis targets
_GROUPING_COLS = {"MonoE", "Harmonic"}
# Columns that are always checked by default
_DEFAULT_CHECKED = {"Roll2", "X2"}


class StatsDialog(QDialog):
    """Load an alignment CSV and show plots + statistical report.

    All numeric columns are shown in the column selector. Roll2 and X2 are
    pre-selected; extra record PVs appear unchecked (user selects them).
    Plots and the per-energy / drift report sections cover every selected column.
    """

    def __init__(self, csv_path="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("CSV Statistical Analysis")
        self.resize(1200, 900)

        self._csv_path    = csv_path
        self._data        = {}      # col → np.ndarray
        self._num_cols    = []      # all numeric columns in the file
        self._n_rows      = 0
        self._report_html = ""

        self._build_ui()
        if csv_path and os.path.isfile(csv_path):
            self._load_csv(csv_path)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # ── CSV source row ────────────────────────────────────────────────────
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

        # ── Column selector + scatter axis pickers ────────────────────────────
        col_grp = QGroupBox("Columns to analyze")
        col_grp.setMaximumHeight(145)
        col_h = QHBoxLayout(col_grp)
        col_h.setSpacing(6)

        self._col_list = QListWidget()
        self._col_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self._col_list.setFlow(QListWidget.Flow.LeftToRight)
        self._col_list.setWrapping(True)
        self._col_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._col_list.setSpacing(2)
        self._col_list.setStyleSheet("QListWidget { border: none; }")
        col_h.addWidget(self._col_list, 1)

        # Scatter axis selectors
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
        refresh_btn.setToolTip("Recompute plots and report with current column selection")
        refresh_btn.clicked.connect(self._refresh)
        scatter_v.addWidget(refresh_btn)
        col_h.addLayout(scatter_v)

        layout.addWidget(col_grp)

        # ── Main content tabs ─────────────────────────────────────────────────
        self._tabs = QTabWidget()

        if _MPL:
            self._plot_container = QWidget()
            pv = QVBoxLayout(self._plot_container)
            pv.setContentsMargins(0, 0, 0, 0)
            pv.setSpacing(0)
            # Canvas and toolbar are created/replaced in _make_plots()
            self._canvas  = None
            self._toolbar = None
            self._tabs.addTab(self._plot_container, "Plots")
        else:
            lbl = QLabel("matplotlib not installed — plots unavailable")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #888; font-style: italic;")
            self._tabs.addTab(lbl, "Plots")

        self._report = QTextBrowser()
        self._report.setOpenExternalLinks(False)
        self._tabs.addTab(self._report, "Report")

        layout.addWidget(self._tabs, 1)

        # ── Bottom buttons ────────────────────────────────────────────────────
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

        # Parse numeric columns; skip datetime
        raw = {k: [r.get(k, "").strip() for r in rows] for k in fieldnames}
        self._data     = {}
        self._num_cols = []
        for k, vals in raw.items():
            if k.lower() == "datetime":
                continue
            try:
                self._data[k] = np.array([float(v) for v in vals])
                self._num_cols.append(k)
            except (ValueError, TypeError):
                pass

        # Info label
        mono     = self._data.get("MonoE", np.array([]))
        unique_e = np.unique(mono) if len(mono) else np.array([])
        e_str    = ", ".join(f"{e:.4g}" for e in unique_e[:8])
        if len(unique_e) > 8:
            e_str += f" … ({len(unique_e)} total)"
        loop_str = ""
        if len(unique_e):
            cnts = [int(np.sum(mono == e)) for e in unique_e]
            loop_str = (
                f" • {cnts[0]} measurement(s) per energy"
                if len(set(cnts)) == 1
                else f" • {min(cnts)}–{max(cnts)} measurements per energy"
            )
        self._info_lbl.setText(
            f"{self._n_rows} rows  •  {len(self._num_cols)} numeric columns"
            + (f"  •  MonoE: {e_str} keV" if e_str else "")
            + loop_str
        )

        self._rebuild_col_selector()
        self._refresh()
        self._export_btn.setEnabled(True)

    # ── Column selector ───────────────────────────────────────────────────────

    def _rebuild_col_selector(self):
        """Populate the column checklist from the loaded numeric columns."""
        # Remember previously checked names to survive a re-load
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
                checked = False   # keep previous selection when re-loading same file
            else:
                checked = col in _DEFAULT_CHECKED
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            self._col_list.addItem(item)

        # Scatter axis selectors
        self._scatter_x.blockSignals(True)
        self._scatter_y.blockSignals(True)
        prev_x = self._scatter_x.currentText()
        prev_y = self._scatter_y.currentText()
        self._scatter_x.clear()
        self._scatter_y.clear()
        self._scatter_x.addItems(analysis_cols)
        self._scatter_y.addItems(analysis_cols)
        # Restore or default
        xi = self._scatter_x.findText(prev_x or "Roll2")
        yi = self._scatter_y.findText(prev_y or "X2")
        self._scatter_x.setCurrentIndex(max(0, xi))
        self._scatter_y.setCurrentIndex(max(0, yi) if yi >= 0 else min(1, self._scatter_y.count() - 1))
        self._scatter_x.blockSignals(False)
        self._scatter_y.blockSignals(False)

    def _selected_cols(self):
        """Return the list of currently checked analysis columns."""
        return [
            self._col_list.item(i).text()
            for i in range(self._col_list.count())
            if self._col_list.item(i).checkState() == Qt.CheckState.Checked
        ]

    # ── Refresh (plots + report) ──────────────────────────────────────────────

    def _refresh(self):
        if not self._data:
            return
        sel = self._selected_cols()
        if _MPL:
            self._make_plots(sel)
        self._make_report(sel)

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _make_plots(self, sel_cols):
        """Build a dynamic figure: 2 subplots per selected column + correlation + scatter."""
        # Remove old canvas/toolbar
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

        # Layout: N rows (one per selected column) + 1 bottom row for heatmap+scatter
        n_sel  = len(sel_cols)
        n_rows = max(n_sel, 1) + 1   # at least 1 data row + 1 bottom row
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
            axes = np.array([axes])   # ensure 2-D

        def _setup(ax, title, xlabel, ylabel):
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.18)

        # ── One row per selected column ───────────────────────────────────────
        for row_i, col in enumerate(sel_cols):
            ax_ts  = axes[row_i, 0]   # time-series
            ax_bar = axes[row_i, 1]   # mean ± std per energy

            if col in d:
                # Time-series, coloured by MonoE
                for e in ue:
                    mask = mono == e
                    ax_ts.plot(idx[mask], d[col][mask], "o-", ms=4, lw=1,
                               color=color_of[e], label=f"{e:.4g} keV")
                if len(ue) <= 8:
                    ax_ts.legend(fontsize=6, ncol=2, loc="best")
                _setup(ax_ts, f"{col} vs measurement index", "Index", col)

                # Bar chart
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

        # ── Bottom row: correlation heatmap (left) + scatter (right) ─────────
        ax_corr   = axes[n_rows - 1, 0]
        ax_scatter = axes[n_rows - 1, 1]

        corr_keys = [k for k in self._num_cols if k not in ("Harmonic",)]
        if len(corr_keys) >= 2:
            mat = np.corrcoef(np.array([d[k] for k in corr_keys]))
            im  = fig.colorbar(
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

        # Scatter — user-chosen X vs Y
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
        elif sx in d and sy in d:
            # Same column — just show histogram
            ax_scatter.hist(d[sx], bins=20, alpha=0.75)
            sy = f"Histogram of {sx}"
        _setup(ax_scatter, f"{sx} vs {sy}", sx, sy)

        canvas.draw()

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

        def _fmt(v, p=6):
            try:
                return f"{float(v):.{p}g}"
            except Exception:
                return "n/a"

        # Analysis columns for per-energy and drift sections
        grp_cols = [c for c in sel_cols if c in d]

        parts = [f"<html><head><style>{self._CSS}</style></head><body>"]
        parts.append("<h2>Statistical Analysis Report</h2>")
        parts.append(
            f"<p><b>File:</b> {self._csv_path}<br>"
            f"<b>Rows:</b> {n} &nbsp;&nbsp; <b>Numeric columns:</b> {len(nc)}<br>"
            f"<b>Selected for analysis:</b> {', '.join(grp_cols) or '(none)'}</p>"
        )

        # ── 1. Descriptive statistics (all numeric) ───────────────────────────
        parts.append("<h3>1. Descriptive Statistics (all numeric columns)</h3>")
        hdr = ("<tr><th>Column</th><th>N</th><th>Mean</th><th>Std Dev</th>"
               "<th>Min</th><th>25 %</th><th>Median</th><th>75 %</th><th>Max</th></tr>")
        rows = []
        for k in nc:
            v = d[k]
            p25, p50, p75 = np.percentile(v, [25, 50, 75])
            rows.append(
                f"<tr><td class='lbl'>{k}</td><td>{len(v)}</td>"
                f"<td>{_fmt(v.mean())}</td><td>{_fmt(v.std(), 4)}</td>"
                f"<td>{_fmt(v.min())}</td><td>{_fmt(p25)}</td>"
                f"<td>{_fmt(p50)}</td><td>{_fmt(p75)}</td>"
                f"<td>{_fmt(v.max())}</td></tr>"
            )
        parts.append(f"<table>{hdr}{''.join(rows)}</table>")

        # ── 2. Per-energy group statistics (selected columns) ─────────────────
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
                    row += (f"<td>{_fmt(m)}</td><td>{_fmt(s, 4)}</td>"
                            f"<td class='{cv_cls}'>{cv_s}</td>")
                row += "</tr>"
                rows2.append(row)
            parts.append(f"<table>{hdr2}{''.join(rows2)}</table>")
            parts.append(
                "<p><small>CV = coefficient of variation (std / |mean| × 100 %). "
                "<span class='ok'>Green</span> &lt; 0.5 %, "
                "<span class='warn'>orange</span> &lt; 2 %.</small></p>"
            )

        # ── 3. Pearson correlation matrix ─────────────────────────────────────
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

        # ── 4. Spearman correlation matrix ────────────────────────────────────
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

        # ── 5. Drift analysis (selected columns) ──────────────────────────────
        if len(ue) and len(mono) == n and grp_cols:
            parts.append(
                f"<h3>5. Drift Analysis — linear trend vs measurement index "
                f"<small>({', '.join(grp_cols)})</small></h3>"
            )
            hdr5 = ("<tr><th>MonoE (keV)</th><th>Column</th><th>N</th>"
                    "<th>Slope (unit/step)</th><th>Intercept</th><th>R²</th></tr>")
            rows5 = []
            for e in ue:
                mask  = mono == e
                ix    = np.where(mask)[0]
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
                "<p><small>Slope is the change per step within the same energy group. "
                "<span class='ok'>Green</span> = negligible drift (&lt; 0.01 % / step), "
                "<span class='warn'>orange</span> = possible drift.</small></p>"
            )

        # ── 6. Strongest pairwise correlations ────────────────────────────────
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
