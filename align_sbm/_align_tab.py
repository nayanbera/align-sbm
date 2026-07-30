"""Alignment runner tab — controls, live log, and scan plot."""
import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QGroupBox, QCheckBox, QPushButton, QLabel,
    QProgressBar, QPlainTextEdit, QListWidget, QListWidgetItem,
)
from .smart_scan_functions import ScanStatus

try:
    import pyqtgraph as pg
    _PG = True
except ImportError:
    _PG = False


class AlignTab(QWidget):
    status_message = pyqtSignal(str)

    def __init__(self, setup_tab, energy_tab, parent=None):
        super().__init__(parent)
        self._setup_tab = setup_tab
        self._energy_tab = energy_tab
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

    # ── Left: controls panel ─────────────────────────────────────────────────

    def _build_controls(self):
        panel = QWidget()
        vbox = QVBoxLayout(panel)
        vbox.setSpacing(8)

        # Mode
        mode_grp = QGroupBox("Mode")
        mv = QVBoxLayout(mode_grp)
        self._sim_cb = QCheckBox("Simulation (no EPICS)")
        self._sim_cb.setChecked(True)
        self._sim_cb.setToolTip(
            "When checked, all scans run in simulation mode — no EPICS hardware needed."
        )
        mv.addWidget(self._sim_cb)
        vbox.addWidget(mode_grp)

        # Energy rows
        energy_grp = QGroupBox("Energy rows to align")
        ev = QVBoxLayout(energy_grp)
        self._row_list = QListWidget()
        self._row_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self._row_list.setMaximumHeight(180)
        ev.addWidget(self._row_list)
        row_btns = QHBoxLayout()
        sel_all = QPushButton("All")
        sel_all.clicked.connect(self._select_all_rows)
        sel_none = QPushButton("None")
        sel_none.clicked.connect(self._select_no_rows)
        sel_refresh = QPushButton("Refresh")
        sel_refresh.clicked.connect(self._refresh_row_list)
        sel_refresh.setToolTip("Reload from Energy Table tab")
        row_btns.addWidget(sel_all)
        row_btns.addWidget(sel_none)
        row_btns.addWidget(sel_refresh)
        ev.addLayout(row_btns)
        vbox.addWidget(energy_grp)

        # Run buttons
        run_grp = QGroupBox("Run")
        rv = QVBoxLayout(run_grp)
        self._start_btn = QPushButton("Start Alignment")
        self._start_btn.setMinimumHeight(36)
        self._start_btn.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; font-weight: bold; border-radius: 4px; }"
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

        # Demo scan (shows plot widget without running full alignment)
        demo_btn = QPushButton("Demo Scan (sim)")
        demo_btn.setToolTip("Run a single simulated scan and display it in the plot")
        demo_btn.clicked.connect(self._demo_scan)
        rv.addWidget(demo_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        rv.addWidget(self._progress)

        self._status_lbl = QLabel("Ready")
        self._status_lbl.setWordWrap(True)
        rv.addWidget(self._status_lbl)
        vbox.addWidget(run_grp)

        vbox.addStretch()

        # Populate energy rows
        self._refresh_row_list()
        return panel

    def _refresh_row_list(self):
        self._row_list.clear()
        for i, label in enumerate(self._energy_tab.get_row_labels()):
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, i)
            self._row_list.addItem(item)
        self._select_all_rows()

    def _select_all_rows(self):
        for i in range(self._row_list.count()):
            self._row_list.item(i).setSelected(True)

    def _select_no_rows(self):
        self._row_list.clearSelection()

    # ── Right: log + plot ────────────────────────────────────────────────────

    def _build_right_panel(self):
        panel = QWidget()
        vbox = QVBoxLayout(panel)
        vbox.setSpacing(4)

        right_split = QSplitter(Qt.Orientation.Vertical)

        # Plot
        if _PG:
            self._plot_widget = pg.PlotWidget(title="Last Scan")
            self._plot_widget.setLabel("bottom", "Position")
            self._plot_widget.setLabel("left", "Signal")
            self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self._plot_widget.setMinimumHeight(200)
            right_split.addWidget(self._plot_widget)
            self._data_item = self._plot_widget.plot(
                [], [], pen=None,
                symbol="o", symbolSize=7,
                symbolBrush=pg.mkBrush("#4fc3f7"),
                symbolPen=pg.mkPen("#4fc3f7"),
            )
            self._fit_item = self._plot_widget.plot(
                [], [], pen=pg.mkPen("#ef5350", width=2)
            )
            self._peak_line = pg.InfiniteLine(
                angle=90, movable=False,
                pen=pg.mkPen("#ffa726", width=1, style=Qt.PenStyle.DashLine)
            )
            self._plot_widget.addItem(self._peak_line)
        else:
            placeholder = QLabel("pyqtgraph not installed — install it for live scan plots")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: #888;")
            right_split.addWidget(placeholder)
            self._data_item = None
            self._fit_item = None
            self._peak_line = None

        # Log
        log_frame = QWidget()
        lv = QVBoxLayout(log_frame)
        lv.setContentsMargins(0, 0, 0, 0)
        log_hdr = QHBoxLayout()
        log_hdr.addWidget(QLabel("<b>Alignment Log</b>"))
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
        right_split.addWidget(log_frame)

        right_split.setStretchFactor(0, 1)
        right_split.setStretchFactor(1, 2)
        vbox.addWidget(right_split)
        return panel

    def _clear_log(self):
        self._log.clear()

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
        kwargs = self._setup_tab.get_kwargs()

        self._log.appendPlainText(
            f"\n{'═'*60}\n"
            f"  Starting alignment  {'[SIMULATION]' if simulate else '[EPICS]'}\n"
            f"  {len(rows)} energy row(s) selected\n"
            f"{'═'*60}"
        )

        from ._worker import AlignWorker
        self._worker = AlignWorker(rows, kwargs, simulate, parent=self)
        self._worker.log_chunk.connect(self._on_log)
        self._worker.scan_finished.connect(self._on_scan_finished)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)

        self._start_btn.setEnabled(False)
        self._abort_btn.setEnabled(True)
        self._progress.setVisible(True)
        self._status_lbl.setText("Running…")
        self.status_message.emit("Alignment running…")

        self._worker.start()

    def _abort_alignment(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._log.appendPlainText("\n⚠ Alignment aborted by user.")
            self._status_lbl.setText("Aborted")
            self.status_message.emit("Alignment aborted")
        self._reset_buttons()

    def _reset_buttons(self):
        self._start_btn.setEnabled(True)
        self._abort_btn.setEnabled(False)
        self._progress.setVisible(False)

    # ── Worker signal handlers ────────────────────────────────────────────────

    def _on_log(self, text):
        # Append without extra newline (text already has \n)
        cursor = self._log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self._log.setTextCursor(cursor)
        self._log.ensureCursorVisible()

    def _on_scan_finished(self, result):
        if not _PG or self._data_item is None:
            return
        if result.status != ScanStatus.SUCCESS:
            return
        self._update_plot(result)

    def _update_plot(self, result):
        if not _PG or self._data_item is None:
            return
        pos = np.asarray(result.positions, dtype=float)
        sig = np.asarray(result.signals, dtype=float)
        self._data_item.setData(pos, sig)

        if result.center is not None and result.sigma is not None and result.amplitude is not None:
            xs = np.linspace(pos.min(), pos.max(), 300)
            offset = result.offset or 0.0
            if result.profile == "lorentzian":
                gamma = result.sigma * np.sqrt(2 * np.log(2))
                ys = result.amplitude / (1 + ((xs - result.center) / gamma) ** 2) + offset
            else:
                ys = result.amplitude * np.exp(
                    -0.5 * ((xs - result.center) / result.sigma) ** 2
                ) + offset
            self._fit_item.setData(xs, ys)
            self._peak_line.setValue(result.center)
        else:
            self._fit_item.setData([], [])

    def _demo_scan(self):
        from .smart_scan_functions import smart_scan
        kwargs = self._setup_tab.get_kwargs()
        center = (kwargs.get("brg2_start", -0.005) + kwargs.get("brg2_stop", 0.005)) / 2
        half = abs(kwargs.get("brg2_stop", 0.005) - kwargs.get("brg2_start", -0.005)) / 2
        result = smart_scan(
            "demo_motor", "demo_det",
            start=center - half, stop=center + half,
            nsteps=kwargs.get("brg2_nsteps", 21),
            simulate=True,
            sim_center=center, sim_sigma=half * 0.3,
            sim_amplitude=1000, sim_noise=15,
            plot=False, fine_scan=True, move_to_peak=False,
        )
        if result.status == ScanStatus.SUCCESS:
            self._update_plot(result)
            self._log.appendPlainText(
                f"[Demo] smart_scan: status={result.status.value}, "
                f"center={result.center:.5g}, sigma={result.sigma:.5g}"
            )

    def _on_done(self, results):
        self._log.appendPlainText(
            f"\n{'═'*60}\n  Alignment complete — {len(results)} row(s) processed.\n{'═'*60}"
        )
        self._status_lbl.setText(f"Done ({len(results)} rows)")
        self.status_message.emit(f"Alignment complete: {len(results)} rows")
        self._reset_buttons()

    def _on_error(self, tb):
        self._log.appendPlainText(f"\n✗ ERROR:\n{tb}")
        self._status_lbl.setText("Error — see log")
        self.status_message.emit("Alignment error — see log")
        self._reset_buttons()
