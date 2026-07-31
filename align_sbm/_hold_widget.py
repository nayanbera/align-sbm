"""Hold Conditions panel — suspends alignment when EPICS PV conditions fail."""
import json

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

_OPS = [">", "<", ">=", "<=", "==", "!="]


class HoldConditionsWidget(QGroupBox):
    """
    Checkable group box listing PV conditions that suspend the alignment.
    When the group box is checked, hold monitoring is active.
    """

    config_changed = pyqtSignal()

    def __init__(self, settings, parent=None):
        super().__init__("Hold Conditions")
        self.setCheckable(True)
        self.setChecked(False)
        self._settings = settings
        self._build_ui()
        self._load_settings()

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(6000)   # refresh status every 6 s
        self._poll_timer.timeout.connect(self._update_status)
        self._poll_timer.start()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        vbox = QVBoxLayout(self)
        vbox.setSpacing(4)
        vbox.setContentsMargins(6, 4, 6, 6)

        # Logic selector
        logic_row = QHBoxLayout()
        logic_row.addWidget(QLabel("Suspend when:"))
        self._logic_cb = QComboBox()
        self._logic_cb.addItems(["any condition is met", "all conditions are met"])
        self._logic_cb.setToolTip(
            "'any' — hold when at least one condition evaluates to True (default)\n"
            "'all' — hold only when every condition is True simultaneously\n\n"
            "Example: 'SR:Current < 10' suspends when current drops below 10."
        )
        self._logic_cb.currentIndexChanged.connect(self.config_changed)
        logic_row.addWidget(self._logic_cb)
        logic_row.addStretch()
        vbox.addLayout(logic_row)

        # Conditions table
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["PV Name", "Op", "Value", "On", "●"])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col, w in [(1, 52), (2, 88), (3, 28), (4, 22)]:
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            self._table.setColumnWidth(col, w)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(24)
        self._table.setMaximumHeight(150)
        self._table.setMinimumHeight(50)
        self._table.setAlternatingRowColors(True)
        self._table.itemChanged.connect(lambda _: self.config_changed.emit())
        vbox.addWidget(self._table)

        # Add / Remove buttons
        btn_row = QHBoxLayout()
        for label, slot in [("+ Add", self._add_row), ("− Remove", self._remove_row)]:
            b = QPushButton(label)
            b.setMaximumWidth(70)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        btn_row.addStretch()
        vbox.addLayout(btn_row)

        # Overall status label
        self._status_lbl = QLabel("Not checked")
        self._status_lbl.setStyleSheet("font-size: 11px; color: #888;")
        self._status_lbl.setWordWrap(True)
        vbox.addWidget(self._status_lbl)

    # ── Row management ────────────────────────────────────────────────────────

    def _add_row(self, pv="", op=">", value="", enabled=True):
        self._table.blockSignals(True)
        r = self._table.rowCount()
        self._table.insertRow(r)

        pv_item = QTableWidgetItem(pv)
        pv_item.setToolTip("EPICS PV name")
        self._table.setItem(r, 0, pv_item)

        op_cb = QComboBox()
        op_cb.addItems(_OPS)
        op_cb.setCurrentText(op)
        op_cb.currentIndexChanged.connect(lambda: self.config_changed.emit())
        self._table.setCellWidget(r, 1, op_cb)

        val_item = QTableWidgetItem(str(value))
        val_item.setToolTip("Threshold — number or quoted/unquoted string")
        self._table.setItem(r, 2, val_item)

        chk = QCheckBox()
        chk.setChecked(enabled)
        chk.stateChanged.connect(lambda: self.config_changed.emit())
        chk_wrap = QWidget()
        hl = QHBoxLayout(chk_wrap)
        hl.addWidget(chk)
        hl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl.setContentsMargins(0, 0, 0, 0)
        self._table.setCellWidget(r, 3, chk_wrap)

        dot = QLabel("●")
        dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dot.setStyleSheet("color: #555;")
        self._table.setCellWidget(r, 4, dot)

        self._table.blockSignals(False)
        self.config_changed.emit()

    def _remove_row(self):
        rows = sorted({i.row() for i in self._table.selectedIndexes()}, reverse=True)
        if not rows:
            rows = [self._table.rowCount() - 1]
        for r in rows:
            if r >= 0:
                self._table.removeRow(r)
        self.config_changed.emit()

    # ── Live status ───────────────────────────────────────────────────────────

    def _update_status(self):
        """Poll all enabled PVs and update status dots + summary label."""
        if not self.isChecked():
            self._status_lbl.setText("Disabled")
            self._status_lbl.setStyleSheet("font-size: 11px; color: #888;")
            return

        try:
            from .smart_scan_functions import _EPICS_AVAILABLE
            if not _EPICS_AVAILABLE:
                self._status_lbl.setText("Simulation mode — conditions not checked")
                self._status_lbl.setStyleSheet("font-size: 11px; color: #888;")
                return
            import epics
        except Exception:
            return

        from .smart_scan_functions import _eval_condition

        active = []
        for r in range(self._table.rowCount()):
            pv_item  = self._table.item(r, 0)
            val_item = self._table.item(r, 2)
            op_cb    = self._table.cellWidget(r, 1)
            chk_wrap = self._table.cellWidget(r, 3)
            dot      = self._table.cellWidget(r, 4)
            if not all([pv_item, val_item, op_cb, chk_wrap, dot]):
                continue

            chk = chk_wrap.findChild(QCheckBox)
            if chk and not chk.isChecked():
                dot.setStyleSheet("color: #555;")
                continue

            pv  = pv_item.text().strip()
            val = val_item.text().strip()
            op  = op_cb.currentText()
            if not pv:
                dot.setStyleSheet("color: #555;")
                continue

            try:
                actual = epics.caget(pv, timeout=1.0)
                if actual is None:
                    dot.setStyleSheet("color: #888;")
                    continue
                # red dot = condition is True = suspension would trigger
                triggered = _eval_condition(actual, op, val)
                dot.setStyleSheet("color: #c62828;" if triggered else "color: #2e7d32;")
                if triggered:
                    active.append(f"{pv} {op} {val} (={actual})")
            except Exception:
                dot.setStyleSheet("color: #888;")

        if active:
            self._status_lbl.setText("⛔ Active: " + "; ".join(active))
            self._status_lbl.setStyleSheet("font-size: 11px; color: #ef5350;")
        else:
            self._status_lbl.setText("✓ No conditions triggered")
            self._status_lbl.setStyleSheet("font-size: 11px; color: #66bb6a;")

    def set_hold_active(self, msg: str):
        self._status_lbl.setText(f"⏸ ON HOLD — {msg}")
        self._status_lbl.setStyleSheet(
            "font-size: 11px; color: #ffa726; font-weight: bold;")

    def set_hold_cleared(self):
        self._status_lbl.setText("✓ Hold cleared — restarting row")
        self._status_lbl.setStyleSheet("font-size: 11px; color: #66bb6a;")

    # ── Config API ────────────────────────────────────────────────────────────

    def get_config(self) -> dict:
        conditions = []
        for r in range(self._table.rowCount()):
            pv_item  = self._table.item(r, 0)
            val_item = self._table.item(r, 2)
            op_cb    = self._table.cellWidget(r, 1)
            chk_wrap = self._table.cellWidget(r, 3)
            pv = pv_item.text().strip() if pv_item else ""
            if not pv:
                continue
            op  = op_cb.currentText() if op_cb else ">"
            val = val_item.text().strip() if val_item else ""
            chk = chk_wrap.findChild(QCheckBox) if chk_wrap else None
            conditions.append({
                "pv":      pv,
                "op":      op,
                "value":   val,
                "enabled": chk.isChecked() if chk else True,
            })
        return {
            "enabled":    self.isChecked(),
            "logic":      "any_triggered" if self._logic_cb.currentIndex() == 0 else "all_triggered",
            "conditions": conditions,
        }

    # ── Settings persistence ──────────────────────────────────────────────────

    def reload_settings(self):
        self._table.setRowCount(0)
        self._load_settings()

    def save_settings(self):
        self._settings.setValue("hold_conditions", json.dumps(self.get_config()))

    def _load_settings(self):
        raw = self._settings.value("hold_conditions", None)
        if not raw:
            return
        try:
            cfg = json.loads(raw)
        except Exception:
            return
        self.setChecked(bool(cfg.get("enabled", False)))
        logic = cfg.get("logic", "any_triggered")
        # backward-compat: map old "any_fail"/"all_fail" keys to new names
        if logic == "any_fail":
            logic = "any_triggered"
        elif logic == "all_fail":
            logic = "all_triggered"
        self._logic_cb.setCurrentIndex(0 if logic == "any_triggered" else 1)
        for cond in cfg.get("conditions", []):
            self._add_row(
                pv=cond.get("pv", ""),
                op=cond.get("op", ">"),
                value=str(cond.get("value", "")),
                enabled=bool(cond.get("enabled", True)),
            )
