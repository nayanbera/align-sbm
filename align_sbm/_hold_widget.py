"""Hold Conditions panel — suspends alignment when EPICS PV conditions fail."""
import json

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)


class _NoScrollComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()

_OPS = [">", "<", ">=", "<=", "==", "!="]


class HoldConditionsWidget(QGroupBox):
    """
    Checkable group box listing PV conditions that suspend the alignment.
    When the group box is checked, hold monitoring is active.
    PV values are received via CA monitors (epics.PV callbacks), not polled.
    """

    config_changed    = pyqtSignal()
    # Emitted from the Qt main thread — never from the worker.
    suspend_triggered = pyqtSignal(str)   # conditions are active → suspend
    suspend_cleared   = pyqtSignal()      # conditions cleared → resume

    # Fired from the CA thread when any subscribed PV changes; processed in Qt thread.
    _pv_changed = pyqtSignal()

    def __init__(self, settings, parent=None):
        super().__init__("Hold Conditions")
        self.setCheckable(True)
        self.setChecked(False)
        self._settings          = settings
        self._conditions_active = False
        self._active_msg        = ""
        self._pv_subs: dict     = {}   # pvname -> epics.PV
        self._pv_values: dict   = {}   # pvname -> latest value (or None if disconnected)
        self._loading           = False

        self._build_ui()
        self._pv_changed.connect(self._evaluate_conditions)
        self.toggled.connect(self._evaluate_conditions)
        self._load_settings()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        vbox = QVBoxLayout(self)
        vbox.setSpacing(4)
        vbox.setContentsMargins(6, 4, 6, 6)

        # Logic selector
        logic_row = QHBoxLayout()
        logic_row.addWidget(QLabel("Suspend when:"))
        self._logic_cb = _NoScrollComboBox()
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
        self._table.itemChanged.connect(self._on_item_changed)
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

        op_cb = _NoScrollComboBox()
        op_cb.addItems(_OPS)
        op_cb.setCurrentText(op)
        op_cb.currentIndexChanged.connect(lambda: (self.config_changed.emit(),
                                                    self._evaluate_conditions()))
        self._table.setCellWidget(r, 1, op_cb)

        val_item = QTableWidgetItem(str(value))
        val_item.setToolTip("Threshold — number or quoted/unquoted string")
        self._table.setItem(r, 2, val_item)

        chk = QCheckBox()
        chk.setChecked(enabled)
        chk.stateChanged.connect(lambda: (self.config_changed.emit(),
                                          self._resubscribe()))
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
        if not self._loading:
            self._resubscribe()

    def _remove_row(self):
        rows = sorted({i.row() for i in self._table.selectedIndexes()}, reverse=True)
        if not rows:
            rows = [self._table.rowCount() - 1]
        for r in rows:
            if r >= 0:
                self._table.removeRow(r)
        self.config_changed.emit()
        self._resubscribe()

    def _on_item_changed(self, item):
        """PV name or threshold value edited in the table — re-sync subscriptions."""
        self.config_changed.emit()
        if item.column() == 0:   # PV name changed
            self._resubscribe()
        else:                     # threshold changed — re-evaluate with cached value
            self._evaluate_conditions()

    # ── CA monitor management ─────────────────────────────────────────────────

    def _active_pv_names(self) -> set:
        """Return the set of PV names from all enabled, non-empty rows."""
        names = set()
        for r in range(self._table.rowCount()):
            pv_item  = self._table.item(r, 0)
            chk_wrap = self._table.cellWidget(r, 3)
            if not (pv_item and chk_wrap):
                continue
            chk = chk_wrap.findChild(QCheckBox)
            if chk and not chk.isChecked():
                continue
            pv = pv_item.text().strip()
            if pv:
                names.add(pv)
        return names

    def _resubscribe(self):
        """Sync CA subscriptions to match the current set of enabled PVs."""
        try:
            from .smart_scan_functions import _EPICS_AVAILABLE
            if not _EPICS_AVAILABLE:
                return
            from epics import PV
        except Exception:
            return

        wanted  = self._active_pv_names()
        current = set(self._pv_subs.keys())

        for pv_name in current - wanted:
            try:
                self._pv_subs[pv_name].disconnect()
            except Exception:
                pass
            self._pv_subs.pop(pv_name, None)
            self._pv_values.pop(pv_name, None)

        for pv_name in wanted - current:
            def _make_cb(name):
                def _val_cb(pvname=None, value=None, char_value=None, **kw):
                    self._pv_values[name] = char_value if char_value is not None else value
                    self._pv_changed.emit()
                def _conn_cb(pvname=None, conn=False, **kw):
                    if not conn:
                        self._pv_values.pop(name, None)
                        self._pv_changed.emit()
                return _val_cb, _conn_cb

            val_cb, conn_cb = _make_cb(pv_name)
            self._pv_subs[pv_name] = PV(
                pv_name,
                callback=val_cb,
                connection_callback=conn_cb,
                auto_monitor=True,
            )

        self._evaluate_conditions()

    def _unsubscribe_all(self):
        for pv_obj in self._pv_subs.values():
            try:
                pv_obj.disconnect()
            except Exception:
                pass
        self._pv_subs.clear()
        self._pv_values.clear()

    # ── Condition evaluation ──────────────────────────────────────────────────

    def _evaluate_conditions(self):
        """Re-evaluate all conditions using cached PV values; update dots and status."""
        if not self.isChecked():
            self._status_lbl.setText("Disabled")
            self._status_lbl.setStyleSheet("font-size: 11px; color: #888;")
            return

        try:
            from .smart_scan_functions import _EPICS_AVAILABLE, _eval_condition
            if not _EPICS_AVAILABLE:
                self._status_lbl.setText("Simulation mode — conditions not checked")
                self._status_lbl.setStyleSheet("font-size: 11px; color: #888;")
                return
        except Exception:
            return

        logic    = self._logic_cb.currentIndex()   # 0 = any, 1 = all
        results  = []   # (triggered: bool, description: str)

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

            if pv not in self._pv_values:
                dot.setStyleSheet("color: #888;")   # not yet received
                results.append((False, None))
                continue

            actual = self._pv_values[pv]
            if actual is None:
                dot.setStyleSheet("color: #888;")   # disconnected
                results.append((False, None))
                continue

            try:
                triggered = _eval_condition(actual, op, val)
            except Exception:
                triggered = False

            dot.setStyleSheet("color: #c62828;" if triggered else "color: #2e7d32;")
            results.append((triggered, f"{pv} {op} {val} (={actual})"))

        # Apply logic gate across enabled rows
        triggered_descs = [desc for trig, desc in results if trig and desc]
        unknown         = any(desc is None for _, desc in results)

        if logic == 0:   # any
            new_active = bool(triggered_descs)
        else:            # all — only if every enabled row with a known value triggered
            known = [(t, d) for t, d in results if d is not None]
            new_active = bool(known) and all(t for t, _ in known)
            if new_active:
                triggered_descs = [d for t, d in known if t]

        msg = "; ".join(triggered_descs)

        if new_active:
            self._status_lbl.setText("⛔ Active: " + msg)
            self._status_lbl.setStyleSheet("font-size: 11px; color: #ef5350;")
        elif unknown and not triggered_descs:
            self._status_lbl.setText("◌ Waiting for PV values…")
            self._status_lbl.setStyleSheet("font-size: 11px; color: #888;")
        else:
            self._status_lbl.setText("✓ No conditions triggered")
            self._status_lbl.setStyleSheet("font-size: 11px; color: #66bb6a;")

        if new_active and not self._conditions_active:
            self._active_msg        = msg
            self._conditions_active = True
            self.suspend_triggered.emit(msg)
        elif not new_active and self._conditions_active:
            self._active_msg        = ""
            self._conditions_active = False
            self.suspend_cleared.emit()

    # ── Worker sync ───────────────────────────────────────────────────────────

    def sync_worker(self, worker):
        """Pre-sync current hold state to a freshly created worker.

        Called from _launch_worker() right after worker.start() so that if
        conditions were already active when the alignment was launched the
        worker is immediately suspended — without waiting for a CA update.
        """
        if self.isChecked() and self._conditions_active:
            worker.suspend(self._active_msg)

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
        self._unsubscribe_all()
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
        if logic == "any_fail":
            logic = "any_triggered"
        elif logic == "all_fail":
            logic = "all_triggered"
        self._logic_cb.setCurrentIndex(0 if logic == "any_triggered" else 1)
        self._loading = True
        for cond in cfg.get("conditions", []):
            self._add_row(
                pv=cond.get("pv", ""),
                op=cond.get("op", ">"),
                value=str(cond.get("value", "")),
                enabled=bool(cond.get("enabled", True)),
            )
        self._loading = False
        self._resubscribe()
