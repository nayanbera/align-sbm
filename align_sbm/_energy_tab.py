"""Energy table editor tab."""
import csv
import io

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QFileDialog, QMessageBox, QHeaderView,
)

from .smart_scan_functions import table400

_COLS = ["MonoE (keV)", "Harmonic", "UndE (eV)", "Roll2 (mdeg)", "X2 (μm)", "Updated"]
_KEYS = ["MonoE", "Harmonic", "UndE", "Roll2", "X2"]


class EnergyTab(QWidget):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._build_ui()
        self._load_settings()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        info = QLabel(
            "Energy lookup table — columns: "
            "<b>MonoE</b> (mono target keV), "
            "<b>Harmonic</b> (undulator harmonic), "
            "<b>UndE</b> (undulator energy), "
            "<b>Roll2</b> (nominal Roll2 encoder value), "
            "<b>X2</b> (nominal X2 position)."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self._table = QTableWidget(0, len(_COLS))
        self._table.setHorizontalHeaderLabels(_COLS)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hdr.setStretchLastSection(False)
        for c, width in enumerate([90, 80, 90, 100, 90, 150]):
            self._table.setColumnWidth(c, width)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSortingEnabled(True)
        layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        for label, slot in [
            ("Add Row", self._add_row),
            ("Remove Row", self._remove_row),
            ("Load CSV", self._load_csv),
            ("Save CSV", self._save_csv),
            ("Reset to Defaults", self._reset),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            btn_row.addWidget(btn)
        predict_btn = QPushButton("Predict from CSV…")
        predict_btn.setToolTip(
            "Fit Roll2 and X2 vs MonoE from the alignment CSV history\n"
            "and predict values for new intermediate energies."
        )
        predict_btn.clicked.connect(self._predict_from_csv)
        btn_row.addWidget(predict_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    # ── public API ──────────────────────────────────────────────────────────

    def get_table(self):
        """Return list of [MonoE, Harmonic, UndE, Roll2, X2] rows (floats)."""
        rows = []
        for r in range(self._table.rowCount()):
            try:
                row = []
                for c in range(len(_KEYS)):   # only the 5 data columns, skip "Updated"
                    item = self._table.item(r, c)
                    text = item.text().strip() if item else ""
                    row.append(float(text) if text else 0.0)
                rows.append(row)
            except ValueError:
                pass
        return rows

    def get_selected_rows(self):
        """Return the rows that are currently selected in the table."""
        selected = sorted({idx.row() for idx in self._table.selectedIndexes()})
        all_rows = self.get_table()
        if not selected:
            return all_rows
        return [all_rows[i] for i in selected if i < len(all_rows)]

    def get_row_labels(self):
        """Return list of 'MonoE keV' strings for display in alignment tab."""
        labels = []
        for r in range(self._table.rowCount()):
            item = self._table.item(r, 0)
            val = item.text() if item else "?"
            labels.append(f"{val} keV")
        return labels

    def update_row_after_alignment(self, mono_e: float, roll2: float, x2: float,
                                   timestamp_str: str):
        """Update Roll2, X2, and the Updated timestamp for the row matching mono_e."""
        for r in range(self._table.rowCount()):
            item = self._table.item(r, 0)
            if item is None:
                continue
            try:
                row_mono_e = float(item.text().strip())
            except ValueError:
                continue
            if abs(row_mono_e - mono_e) < 0.001:
                for col, text in [(3, f"{roll2:.6g}"), (4, f"{x2:.6g}")]:
                    cell = self._table.item(r, col)
                    if cell is None:
                        cell = QTableWidgetItem()
                        cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        self._table.setItem(r, col, cell)
                    cell.setText(text)
                updated_col = len(_KEYS)
                ts_item = self._table.item(r, updated_col)
                if ts_item is None:
                    ts_item = QTableWidgetItem()
                    ts_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    ts_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self._table.setItem(r, updated_col, ts_item)
                ts_item.setText(timestamp_str)
                break

    def reload_settings(self):
        self._load_settings()

    def save_settings(self):
        rows = self.get_table()
        self._settings.setValue("energy_table", repr(rows))

    # ── private ─────────────────────────────────────────────────────────────

    def _load_settings(self):
        raw = self._settings.value("energy_table", None)
        if raw:
            try:
                rows = eval(raw)  # noqa: S307  (trusted local QSettings)
                if rows:
                    self._populate(rows)
                    return
            except Exception:
                pass
        self._reset()

    def _populate(self, rows):
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)
        for row in rows:
            self._append_row(row)
        self._table.setSortingEnabled(True)

    def _append_row(self, values=None):
        self._table.setSortingEnabled(False)
        r = self._table.rowCount()
        self._table.insertRow(r)
        defaults = [0.0] * len(_KEYS)
        vals = list(values) if values is not None else defaults
        # Fill the 5 data columns
        for c, v in enumerate(vals[:len(_KEYS)]):
            item = QTableWidgetItem(str(v))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(r, c, item)
        # 6th column: "Updated" — read-only, non-editable
        updated_item = QTableWidgetItem("")
        updated_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        updated_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setItem(r, len(_KEYS), updated_item)
        self._table.setSortingEnabled(True)

    def _add_row(self):
        self._append_row()

    def _remove_row(self):
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()}, reverse=True)
        if not rows:
            rows = [self._table.rowCount() - 1]
        for r in rows:
            if r >= 0:
                self._table.removeRow(r)

    def _load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Energy Table", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                rows = []
                for row in reader:
                    try:
                        rows.append([float(row.get(k, 0)) for k in _KEYS])
                    except ValueError:
                        pass
            if not rows:
                QMessageBox.warning(self, "Load CSV", "No valid rows found.")
                return
            self._populate(rows)
        except Exception as e:
            QMessageBox.critical(self, "Load CSV", str(e))

    def _save_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Energy Table", "energy_table.csv",
                                              "CSV Files (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_KEYS)
                writer.writeheader()
                for row in self.get_table():
                    writer.writerow(dict(zip(_KEYS, row)))
        except Exception as e:
            QMessageBox.critical(self, "Save CSV", str(e))

    def _predict_from_csv(self):
        from ._predict_dialog import PredictDialog
        csv_path = self._settings.value("last_csv_path", "")
        dlg = PredictDialog(csv_path=csv_path, parent=self)
        if dlg.exec():
            for row in dlg.get_predicted_rows():
                self._append_row(row)

    def _reset(self):
        self._populate(table400)
