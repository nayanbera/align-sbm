"""Energy table editor tab."""
import csv
import io

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QFileDialog, QMessageBox, QHeaderView,
)

from .smart_scan_functions import table400

_COLS = ["MonoE (keV)", "Harmonic", "UndE (eV)", "Roll2 (mdeg)", "X2 (μm)",
         "Roll1 (mdeg)", "Crystal", "Updated"]
_KEYS = ["MonoE", "Harmonic", "UndE", "Roll2", "X2"]
_ROLL1_COL   = len(_KEYS)       # 5
_CRYSTAL_COL = len(_KEYS) + 1   # 6
_UPDATED_COL  = len(_KEYS) + 2  # 7

# Save-array indices (separate from table column indices):
#   [0-4] floats: MonoE Harmonic UndE Roll2 X2
#   [5]   str:    timestamp
#   [6]   str:    crystal name
#   [7]   str:    Roll1 RBV
_SAVE_TS      = 5
_SAVE_CRYSTAL = 6
_SAVE_ROLL1   = 7


class _NumericItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically instead of lexicographically."""
    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return super().__lt__(other)


class EnergyTab(QWidget):
    rows_changed = pyqtSignal()   # emitted whenever row count or MonoE values change

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self._settings          = settings
        self._auto_fill_fn      = None   # callable() → {"crystal": str, "roll1": str}
        self._crystal_color_fn  = None   # callable(crystal_name) → hex color str or ""
        self._crystal_label_fn  = None   # callable(raw energyMode value) → crystal label or ""
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
            "<b>X2</b> (nominal X2 position), "
            "<b>Roll1</b> (Roll1 position at this energy), "
            "<b>Crystal</b> (crystal for this energy range)."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self._table = QTableWidget(0, len(_COLS))
        self._table.setHorizontalHeaderLabels(_COLS)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hdr.setStretchLastSection(False)
        hdr.setSectionsClickable(True)
        hdr.setSortIndicatorShown(True)
        hdr.setSortIndicator(0, Qt.SortOrder.AscendingOrder)
        for c, width in enumerate([90, 80, 90, 100, 90, 100, 110, 150]):
            self._table.setColumnWidth(c, width)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSortingEnabled(True)
        self._table.itemChanged.connect(self._on_item_changed)
        self._table.horizontalHeader().sortIndicatorChanged.connect(
            lambda *_: QTimer.singleShot(0, self.rows_changed.emit)
        )
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
        update_btn = QPushButton("Update from CSV")
        update_btn.setToolTip(
            "Set Roll2 / X2 of every row to the most recent record in the alignment CSV\n"
            "with the same MonoE and crystal.\n"
            "The crystal of a CSV record comes from its 'energyMode' column (via the Crystal\n"
            "mappings). If that is unavailable the record matches only when no other row has\n"
            "the same MonoE. Records with nan Roll2/X2 are ignored."
        )
        update_btn.clicked.connect(self._update_from_csv)
        btn_row.addWidget(update_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    # ── public API ──────────────────────────────────────────────────────────

    def set_auto_fill_fn(self, fn):
        """Set a callable invoked when Add Row is clicked: fn() → {"crystal": str, "roll1": str}."""
        self._auto_fill_fn = fn

    def set_crystal_label_fn(self, fn):
        """Set fn(raw_energy_mode_value) → crystal label, used to identify the crystal of CSV rows."""
        self._crystal_label_fn = fn

    def set_crystal_color_fn(self, fn):
        """Set a callable fn(crystal_name) → hex color str used to color table rows."""
        self._crystal_color_fn = fn
        self.refresh_row_colors()

    def refresh_row_colors(self):
        """Re-apply crystal colors to every row (call when mappings change)."""
        for r in range(self._table.rowCount()):
            self._apply_row_color(r)

    def _apply_row_color(self, r: int):
        from PyQt6.QtGui import QColor, QBrush
        if self._crystal_color_fn is None:
            return
        cr_item = self._table.item(r, _CRYSTAL_COL)
        crystal  = cr_item.text().strip() if cr_item else ""
        hex_color = self._crystal_color_fn(crystal) if crystal else ""
        self._table.blockSignals(True)
        try:
            if hex_color:
                bg = QColor(hex_color)
                bg.setAlpha(255)   # force opaque — stored as #AARRGGBB, alpha may be 0
                lum = 0.299 * bg.red() + 0.587 * bg.green() + 0.114 * bg.blue()
                fg  = QColor("#000000" if lum > 128 else "#ffffff")
                bg_brush = QBrush(bg)
                fg_brush = QBrush(fg)
            for c in range(self._table.columnCount()):
                item = self._table.item(r, c)
                if item is None:
                    continue
                if hex_color:
                    item.setBackground(bg_brush)
                    item.setForeground(fg_brush)
                else:
                    item.setBackground(QBrush())
                    item.setForeground(QBrush())
        finally:
            self._table.blockSignals(False)

    def _on_item_changed(self, item):
        self.rows_changed.emit()
        if item.column() == _CRYSTAL_COL:
            self._apply_row_color(item.row())

    def get_row_crystals(self) -> list:
        """Return the crystal name for each row (empty string if none)."""
        result = []
        for r in range(self._table.rowCount()):
            item = self._table.item(r, _CRYSTAL_COL)
            result.append(item.text().strip() if item else "")
        return result

    def get_table(self):
        """Return list of [MonoE, Harmonic, UndE, Roll2, X2, Roll1] rows.

        Roll1 (index 5) is a float if the cell has a value, or None if blank.
        """
        rows = []
        for r in range(self._table.rowCount()):
            try:
                row = []
                for c in range(len(_KEYS)):
                    item = self._table.item(r, c)
                    text = item.text().strip() if item else ""
                    row.append(float(text) if text else 0.0)
                roll1_item = self._table.item(r, _ROLL1_COL)
                roll1_text = roll1_item.text().strip() if roll1_item else ""
                row.append(float(roll1_text) if roll1_text else None)
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
        """Return display strings for the alignment tab list, including crystal if set."""
        labels = []
        for r in range(self._table.rowCount()):
            item    = self._table.item(r, 0)
            val     = item.text() if item else "?"
            cr_item = self._table.item(r, _CRYSTAL_COL)
            crystal = cr_item.text().strip() if cr_item else ""
            labels.append(f"{val} keV  ·  {crystal}" if crystal else f"{val} keV")
        return labels

    def update_row_after_alignment(self, mono_e: float, roll2: float, x2: float,
                                   timestamp_str: str, crystal=None) -> bool:
        """Update Roll2, X2, and the Updated timestamp for the row matching mono_e.

        If *crystal* is not None the row's Crystal cell must also match (blank matches
        blank), so rows sharing a MonoE but using different crystals are kept apart.
        Returns True if a matching row was found and updated.
        """
        matched = False
        self._table.setSortingEnabled(False)
        self._table.blockSignals(True)
        try:
            for r in range(self._table.rowCount()):
                item = self._table.item(r, 0)
                if item is None:
                    continue
                try:
                    row_mono_e = float(item.text().strip())
                except ValueError:
                    continue
                if abs(row_mono_e - mono_e) < 0.001:
                    if crystal is not None:
                        c_item = self._table.item(r, _CRYSTAL_COL)
                        if (c_item.text().strip() if c_item else "") != crystal.strip():
                            continue
                    for col, text in [(3, f"{roll2:.6g}"), (4, f"{x2:.6g}")]:
                        cell = self._table.item(r, col)
                        if cell is None:
                            cell = _NumericItem()
                            cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                            self._table.setItem(r, col, cell)
                        cell.setText(text)
                    ts_item = self._table.item(r, _UPDATED_COL)
                    if ts_item is None:
                        ts_item = QTableWidgetItem()
                        ts_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                        ts_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        self._table.setItem(r, _UPDATED_COL, ts_item)
                    ts_item.setText(timestamp_str)
                    matched = True
                    break
        finally:
            self._table.blockSignals(False)
            self._table.setSortingEnabled(True)
            self.rows_changed.emit()
        return matched

    def reload_settings(self):
        self._load_settings()

    def save_settings(self):
        rows = []
        for r in range(self._table.rowCount()):
            try:
                row = []
                for c in range(len(_KEYS)):
                    item = self._table.item(r, c)
                    text = item.text().strip() if item else ""
                    row.append(float(text) if text else 0.0)
                # index 5: timestamp
                ts_item = self._table.item(r, _UPDATED_COL)
                row.append(ts_item.text() if ts_item else "")
                # index 6: crystal
                cr_item = self._table.item(r, _CRYSTAL_COL)
                row.append(cr_item.text().strip() if cr_item else "")
                # index 7: roll1
                roll1_item = self._table.item(r, _ROLL1_COL)
                row.append(roll1_item.text() if roll1_item else "")
                rows.append(row)
            except ValueError:
                pass
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
        self._table.blockSignals(True)
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)
        for row in rows:
            self._append_row(row)
        self._table.setSortingEnabled(True)
        self._table.blockSignals(False)
        self.refresh_row_colors()
        self.rows_changed.emit()

    def _append_row(self, values=None):
        """Append one row from a saved-data array.

        Save-array layout (for backwards compatibility):
          [0-4]  floats  MonoE Harmonic UndE Roll2 X2
          [5]    str     timestamp          (old saves: 6 elements)
          [6]    str     crystal name       (added 2026-09-22, 7+ elements)
          [7]    str     Roll1 RBV          (added 2026-09-22, 8+ elements)
        """
        self._table.setSortingEnabled(False)
        r = self._table.rowCount()
        self._table.insertRow(r)
        defaults = [0.0] * len(_KEYS)
        vals = list(values) if values is not None else defaults

        # Data columns (0-4)
        for c, v in enumerate(vals[:len(_KEYS)]):
            item = _NumericItem(str(v))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(r, c, item)

        # Roll1 column (5) — plain editable text
        roll1_text = str(vals[_SAVE_ROLL1]) if len(vals) > _SAVE_ROLL1 else ""
        roll1_item = _NumericItem(roll1_text)
        roll1_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setItem(r, _ROLL1_COL, roll1_item)

        # Crystal column (6) — plain editable text
        crystal_name = str(vals[_SAVE_CRYSTAL]) if len(vals) > _SAVE_CRYSTAL else ""
        cr_item = QTableWidgetItem(crystal_name)
        cr_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setItem(r, _CRYSTAL_COL, cr_item)

        # Updated column (7) — read-only timestamp
        ts_text = str(vals[_SAVE_TS]) if len(vals) > _SAVE_TS else ""
        updated_item = QTableWidgetItem(ts_text)
        updated_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        updated_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setItem(r, _UPDATED_COL, updated_item)

        self._apply_row_color(r)
        self._table.setSortingEnabled(True)

    def _add_row(self):
        """Add a blank row, auto-filling Roll1 and Crystal from the current live state."""
        extras = {}
        if self._auto_fill_fn is not None:
            try:
                extras = self._auto_fill_fn() or {}
            except Exception:
                pass
        # Build an 8-element values array with blanks for data cols
        values = [0.0, 0.0, 0.0, 0.0, 0.0,   # MonoE..X2
                  "",                           # timestamp
                  extras.get("crystal", ""),    # crystal
                  extras.get("roll1", "")]      # Roll1
        self._append_row(values)

    def _remove_row(self):
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()}, reverse=True)
        if not rows:
            rows = [self._table.rowCount() - 1]
        for r in rows:
            if r >= 0:
                self._table.removeRow(r)
        self.rows_changed.emit()

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

    def compute_csv_updates(self, csv_path: str):
        """Match every table row to its most recent CSV record (same MonoE and crystal).

        Returns (updates, no_match, ambiguous) where *updates* is a list of
        (table_row, roll2, x2, datetime_str); *no_match* / *ambiguous* are lists of table rows.
        """
        import math
        from datetime import datetime

        with open(csv_path, newline="") as f:
            records = list(csv.DictReader(f))

        def _f(v):
            try:
                x = float(v)
                return x if math.isfinite(x) else None
            except (TypeError, ValueError):
                return None

        recs = []   # (sort_key, mono, crystal-or-None, roll2, x2, dt_str)
        for i, rec in enumerate(records):
            mono, roll2, x2 = _f(rec.get("MonoE")), _f(rec.get("Roll2")), _f(rec.get("X2"))
            if mono is None or roll2 is None or x2 is None:
                continue
            dt_str = (rec.get("datetime") or "").strip()
            try:
                dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                dt = datetime.min
            crystal = None
            raw = (rec.get("energyMode") or "").strip()
            if raw and self._crystal_label_fn is not None:
                crystal = self._crystal_label_fn(raw) or None
            recs.append(((dt, i), mono, crystal, roll2, x2, dt_str))
        recs.sort(key=lambda r: r[0])   # oldest → newest

        table_rows = []   # (table_row, mono, crystal)
        for r in range(self._table.rowCount()):
            item = self._table.item(r, 0)
            try:
                mono = float(item.text().strip())
            except (AttributeError, ValueError):
                continue
            c_item = self._table.item(r, _CRYSTAL_COL)
            table_rows.append((r, mono, c_item.text().strip() if c_item else ""))

        updates, no_match, ambiguous = [], [], []
        for r, mono, crystal in table_rows:
            shared = sum(1 for _, m, _c in table_rows if abs(m - mono) < 0.001) > 1
            best = None
            seen_unknown_blocked = False
            for rec in recs:   # ascending → last acceptable wins
                if abs(rec[1] - mono) >= 0.001:
                    continue
                rec_cry = rec[2]
                if rec_cry is not None:
                    ok = (not crystal) or rec_cry == crystal
                else:
                    ok = not shared
                    if not ok:
                        seen_unknown_blocked = True
                if ok:
                    best = rec
            if best is not None:
                updates.append((r, best[3], best[4], best[5]))
            elif seen_unknown_blocked:
                ambiguous.append(r)
            else:
                no_match.append(r)
        return updates, no_match, ambiguous

    def _update_from_csv(self):
        import os
        csv_path = self._settings.value("last_csv_path", "") if self._settings else ""
        if not csv_path or not os.path.isfile(csv_path):
            QMessageBox.warning(self, "Update from CSV",
                                "No alignment CSV is open.\n"
                                "Use 'Open CSV…' in the Alignment tab first.")
            return
        try:
            updates, no_match, ambiguous = self.compute_csv_updates(csv_path)
        except Exception as e:
            QMessageBox.critical(self, "Update from CSV", f"Could not read CSV:\n{e}")
            return
        if not updates:
            QMessageBox.information(
                self, "Update from CSV",
                "No table rows have a matching record in the CSV.\n"
                + (f"{len(ambiguous)} row(s) were skipped because their crystal could not be "
                   "determined from the CSV." if ambiguous else ""))
            return
        reply = QMessageBox.question(
            self, "Update from CSV",
            f"Overwrite Roll2 / X2 of {len(updates)} of {self._table.rowCount()} row(s) "
            f"with the most recent matching record in:\n{csv_path}\n\n"
            f"No matching record: {len(no_match)} row(s)\n"
            f"Crystal ambiguous (skipped): {len(ambiguous)} row(s)\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._table.setSortingEnabled(False)
        self._table.blockSignals(True)
        try:
            for r, roll2, x2, dt_str in updates:
                for col, text in [(3, f"{roll2:.6g}"), (4, f"{x2:.6g}")]:
                    cell = self._table.item(r, col)
                    if cell is None:
                        cell = _NumericItem()
                        cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        self._table.setItem(r, col, cell)
                    cell.setText(text)
                if dt_str:
                    ts_item = self._table.item(r, _UPDATED_COL)
                    if ts_item is None:
                        ts_item = QTableWidgetItem()
                        ts_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                        ts_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        self._table.setItem(r, _UPDATED_COL, ts_item)
                    ts_item.setText(dt_str)
        finally:
            self._table.blockSignals(False)
            self._table.setSortingEnabled(True)
            self.rows_changed.emit()
        QMessageBox.information(
            self, "Update from CSV",
            f"Updated {len(updates)} row(s).  No match: {len(no_match)}.  "
            f"Ambiguous crystal: {len(ambiguous)}.")

    def _predict_from_csv(self):
        from ._predict_dialog import PredictDialog
        csv_path = self._settings.value("last_csv_path", "")
        dlg = PredictDialog(csv_path=csv_path, parent=self)
        if dlg.exec():
            for row in dlg.get_predicted_rows():
                self._append_row(row)

    def _reset(self):
        self._populate(table400)
