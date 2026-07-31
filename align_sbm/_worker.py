"""Background worker thread — runs alignment row-by-row in subprocesses."""
import os
import time
import traceback
from types import SimpleNamespace

from PyQt6.QtCore import QThread, pyqtSignal


class AlignWorker(QThread):
    log_chunk           = pyqtSignal(str)
    scan_started        = pyqtSignal(str)            # tab name
    point_measured      = pyqtSignal(float, float)   # coarse (x, y)
    fine_point_measured = pyqtSignal(float, float)   # fine   (x, y)
    scan_finished       = pyqtSignal(object)         # SimpleNamespace result summary
    step_update         = pyqtSignal(str, int, int)  # (label, current, total)
    row_done            = pyqtSignal(dict)           # record after each row
    done                = pyqtSignal(list)           # all records at end
    error               = pyqtSignal(str)            # traceback string
    hold_triggered      = pyqtSignal(str)            # failure description
    hold_cleared        = pyqtSignal()               # conditions restored

    def __init__(self, table, kwargs, simulate, hold_config=None, parent=None):
        super().__init__(parent)
        self._table       = table
        self._kwargs      = kwargs
        self._simulate    = simulate
        self._hold_config = hold_config or {"enabled": False, "conditions": []}
        self._current_proc = None   # active subprocess (for abort())

    # ── Public abort API ──────────────────────────────────────────────────────

    def abort(self):
        """Terminate the running subprocess and request the thread to stop."""
        if self._current_proc is not None:
            try:
                if self._current_proc.is_alive():
                    self._current_proc.terminate()
                    self._current_proc.join(timeout=2)
                    if self._current_proc.is_alive():
                        self._current_proc.kill()
            except Exception:
                pass
            self._current_proc = None
        self.requestInterruption()

    # ── Hold condition helpers ────────────────────────────────────────────────

    def _check_hold(self) -> list:
        """Return triggered-condition strings. Empty → no hold needed.

        Semantics: hold when the condition expression evaluates to *True*.
        Example: 'SR:Current < 10' triggers when current drops below 10.
        """
        cfg = self._hold_config
        if not cfg.get("enabled", False):
            return []
        try:
            from .smart_scan_functions import _EPICS_AVAILABLE, _eval_condition
            if not _EPICS_AVAILABLE:
                return []
            import epics
        except Exception:
            return []

        conditions = [
            c for c in cfg.get("conditions", [])
            if c.get("enabled", True) and c.get("pv", "").strip()
        ]
        if not conditions:
            return []

        logic     = cfg.get("logic", "any_triggered")
        triggered = []
        for cond in conditions:
            try:
                actual = epics.caget(cond["pv"].strip(), timeout=1.0)
                if actual is not None and _eval_condition(actual, cond["op"], cond["value"]):
                    triggered.append(
                        f"{cond['pv']} {cond['op']} {cond['value']} (actual={actual})"
                    )
            except Exception as e:
                self.log_chunk.emit(f"[Hold] CA read error {cond['pv']}: {e}\n")

        # "any_triggered" (default): hold when at least one condition is True
        # "all_triggered": hold only when every enabled condition is True simultaneously
        # backward-compat: accept old keys "any_fail" / "all_fail"
        if logic in ("any_triggered", "any_fail"):
            return triggered
        return triggered if len(triggered) >= len(conditions) else []

    def _wait_for_clear(self):
        """Block, polling every 2 s, until all hold conditions pass (or abort)."""
        while not self.isInterruptionRequested():
            if not self._check_hold():
                return
            time.sleep(2.0)

    # ── Queue message dispatcher ──────────────────────────────────────────────

    def _dispatch(self, msg, row_idx: int, n_total: int):
        """Route one queue message to the appropriate Qt signal."""
        from .smart_scan_functions import ScanStatus
        kind = msg[0]
        if kind == "log":
            self.log_chunk.emit(msg[1])
        elif kind == "scan_started":
            self.scan_started.emit(msg[1])
        elif kind == "point":
            self.point_measured.emit(msg[1], msg[2])
        elif kind == "fine_point":
            self.fine_point_measured.emit(msg[1], msg[2])
        elif kind == "scan_finished":
            d = msg[1]
            result = SimpleNamespace(
                status=ScanStatus(d["status"]),
                center=d["center"], sigma=d["sigma"],
                amplitude=d["amplitude"], offset=d["offset"],
                profile=d["profile"], stats=d["stats"],
            )
            self.scan_finished.emit(result)
        elif kind == "step":
            steps_per_row = 11 if self._kwargs.get("do_pitch_scan", True) else 8
            global_current = row_idx * steps_per_row + msg[2]
            global_total   = n_total * steps_per_row
            self.step_update.emit(msg[1], global_current, global_total)
        elif kind == "error":
            self.error.emit(msg[1])
        # "row_done" and "subprocess_done" are handled by the caller

    # ── Main thread loop ──────────────────────────────────────────────────────

    def run(self):
        import sys
        import multiprocessing as mp
        from ._subprocess_runner import run_alignment_row

        # fork on Linux: inherits loaded CA library and EPICS env so connections
        # succeed reliably; ca.create_context() in the child gives a fresh context.
        # spawn on Windows (fork unavailable) and macOS (works fine there).
        if sys.platform == "win32":
            ctx = mp.get_context("spawn")
        else:
            ctx = mp.get_context("fork")

        n_total     = len(self._table)
        all_records = []
        row_idx     = 0

        # Absolute-ify the CSV filename so the subprocess writes to the right place
        kwargs_base = dict(self._kwargs)
        raw_fname   = kwargs_base.get("filename", "alignment_results.csv")
        kwargs_base["filename"] = os.path.abspath(raw_fname)

        steps_per_row = 11 if kwargs_base.get("do_pitch_scan", True) else 8

        while row_idx < n_total and not self.isInterruptionRequested():
            row = self._table[row_idx]

            # ── Pre-row: wait for hold conditions to pass ─────────────────────
            failures = self._check_hold()
            if failures:
                self.hold_triggered.emit("; ".join(failures))
                self._wait_for_clear()
                if self.isInterruptionRequested():
                    break
                self.hold_cleared.emit()

            # ── Spawn subprocess for this row ─────────────────────────────────
            q   = ctx.Queue()
            proc = ctx.Process(
                target=run_alignment_row,
                args=(q, row, dict(kwargs_base), self._simulate),
                daemon=True,
            )
            self._current_proc = proc
            proc.start()

            held       = False
            row_record = None
            last_hold_check = time.monotonic()
            HOLD_INTERVAL   = 3.0   # seconds between condition polls

            # ── Monitor subprocess & drain queue ──────────────────────────────
            while True:
                # Drain all currently available messages
                while True:
                    try:
                        msg = q.get(timeout=0.1)
                    except Exception:
                        break
                    if msg[0] == "row_done":
                        row_record = msg[1]
                    elif msg[0] != "subprocess_done":
                        self._dispatch(msg, row_idx, n_total)

                if self.isInterruptionRequested():
                    proc.terminate()
                    proc.join(timeout=2)
                    if proc.is_alive():
                        proc.kill()
                    held = False   # treat as clean abort, not hold
                    break

                if not proc.is_alive():
                    # Drain any messages queued just before exit
                    while True:
                        try:
                            msg = q.get_nowait()
                        except Exception:
                            break
                        if msg[0] == "row_done":
                            row_record = msg[1]
                        elif msg[0] != "subprocess_done":
                            self._dispatch(msg, row_idx, n_total)
                    break

                # Periodic hold check (every HOLD_INTERVAL seconds)
                now = time.monotonic()
                if now - last_hold_check >= HOLD_INTERVAL:
                    last_hold_check = now
                    failures = self._check_hold()
                    if failures:
                        # Kill subprocess, wait for conditions, restart row
                        proc.terminate()
                        proc.join(timeout=3)
                        if proc.is_alive():
                            proc.kill()
                        msg_str = "; ".join(failures)
                        self.hold_triggered.emit(msg_str)
                        print(f"\n  ⏸ HOLD triggered: {msg_str}")
                        print("  Waiting for conditions to clear …")
                        self._wait_for_clear()
                        if not self.isInterruptionRequested():
                            self.hold_cleared.emit()
                            print("  ✓ Hold cleared — restarting energy row from beginning")
                        held = True
                        break

            # Drain & close queue
            try:
                while True:
                    q.get_nowait()
            except Exception:
                pass
            try:
                q.close()
            except Exception:
                pass
            self._current_proc = None

            if self.isInterruptionRequested():
                break

            if held:
                continue   # restart same row_idx

            # Row completed normally
            if row_record:
                all_records.append(row_record)
                self.row_done.emit(dict(row_record))

            row_idx += 1

        self._current_proc = None
        self.done.emit(all_records)
