"""Background worker thread — runs alignment row-by-row in subprocesses."""
import os
import threading
import time
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
    hold_triggered      = pyqtSignal(str)            # hold is active (msg)
    hold_cleared        = pyqtSignal()               # hold lifted

    def __init__(self, table, kwargs, simulate, parent=None):
        super().__init__(parent)
        self._table        = table
        self._kwargs       = kwargs
        self._simulate     = simulate
        self._current_proc = None          # active subprocess
        # Threading event set by the main thread via suspend()/resume().
        # The worker checks this flag instead of doing its own CA reads,
        # which avoids post-fork CA socket reliability issues entirely.
        self._suspend_flag = threading.Event()
        self._suspend_msg  = ""

    # ── Public API called from the main thread ────────────────────────────────

    def abort(self):
        """Terminate the running subprocess and request the thread to stop."""
        self._suspend_flag.clear()   # unblock any pending wait
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

    def suspend(self, msg: str):
        """Called from the main thread when hold conditions become active.

        Sets the suspend flag and immediately kills the running subprocess so
        the monitoring loop can react within one drain-cycle (~0.1 s).
        """
        self._suspend_msg = msg
        self._suspend_flag.set()
        try:
            if self._current_proc is not None and self._current_proc.is_alive():
                self._current_proc.terminate()
        except Exception:
            pass

    def resume(self):
        """Called from the main thread when hold conditions clear."""
        self._suspend_flag.clear()

    # ── Queue message dispatcher ──────────────────────────────────────────────

    def _dispatch(self, msg, row_idx: int, n_total: int):
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
            steps_per_row  = 11 if self._kwargs.get("do_pitch_scan", True) else 8
            global_current = row_idx * steps_per_row + msg[2]
            global_total   = n_total * steps_per_row
            self.step_update.emit(msg[1], global_current, global_total)
        elif kind == "error":
            self.error.emit(msg[1])

    # ── Main thread loop ──────────────────────────────────────────────────────

    def run(self):
        import sys
        import multiprocessing as mp
        from ._subprocess_runner import run_alignment_row

        # fork on Linux/macOS: child inherits the loaded CA library and EPICS
        # env variables, so connections succeed without re-initialising libca.
        # ca.create_context() in the child gives it a fresh CA context.
        # spawn on Windows (fork not available).
        if sys.platform == "win32":
            ctx = mp.get_context("spawn")
        else:
            ctx = mp.get_context("fork")

        n_total     = len(self._table)
        all_records = []
        row_idx     = 0

        kwargs_base = dict(self._kwargs)
        raw_fname   = kwargs_base.get("filename", "alignment_results.csv")
        kwargs_base["filename"] = os.path.abspath(raw_fname)

        while row_idx < n_total and not self.isInterruptionRequested():
            row = self._table[row_idx]

            # ── Pre-row: wait for any active hold to clear ────────────────────
            if self._suspend_flag.is_set():
                self.hold_triggered.emit(self._suspend_msg)
                print(f"\n  ⏸ HOLD before row: {self._suspend_msg}")
                while not self.isInterruptionRequested() and self._suspend_flag.is_set():
                    time.sleep(0.5)
                if self.isInterruptionRequested():
                    break
                self.hold_cleared.emit()
                print("  ✓ Hold cleared — starting row")

            # ── Spawn subprocess for this row ─────────────────────────────────
            q    = ctx.Queue()
            proc = ctx.Process(
                target=run_alignment_row,
                args=(q, row, dict(kwargs_base), self._simulate),
                daemon=True,
            )
            self._current_proc = proc
            proc.start()

            held       = False
            row_record = None

            # ── Monitor subprocess & drain queue ──────────────────────────────
            while True:
                # Drain all currently available messages (~0.1 s block)
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
                    if proc.is_alive():
                        proc.terminate()
                        proc.join(timeout=2)
                        if proc.is_alive():
                            proc.kill()
                    break

                # Hold flag set by main thread via suspend() — subprocess may
                # already be terminated (suspend() calls terminate() directly).
                if self._suspend_flag.is_set():
                    if proc.is_alive():
                        proc.terminate()
                        proc.join(timeout=3)
                        if proc.is_alive():
                            proc.kill()
                    self.hold_triggered.emit(self._suspend_msg)
                    print(f"\n  ⏸ HOLD triggered: {self._suspend_msg}")
                    print("  Waiting for conditions to clear …")
                    # Block until main thread calls resume() or abort() is called
                    while not self.isInterruptionRequested() and self._suspend_flag.is_set():
                        time.sleep(0.5)
                    if not self.isInterruptionRequested():
                        self.hold_cleared.emit()
                        print("  ✓ Hold cleared — restarting row from beginning")
                    held = True
                    break

                if not proc.is_alive():
                    # Drain any messages buffered just before the process exited
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

            # ── Drain & close queue ───────────────────────────────────────────
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
                continue   # restart same row_idx from the top

            if row_record:
                all_records.append(row_record)
                self.row_done.emit(dict(row_record))

            row_idx += 1

        self._current_proc = None
        self.done.emit(all_records)
