"""Background worker thread for running alignment scans."""
import contextlib
import io
import time
import traceback

from PyQt6.QtCore import QThread, pyqtSignal

from .smart_scan_functions import ScanResult


class _LogStream(io.TextIOBase):
    def __init__(self, signal):
        super().__init__()
        self._sig = signal

    def write(self, text):
        if text:
            self._sig.emit(text)
        return len(text)

    def flush(self):
        pass


class AlignWorker(QThread):
    log_chunk      = pyqtSignal(str)
    scan_started   = pyqtSignal(str)            # tab name: BRG2 / Pitch / Roll2 / X2
    point_measured = pyqtSignal(float, float)   # (x, y) live data point
    scan_finished  = pyqtSignal(object)         # ScanResult after each scan completes
    step_update    = pyqtSignal(str, int, int)  # (label, current, total)
    row_done       = pyqtSignal(dict)           # record dict after each row completes
    done           = pyqtSignal(list)           # final results list
    error          = pyqtSignal(str)            # traceback string

    def __init__(self, table, kwargs, simulate, parent=None):
        super().__init__(parent)
        self._table    = table
        self._kwargs   = kwargs
        self._simulate = simulate

    def _tab_for_motor(self, motor):
        from .smart_scan_functions import PVAxis
        if isinstance(motor, PVAxis):
            return "Pitch"
        label = str(motor)
        for tab, key in [("BRG2", "brg2"), ("Roll2", "roll2_motor"), ("X2", "x2_motor")]:
            pv = self._kwargs.get(key, "")
            if pv and label == pv:
                return tab
        return "BRG2"

    def run(self):
        from . import smart_scan_functions as _m

        worker = self

        orig_ss          = _m.smart_scan
        orig_fs          = _m.fly_scan
        orig_if_read     = _m._Interface.read
        orig_sample_loop = _m._sample_loop

        # Patch _Interface.read to emit one point per smart_scan step.
        def _if_read_emit(iface_self):
            sig = orig_if_read(iface_self)
            try:
                pos = iface_self.position()   # works for both sim and real EPICS
            except Exception:
                pos = getattr(iface_self, "_pos", 0.0)
            worker.point_measured.emit(float(pos), float(sig))
            return sig

        # Patch _sample_loop to emit one point per fly_scan sample.
        def _sample_loop_emit(iface, sample_interval, pos_list, sig_list,
                               stop_event, verbose):
            while not stop_event.is_set():
                t_next = time.monotonic() + sample_interval
                p = iface.read_position()
                s = iface.read_detector()
                pos_list.append(p)
                sig_list.append(s)
                worker.point_measured.emit(float(p), float(s))
                if verbose:
                    print(f"  pos={p:>13.8g}   signal={s:>14.5g}")
                remaining = t_next - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)

        def _patched_ss(motor, det, start, stop, *args, **kw):
            tab = worker._tab_for_motor(motor)
            worker.scan_started.emit(tab)
            _m._Interface.read = _if_read_emit
            try:
                result = orig_ss(motor, det, start, stop, *args, **kw)
            finally:
                _m._Interface.read = orig_if_read
            worker.scan_finished.emit(result)
            return result

        def _patched_fs(motor, det, start, stop, *args, **kw):
            worker.scan_started.emit("Pitch")
            _m._sample_loop = _sample_loop_emit
            try:
                result = orig_fs(motor, det, start, stop, *args, **kw)
            finally:
                _m._sample_loop = orig_sample_loop
            worker.scan_finished.emit(result)
            return result

        _m.smart_scan = _patched_ss
        _m.fly_scan   = _patched_fs

        # Progress callbacks
        do_pitch = self._kwargs.get("do_pitch_scan", True)
        steps_per_row = 11 if do_pitch else 8
        total_steps = max(len(self._table) * steps_per_row, 1)
        step_counter = [0]

        def _step_cb(label):
            step_counter[0] += 1
            worker.step_update.emit(label, step_counter[0], total_steps)

        def _row_cb(record):
            worker.row_done.emit(dict(record))

        stream = _LogStream(self.log_chunk)
        try:
            with contextlib.redirect_stdout(stream):
                from .smart_scan_functions import align_beamline
                results = align_beamline(
                    self._table,
                    simulate=self._simulate,
                    verbose=True,
                    debug=False,
                    step_cb=_step_cb,
                    row_cb=_row_cb,
                    **self._kwargs,
                )
            self.done.emit(results or [])
        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            _m.smart_scan      = orig_ss
            _m.fly_scan        = orig_fs
            _m._Interface.read = orig_if_read
            _m._sample_loop    = orig_sample_loop
