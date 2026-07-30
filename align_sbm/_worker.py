"""Background worker thread for running alignment scans."""
import contextlib
import io
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
    log_chunk = pyqtSignal(str)
    scan_finished = pyqtSignal(object)   # ScanResult after each scan
    step_started = pyqtSignal(str)       # step label
    done = pyqtSignal(list)              # final results list
    error = pyqtSignal(str)              # traceback string

    def __init__(self, table, kwargs, simulate, parent=None):
        super().__init__(parent)
        self._table = table
        self._kwargs = kwargs
        self._simulate = simulate

    def run(self):
        from . import smart_scan_functions as _m

        # Wrap smart_scan and fly_scan to emit per-scan results
        orig_ss = _m.smart_scan
        orig_fs = _m.fly_scan

        def _patched_smart_scan(*a, **kw):
            res = orig_ss(*a, **kw)
            self.scan_finished.emit(res)
            return res

        def _patched_fly_scan(*a, **kw):
            res = orig_fs(*a, **kw)
            self.scan_finished.emit(res)
            return res

        _m.smart_scan = _patched_smart_scan
        _m.fly_scan = _patched_fly_scan

        stream = _LogStream(self.log_chunk)
        try:
            with contextlib.redirect_stdout(stream):
                from .smart_scan_functions import align_beamline
                results = align_beamline(
                    self._table,
                    simulate=self._simulate,
                    verbose=True,
                    debug=False,
                    **self._kwargs,
                )
            self.done.emit(results or [])
        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            _m.smart_scan = orig_ss
            _m.fly_scan = orig_fs
