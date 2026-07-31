"""
Subprocess entry point for single-row alignment.
Must be a top-level module function for multiprocessing 'spawn' compatibility.
"""
import io
import time
import traceback


def _result_to_dict(result):
    """Serialise a ScanResult to a plain dict for queue transmission."""
    return {
        "status":    result.status.value,
        "center":    result.center,
        "sigma":     result.sigma,
        "amplitude": result.amplitude,
        "offset":    result.offset,
        "profile":   result.profile,
        "stats":     result.stats,
    }


def run_alignment_row(q, row, kwargs, simulate):
    """
    Run align_beamline for a single energy row, streaming progress via q.

    Queue message types emitted:
        ('log',            text)
        ('scan_started',   tab_name)        # 'BRG2' / 'Pitch' / 'Roll2' / 'X2'
        ('point',          x, y)            # coarse scan point
        ('fine_point',     x, y)            # fine scan point
        ('scan_finished',  result_dict)
        ('step',           label, current, total)
        ('row_done',       record_dict)     # sent after CSV row is written
        ('error',          traceback_str)
        ('subprocess_done',)               # always last, even on error
    """
    import contextlib
    from . import smart_scan_functions as _m
    from .smart_scan_functions import align_beamline

    # ── stdout → queue ───────────────────────────────────────────────────────
    class _QStream(io.TextIOBase):
        def write(self, text):
            if text:
                q.put(("log", text))
            return len(text)
        def flush(self):
            pass

    # ── motor PV → tab name ──────────────────────────────────────────────────
    _motor_tab = {
        kwargs.get("brg2",        ""): "BRG2",
        kwargs.get("roll2_motor", ""): "Roll2",
        kwargs.get("x2_motor",    ""): "X2",
    }

    def _tab_for(motor):
        from .smart_scan_functions import PVAxis
        if isinstance(motor, PVAxis):
            return "Pitch"
        return _motor_tab.get(str(motor), "BRG2")

    # ── patching state ───────────────────────────────────────────────────────
    orig_ss          = _m.smart_scan
    orig_fs          = _m.fly_scan
    orig_if_read     = _m._Interface.read
    orig_sample_loop = _m._sample_loop
    orig_fine_hook   = _m._fine_scan_start_hook

    _in_fine       = [False]
    _last_emit_pos = [None]

    def _if_read_emit(iface_self):
        sig = orig_if_read(iface_self)
        try:
            pos = float(iface_self.position())
        except Exception:
            pos = float(getattr(iface_self, "_pos", 0.0))
        last = _last_emit_pos[0]
        if last is None or abs(pos - last) > 1e-9:
            _last_emit_pos[0] = pos
            q.put(("fine_point" if _in_fine[0] else "point", pos, float(sig)))
        return sig

    def _sl_emit(iface, interval, pos_list, sig_list, stop_ev, verbose):
        while not stop_ev.is_set():
            t_next = time.monotonic() + interval
            p = iface.read_position()
            s = iface.read_detector()
            pos_list.append(p)
            sig_list.append(s)
            q.put(("point", float(p), float(s)))
            if verbose:
                print(f"  pos={p:>13.8g}   signal={s:>14.5g}")
            rem = t_next - time.monotonic()
            if rem > 0:
                time.sleep(rem)

    def _patched_ss(motor, det, start, stop, *args, **kw):
        _last_emit_pos[0] = None
        _in_fine[0] = False
        q.put(("scan_started", _tab_for(motor)))
        _m._Interface.read = _if_read_emit
        try:
            result = orig_ss(motor, det, start, stop, *args, **kw)
        finally:
            _m._Interface.read = orig_if_read
        q.put(("scan_finished", _result_to_dict(result)))
        return result

    def _patched_fs(motor, det, start, stop, *args, **kw):
        q.put(("scan_started", "Pitch"))
        _m._sample_loop = _sl_emit
        try:
            result = orig_fs(motor, det, start, stop, *args, **kw)
        finally:
            _m._sample_loop = orig_sample_loop
        q.put(("scan_finished", _result_to_dict(result)))
        return result

    _m.smart_scan            = _patched_ss
    _m.fly_scan              = _patched_fs
    _m._fine_scan_start_hook = lambda: _in_fine.__setitem__(0, True)

    # ── callbacks ────────────────────────────────────────────────────────────
    do_pitch    = kwargs.get("do_pitch_scan", True)
    total_steps = 11 if do_pitch else 8
    step_cnt    = [0]

    def _step_cb(label):
        step_cnt[0] += 1
        q.put(("step", label, step_cnt[0], total_steps))

    def _row_cb(record):
        q.put(("row_done", dict(record)))

    # ── run ──────────────────────────────────────────────────────────────────
    try:
        with contextlib.redirect_stdout(_QStream()):
            align_beamline(
                [row],
                simulate=simulate,
                verbose=True,
                debug=False,
                step_cb=_step_cb,
                row_cb=_row_cb,
                **kwargs,
            )
    except Exception:
        q.put(("error", traceback.format_exc()))
    finally:
        _m.smart_scan            = orig_ss
        _m.fly_scan              = orig_fs
        _m._Interface.read       = orig_if_read
        _m._sample_loop          = orig_sample_loop
        _m._fine_scan_start_hook = orig_fine_hook
        q.put(("subprocess_done",))
