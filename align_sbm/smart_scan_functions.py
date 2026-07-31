"""
smart_scan_functions.py  (modified)
====================================
Key change: after each scan the motor moves to a position chosen by
_choose_centre():
  • If FWHM > |peak_pos − centroid|  →  move to peak_pos
  • Otherwise                         →  move to centroid
"""

import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

try:
    import epics
    _EPICS_AVAILABLE = True
except ImportError:
    _EPICS_AVAILABLE = False
    warnings.warn("pyepics not found – SIMULATION mode active.", stacklevel=1)


def caget(pv_name: str, timeout: float = 5.0):
    """Read the current value of an EPICS PV. Returns None if unavailable."""
    if not _EPICS_AVAILABLE or not isinstance(pv_name, str) or not pv_name.strip():
        return None
    try:
        return epics.caget(pv_name.strip(), timeout=timeout)
    except Exception:
        return None


def caput(pv_name: str, value, wait: bool = True, timeout: float = 10.0):
    """Write *value* to an EPICS PV. No-op if EPICS is unavailable."""
    if not _EPICS_AVAILABLE or not isinstance(pv_name, str) or not pv_name.strip():
        return
    try:
        epics.caput(pv_name.strip(), value, wait=wait, timeout=timeout)
    except Exception:
        pass


def create_pv_monitor(pv_name: str, callback):
    """Subscribe to a PV via CA monitor.

    *callback(value)* is called from the CA thread on connection and on every
    value change.  Returns an ``epics.PV`` handle (call ``pv.clear_callbacks()``
    to unsubscribe) or ``None`` when EPICS is unavailable.
    """
    if not _EPICS_AVAILABLE or not isinstance(pv_name, str) or not pv_name.strip():
        return None
    try:
        def _cb(pvname=None, value=None, **kw):
            callback(value)
        pv = epics.PV(pv_name.strip(), callback=_cb, auto_monitor=True)
        return pv
    except Exception:
        return None


# Patchable hook: called once when smart_scan transitions to its fine-scan phase.
# AlignWorker sets this to differentiate coarse vs fine points in the live plot.
_fine_scan_start_hook = None


def _eval_condition(actual, op: str, value_str: str) -> bool:
    """Return True if the condition passes (PV satisfies the constraint).

    Tries numeric comparison first; falls back to string equality for
    string-valued PVs or when the threshold cannot be parsed as a float.
    Quotes around *value_str* are stripped before comparison.
    """
    value_str = str(value_str).strip().strip("\"'")
    try:
        a_f = float(actual)
        v_f = float(value_str)
        return {
            ">":  a_f >  v_f,
            "<":  a_f <  v_f,
            ">=": a_f >= v_f,
            "<=": a_f <= v_f,
            "==": a_f == v_f,
            "!=": a_f != v_f,
        }.get(op, True)
    except (ValueError, TypeError):
        pass
    a_s = str(actual).strip()
    if op == "==":
        return a_s == value_str
    if op == "!=":
        return a_s != value_str
    return True   # numeric ops on non-numeric values → pass (don't hold)

# ── Result / status types ────────────────────────────────────────────────────

class ScanStatus(Enum):
    SUCCESS          = "success"
    NO_PEAK          = "no_peak"
    OUT_OF_RANGE     = "out_of_range"
    FIT_FAILED       = "fit_failed"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class ScanResult:
    status             : ScanStatus
    positions          : np.ndarray
    signals            : np.ndarray
    center             : Optional[float] = None
    sigma              : Optional[float] = None
    amplitude          : Optional[float] = None
    offset             : Optional[float] = None
    profile            : str = "gaussian"
    stats              : Optional[dict] = None
    message            : str = ""

    def __str__(self):
        lines = [f"Status  : {self.status.value}", f"Message : {self.message}"]
        if self.center is not None:
            lines += [
                f"Centre  : {self.center:.5g}",
                f"Sigma   : {self.sigma:.5g}",
                f"Amplitude: {self.amplitude:.5g}",
                f"Offset  : {self.offset:.5g}",
            ]
        return "\n".join(lines)


def _safe_label(obj) -> str:
    if isinstance(obj, str):
        return obj
    if obj is None:
        return 'None'
    try:
        v = object.__getattribute__(obj, '_prefix')
        if v:
            return str(v)
    except AttributeError:
        pass
    try:
        v = object.__getattribute__(obj, 'pvname')
        if v:
            return str(v)
    except AttributeError:
        pass
    return repr(obj)


# ── Peak profile models ──────────────────────────────────────────────────────

def _gaussian(x, amplitude, center, sigma, offset):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + offset


def _lorentzian(x, amplitude, center, gamma, offset):
    return amplitude / (1.0 + ((x - center) / gamma) ** 2) + offset


def _supergaussian(x, amplitude, center, sigma, p, offset):
    """
    Super-Gaussian (generalised Gaussian).
    p=2  → standard Gaussian
    p>2  → flat-topped; larger p = flatter top
    p→∞  → top-hat

    Fitted as a 5-parameter model; width reported as the half-width at 1/e
    of the amplitude (same meaning as sigma for p=2).
    """
    return amplitude * np.exp(-np.abs((x - center) / sigma) ** p) + offset


def _supergaussian_sigma_to_fwhm(sigma, p):
    """Convert super-Gaussian sigma (half-width at 1/e) to FWHM."""
    # FWHM = 2 * sigma * (ln2)^(1/p)
    return 2.0 * sigma * (np.log(2.0) ** (1.0 / p))


# Module-level store: the p exponent from the most recent super-Gaussian fit.
# Populated by _try_fit; read by smart_scan to annotate ScanResult.stats.
_last_supergaussian_p: float = 2.0


def _smooth(y: np.ndarray) -> np.ndarray:
    n = len(y)
    if n < 5:
        return y.copy()
    window = min((n // 3) * 2 + 1, 11)
    if window % 2 == 0:
        window += 1
    try:
        return savgol_filter(y, window_length=window, polyorder=2)
    except Exception:
        return y.copy()


def _seed_params(positions, y_s):
    offset0 = np.percentile(y_s, 10)
    amp0    = y_s.max() - offset0
    center0 = positions[np.argmax(y_s)]
    above   = positions[y_s >= offset0 + amp0 / 2.0]
    width0  = (above[-1] - above[0]) / 2.0 if len(above) > 1 \
              else abs(positions[-1] - positions[0]) / 4.0
    return offset0, amp0, center0, max(abs(width0), 1e-9)


def _residual_rms(func, popt, positions, y_s):
    return float(np.sqrt(np.mean((y_s - func(positions, *popt)) ** 2)))


def _try_fit(positions: np.ndarray, signals: np.ndarray,
             mode: str,
             profile: str = "auto") -> tuple:
    """
    Fit a peak/valley profile to (positions, signals).

    profile : "gaussian"      – force Gaussian
              "lorentzian"    – force Lorentzian
              "supergaussian" – force super-Gaussian (best for flat-topped peaks)
              "auto"          – fit all three, return whichever has lowest RMS residual

    Returns (success, center, width, amplitude, offset, profile_used).
    For Gaussian/Lorentzian, width is sigma/gamma.
    For super-Gaussian, width is the sigma parameter (half-width at 1/e amplitude).
    """
    y = signals.copy()
    if mode == "min":
        y = -y

    y_s                        = _smooth(y)
    offset0, amp0, center0, w0 = _seed_params(positions, y_s)

    # off_min: lower bound for offset.  Use raw-signal 5th-percentile as a
    # robust baseline, clamped to a small positive floor so the bound is > 0.
    raw_baseline = float(np.percentile(y, 5))
    off_min      = max(raw_baseline * 0.5, 1e-6)

    # Clamp offset seed so p0 is always inside bounds; recompute amp0 consistently.
    offset0 = max(float(offset0), off_min)
    amp0    = max(y_s.max() - offset0, 1e-6)

    span          = abs(positions[-1] - positions[0])
    bounds_common = ([0,      positions.min() - span, 0,    off_min],
                     [np.inf, positions.max() + span, span, np.inf ])

    results = {}

    if profile in ("gaussian", "auto"):
        try:
            popt, _ = curve_fit(
                _gaussian, positions, y_s,
                p0=[amp0, center0, w0, offset0],
                bounds=bounds_common, maxfev=10_000,
            )
            results["gaussian"] = (popt, _residual_rms(_gaussian, popt, positions, y_s))
        except (RuntimeError, ValueError):
            pass

    if profile in ("lorentzian", "auto"):
        try:
            popt, _ = curve_fit(
                _lorentzian, positions, y_s,
                p0=[amp0, center0, w0, offset0],
                bounds=bounds_common, maxfev=10_000,
            )
            results["lorentzian"] = (popt, _residual_rms(_lorentzian, popt, positions, y_s))
        except (RuntimeError, ValueError):
            pass

    if profile in ("supergaussian", "auto"):
        # Super-Gaussian is fitted to RAW signals (not smoothed) because
        # the smoother rounds off the flat top, hiding the very shape
        # that makes the super-Gaussian a better model.
        bounds_sg = ([0,      positions.min() - span, 0,    1.5,  off_min],
                     [np.inf, positions.max() + span, span, 20.0, np.inf ])
        # Re-seed from raw signal for a better starting point
        raw_offset0 = max(float(np.percentile(y, 5)), off_min)
        raw_amp0    = max(float(y.max()) - raw_offset0, 1e-6)
        raw_center0 = positions[int(np.argmax(y))]
        above_raw   = positions[y >= raw_offset0 + raw_amp0 / 2.0]
        raw_w0      = ((above_raw[-1] - above_raw[0]) / 2.0
                       if len(above_raw) > 1 else span / 4.0)
        raw_w0      = max(abs(raw_w0), 1e-9)
        # Multi-start: try several p seeds so the solver doesn't get stuck at p≈2
        _best_sg_popt, _best_sg_rms = None, np.inf
        for _p_seed in [2.0, 4.0, 8.0, 16.0]:
            try:
                _popt, _ = curve_fit(
                    _supergaussian, positions, y,
                    p0=[raw_amp0, raw_center0, raw_w0, _p_seed, raw_offset0],
                    bounds=bounds_sg, maxfev=20_000,
                )
                _rms = float(np.sqrt(np.mean((y - _supergaussian(positions, *_popt)) ** 2)))
                if _rms < _best_sg_rms:
                    _best_sg_rms  = _rms
                    _best_sg_popt = _popt
            except (RuntimeError, ValueError):
                pass
        if _best_sg_popt is not None:
            results["supergaussian"] = (_best_sg_popt, _best_sg_rms)

    if not results:
        return False, 0.0, 0.0, 0.0, 0.0, "none"

    # For "auto", compare all models on a common basis (raw signal RMS).
    # Re-evaluate Gaussian and Lorentzian on raw signal so the comparison is fair.
    if profile == "auto" and len(results) > 1:
        def _raw_rms(name, popt_val):
            if name == "gaussian":
                return float(np.sqrt(np.mean((y - _gaussian(positions, *popt_val)) ** 2)))
            elif name == "lorentzian":
                return float(np.sqrt(np.mean((y - _lorentzian(positions, *popt_val)) ** 2)))
            else:  # supergaussian already on raw
                return results[name][1]
        best_profile = min(results, key=lambda k: _raw_rms(k, results[k][0]))
    else:
        best_profile = min(results, key=lambda k: results[k][1])

    popt, _ = results[best_profile]

    if best_profile == "supergaussian":
        amp, cen, width, p_val, off = popt
        global _last_supergaussian_p
        _last_supergaussian_p = float(p_val)
    else:
        amp, cen, width, off = popt

    if mode == "min":
        amp = -amp

    return True, float(cen), float(width), float(amp), float(off), best_profile


# ── Model-free statistical peak estimators ───────────────────────────────────

def stats_peak(positions: np.ndarray, signals: np.ndarray,
               mode: str = "max",
               baseline: Optional[float] = None) -> dict:
    pos = np.asarray(positions, dtype=float)
    sig = np.asarray(signals,   dtype=float)

    if mode == "min":
        sig = -sig

    if baseline is None:
        base = np.percentile(sig, 10)
    else:
        base = float(baseline) if mode == "max" else -float(baseline)

    y = np.clip(sig - base, 0, None)

    result = {}

    peak_idx        = int(np.argmax(sig))
    result["peak_pos"] = float(pos[peak_idx])
    result["peak_val"] = float(signals[peak_idx])

    total_weight = y.sum()
    if total_weight <= 0:
        for k in ("centroid", "rms_width", "fwhm_interp",
                  "fwhm_width", "weighted_median"):
            result[k] = np.nan
        return result

    centroid = float(np.sum(pos * y) / total_weight)
    result["centroid"] = centroid

    variance = float(np.sum(y * (pos - centroid) ** 2) / total_weight)
    result["rms_width"] = float(np.sqrt(variance)) if variance >= 0 else np.nan

    half_max = y.max() / 2.0
    diff = y - half_max
    crossings = []
    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] <= 0:
            frac = diff[i] / (diff[i] - diff[i + 1])
            crossings.append(pos[i] + frac * (pos[i + 1] - pos[i]))

    if len(crossings) >= 2:
        fwhm = abs(crossings[-1] - crossings[0])
        result["fwhm_interp"] = float(fwhm)
        result["fwhm_width"]  = float(fwhm / 2.355)
    else:
        result["fwhm_interp"] = np.nan
        result["fwhm_width"]  = np.nan

    cum = np.cumsum(y)
    cum /= cum[-1]
    idx = np.searchsorted(cum, 0.5)
    if idx == 0:
        result["weighted_median"] = float(pos[0])
    elif idx >= len(pos):
        result["weighted_median"] = float(pos[-1])
    else:
        frac = (0.5 - cum[idx - 1]) / (cum[idx] - cum[idx - 1])
        result["weighted_median"] = float(pos[idx - 1] + frac * (pos[idx] - pos[idx - 1]))

    return result


def print_stats(s: dict) -> None:
    print(f"  Peak position    : {s['peak_pos']:.6g}   (signal={s['peak_val']:.5g})")
    print(f"  Centroid         : {s['centroid']:.6g}")
    print(f"  Weighted median  : {s['weighted_median']:.6g}")
    print(f"  RMS width (σ)    : {s['rms_width']:.5g}")
    print(f"  FWHM (interp)    : {s['fwhm_interp']:.5g}   → σ-equiv = {s['fwhm_width']:.5g}")


# ── Peak-detection heuristics ────────────────────────────────────────────────

def _has_peak_character(signals: np.ndarray, mode: str,
                        min_prominence_ratio: float) -> bool:
    """
    Return True when the data looks like it contains a peak (or dip).

    If False, smart_scan falls back to the raw maximum position from stats
    rather than failing completely.
    """
    if len(signals) < 3:
        return False
    y = signals if mode == "max" else -signals
    y_s = _smooth(y)
    signal_range = y_s.max() - y_s.min()
    diffs = np.abs(np.diff(y))
    noise = np.median(diffs) if len(diffs) else 1e-9
    noise = max(noise, 1e-12)
    return signal_range >= min_prominence_ratio * noise


def _peak_near_edge(positions: np.ndarray, signals: np.ndarray,
                    mode: str, edge_fraction: float,
                    scan_descending: bool = False) -> Optional[str]:
    y      = _smooth(signals if mode == "max" else -signals)
    idx    = int(np.argmax(y))
    n      = len(positions)
    margin = max(1, int(np.ceil(edge_fraction * n)))
    if idx < margin:
        return "start"
    if idx >= n - margin:
        return "end"
    return None


def _data_sufficient(positions: np.ndarray, signals: np.ndarray,
                     mode: str, sigma_coverage: float) -> bool:
    ok, cen, sig, _, _, _ = _try_fit(positions, signals, mode)
    if not ok or sig <= 0:
        return False
    left_cover  = (cen - positions.min()) / sig
    right_cover = (positions.max() - cen) / sig
    return left_cover >= sigma_coverage and right_cover >= sigma_coverage


# =============================================================================
# _choose_centre  – NEW helper
# =============================================================================

def _choose_centre(result: "ScanResult", verbose: bool = True,
                   move_target: str = "auto") -> float:
    """
    Decide where to move the motor after a successful scan.

    Parameters
    ----------
    result      : ScanResult with .stats populated.
    verbose     : Print the decision including which estimator was used.
    move_target : "auto"     – apply the FWHM vs displacement rule (default).
                  "peak_pos" – always move to the raw intensity maximum.
                  "centroid" – always move to the intensity-weighted centroid.

    Rule for "auto"
    ---------------
    Let  D    = |peak_pos − centroid|
    Let  FWHM = fwhm_interp  (preferred)
                or rms_width × 2.355  (fallback when fwhm_interp is nan,
                e.g. the scan starts/ends above the half-maximum so only
                one half-max crossing exists in the data)

    • FWHM > D  →  move to centroid   (peak is broad relative to peak/centroid offset;
                                        centroid averages over the full width)
    • FWHM ≤ D  →  move to peak_pos   (peak is sharp; the raw maximum is most precise)

    Falls back to result.center if the requested estimator is unavailable.

    Returns
    -------
    float – chosen motor target position.
    """
    s = result.stats

    def _ok(v):
        return v is not None and not np.isnan(float(v))

    # ── Explicit overrides ────────────────────────────────────────────────────
    if move_target == "peak_pos":
        val = (s.get("peak_pos") if s else None)
        if _ok(val):
            if verbose:
                print(f"  Centre choice: forced peak_pos ({float(val):.6g})")
            return float(val)
        fallback = result.center
        if verbose:
            print(f"  Centre choice: peak_pos unavailable – using fitted centre {fallback:.6g}")
        return fallback

    if move_target == "centroid":
        val = (s.get("centroid") if s else None)
        if _ok(val):
            if verbose:
                print(f"  Centre choice: forced centroid ({float(val):.6g})")
            return float(val)
        # centroid not available – fall back to peak_pos then fitted centre
        fallback_val = (s.get("peak_pos") if s else None)
        fallback = float(fallback_val) if _ok(fallback_val) else result.center
        if verbose:
            print(f"  Centre choice: centroid unavailable – using {fallback:.6g}")
        return fallback

    # ── "auto" mode: FWHM vs displacement rule ────────────────────────────────
    if s is None:
        if verbose:
            print("  _choose_centre: no stats – falling back to fitted centre")
        return result.center

    peak_pos = s.get("peak_pos")
    centroid = s.get("centroid")
    fwhm     = s.get("fwhm_interp")
    rms      = s.get("rms_width")

    if not (_ok(peak_pos) and _ok(centroid)):
        fallback = result.center if result.center is not None else float(peak_pos or 0)
        if verbose:
            print(f"  _choose_centre: peak_pos/centroid unavailable – using fitted centre {fallback:.6g}")
        return fallback

    peak_pos = float(peak_pos)
    centroid = float(centroid)

    # Prefer fwhm_interp; fall back to rms_width × 2.355 (Gaussian FWHM equivalent).
    # fwhm_interp is nan when the scan starts or ends above the half-maximum level
    # (only one crossing is found).  rms_width is always computable.
    if _ok(fwhm):
        fwhm_used   = float(fwhm)
        fwhm_source = "fwhm_interp"
    elif _ok(rms):
        fwhm_used   = float(rms) * 2.355
        fwhm_source = "rms_width×2.355 (fwhm_interp unavailable – scan clipped at half-max)"
    else:
        fallback = result.center if result.center is not None else peak_pos
        if verbose:
            print(f"  _choose_centre: no width estimator available – using fitted centre {fallback:.6g}")
        return fallback

    D = abs(peak_pos - centroid)

    if fwhm_used > D:
        chosen = centroid
        reason = (f"FWHM ({fwhm_source}={fwhm_used:.5g}) > |peak−centroid| ({D:.5g}) "
                  f"→ moving to centroid ({centroid:.6g})")
    else:
        chosen = peak_pos
        reason = (f"FWHM ({fwhm_source}={fwhm_used:.5g}) ≤ |peak−centroid| ({D:.5g}) "
                  f"→ moving to peak_pos ({peak_pos:.6g})")

    if verbose:
        print(f"  Centre choice: {reason}")

    return chosen



def _backlash_fwhm(result: "ScanResult") -> float:
    """
    Return the FWHM of the peak from stats, used as the backlash overshoot
    distance.  Falls back to rms_width * 2.355, then sigma, then 0.
    """
    s = result.stats
    if s:
        fwhm = s.get("fwhm_interp")
        if fwhm is not None and not np.isnan(float(fwhm)):
            return float(fwhm)
        rms = s.get("rms_width")
        if rms is not None and not np.isnan(float(rms)):
            return float(rms) * 2.355
    if result.sigma is not None and result.sigma > 0:
        return float(result.sigma) * 2.355
    return 0.0


def _move_with_backlash(iface, target: float, scan_descending: bool,
                        backlash: float, motor_timeout: float,
                        verbose: bool) -> None:
    """
    Move to *target* with a backlash correction overshoot of *backlash* units.

    The final approach is always made in the same direction as the scan:
    • scan_descending=True  (high→low): approach from above → overshoot to
                             target + backlash, then come back down to target.
    • scan_descending=False (low→high): approach from below → overshoot to
                             target - backlash, then come back up to target.

    If backlash ≤ 0 the move is made directly with no overshoot.
    """
    if backlash <= 0:
        iface.move(target, timeout=motor_timeout)
        return

    if scan_descending:
        overshoot = target + backlash   # go high first, then come down
    else:
        overshoot = target - backlash   # go low first, then come up

    if verbose:
        print(f"  Backlash correction: overshoot to {overshoot:.6g} "
              f"then approach {target:.6g} from "
              f"{'above' if scan_descending else 'below'} (FWHM={backlash:.4g})")
    iface.move(overshoot, timeout=motor_timeout)
    iface.move(target,    timeout=motor_timeout)


# =============================================================================
# PVAxis
# =============================================================================

class PVAxis:
    def __init__(self,
                 setpoint_pv,
                 readback_pv  = None,
                 settle_tol   : float = 0.001,
                 settle_time  : float = 0.2,
                 settle_checks: int   = 3,
                 soft_limits  : tuple = (-1e9, 1e9),
                 timeout      : float = 60.0):

        self._tol     = settle_tol
        self._st      = settle_time
        self.settle_time = settle_time
        self._checks  = settle_checks
        self._limits  = soft_limits
        self._timeout = timeout
        self._sim     = False

        if not _EPICS_AVAILABLE:
            raise RuntimeError("pyepics is required to use PVAxis.")

        if isinstance(setpoint_pv, str):
            self._sp = epics.PV(setpoint_pv)
            if not self._sp.connect(timeout=20):
                raise ConnectionError(f"Setpoint PV not connected: {setpoint_pv}")
        else:
            self._sp = setpoint_pv

        if readback_pv is None:
            self._rbv = self._sp
        elif isinstance(readback_pv, str):
            self._rbv = epics.PV(readback_pv)
            if not self._rbv.connect(timeout=20):
                raise ConnectionError(f"Readback PV not connected: {readback_pv}")
        else:
            self._rbv = readback_pv

        self.pvname = _safe_label(self._sp)

    def put(self, value: float, wait: bool = False) -> None:
        self._sp.put(value, wait=wait)
        epics.ca.flush_io()

    def get_position(self) -> float:
        v = self._rbv.get(use_monitor=False)
        return float(v) if v is not None else float(self._sp.get())

    def move(self, target: float, timeout: Optional[float] = None) -> bool:
        tmo = timeout if timeout is not None else self._timeout
        self.put(target)
        t0 = time.monotonic()
        ok_count = 0
        while time.monotonic() - t0 < tmo:
            time.sleep(0.05)
            rbv = self.get_position()
            if abs(rbv - target) < self._tol:
                ok_count += 1
                if ok_count >= self._checks:
                    if self._st > 0:
                        time.sleep(self._st)
                    return True
            else:
                ok_count = 0
        return False

    def is_moving(self) -> bool:
        sp = self._sp.get(use_monitor=False)
        rbv = self.get_position()
        return abs(rbv - sp) >= self._tol if sp is not None else False

    def soft_limits(self) -> tuple:
        return self._limits

    def get_velocity(self) -> float:
        return float("nan")

    def set_velocity(self, v: float) -> None:
        pass

    def get_base_velocity(self) -> float:
        return float("nan")

    def set_base_velocity(self, v: float) -> None:
        pass


# ── EPICS / simulation interface ─────────────────────────────────────────────

class _Interface:
    def __init__(self, motor, det,
                 simulate: bool, sim_center: float,
                 sim_sigma: float, sim_amp: float,
                 sim_offset: float, sim_noise: float):
        self._sim   = simulate or not _EPICS_AVAILABLE
        self._pos   = 0.0
        self._pvaxis = None
        self._motor  = None
        self._sim_c = sim_center
        self._sim_s = sim_sigma
        self._sim_a = sim_amp
        self._sim_o = sim_offset
        self._sim_n = sim_noise

        if not self._sim:
            if isinstance(motor, PVAxis):
                self._pvaxis = motor
                self._motor  = None
            else:
                self._pvaxis = None
                if isinstance(motor, str):
                    print(f"    [DEBUG] Creating epics.Motor({motor!r})")
                    self._motor = epics.Motor(motor)
                    print(f"    [DEBUG] Motor type: {type(self._motor).__name__}")
                    try:
                        _conn = self._motor.connected
                        print(f"    [DEBUG] Motor.connected = {_conn}")
                    except Exception as _ae:
                        print(f"    [DEBUG] Motor.connected raised {type(_ae).__name__}: {_ae} — skipping check")
                        _conn = True
                    if not _conn:
                        raise ConnectionError(f"Motor PV not connected: {motor}")
                else:
                    print(f"    [DEBUG] Motor passed as object: {type(motor).__name__}")
                    self._motor = motor

            if isinstance(det, str):
                self._det = epics.PV(det, auto_monitor=False)
                if not self._det.connect(timeout=5):
                    raise ConnectionError(f"Detector PV not connected: {det}")
            else:
                self._det = det

    def move(self, pos: float, timeout: float = 60.0) -> bool:
        if self._sim:
            self._pos = pos
            return True
        if self._pvaxis is not None:
            return self._pvaxis.move(pos, timeout=timeout)
        self._motor.move(pos, wait=False)
        time.sleep(0.15)
        t0 = time.monotonic()
        while True:
            try:
                dmov = self._motor.PV("DMOV").get(use_monitor=False)
                done = (dmov == 1)
            except Exception:
                done = True
            if done:
                break
            if time.monotonic() - t0 > timeout:
                return False
            time.sleep(0.05)
        return True

    def position(self) -> float:
        if self._sim:
            return self._pos
        if self._pvaxis is not None:
            return self._pvaxis.get_position()
        return self._motor.get_position()

    def read(self) -> float:
        if self._sim:
            return (self._sim_a
                    * np.exp(-0.5 * ((self._pos - self._sim_c) / self._sim_s) ** 2)
                    + self._sim_o
                    + np.random.normal(0, self._sim_n))
        val = self._det.get(use_monitor=False)
        if val is None:
            raise RuntimeError("Detector PV returned None.")
        return float(val)


# =============================================================================
# smart_scan
# =============================================================================

def smart_scan(
    motor,
    det,
    start               : float,
    stop                : float,
    nsteps              : int   = 21,
    mode                : str   = "max",
    settle              : float = 0.1,
    det_update_interval : float = 0.2,
    move_to_peak        : bool  = True,
    move_target         : str   = "auto",  # "auto" | "peak_pos" | "centroid"
    backlash_correction : bool  = False,   # overshoot by FWHM then return
    motor_timeout       : float = 60.0,
    min_prominence_ratio: float = 5.0,
    edge_fraction       : float = 0.2,
    sigma_coverage      : float = 2.0,
    max_extend_steps    : int   = 30,
    extend_step_size    : Optional[float] = None,
    peak_method         : str   = "stats",
    stats_centre        : str   = "centroid",
    fit_profile         : str   = "auto",
    fine_scan           : bool  = True,
    fine_sigma_range    : float = 3.0,
    fine_nsteps         : int   = 21,
    fine_scan_iter      : int   = 2,
    plot                : bool  = False,
    simulate            : bool  = False,
    sim_center          : float = 1.5,
    sim_sigma           : float = 1.2,
    sim_amplitude       : float = 100.0,
    sim_offset          : float = 10.0,
    sim_noise           : float = 2.0,
    verbose             : bool  = True,
    debug               : bool  = False,
) -> ScanResult:
    """
    Smart EPICS motor scan with automatic peak/valley finding.

    Motor homing after each scan uses _choose_centre():
      • FWHM > |peak_pos − centroid|  →  move to peak_pos
      • FWHM ≤ |peak_pos − centroid|  →  move to centroid
    """
    if nsteps < 5:
        raise ValueError("nsteps must be ≥ 5 for a reliable fit.")
    if mode not in ("max", "min"):
        raise ValueError("mode must be 'max' or 'min'.")

    motor_label = _safe_label(motor)
    det_label   = _safe_label(det)

    if debug:
        verbose = False

    iface = _Interface(
        motor, det,
        simulate=simulate,
        sim_center=sim_center, sim_sigma=sim_sigma,
        sim_amp=sim_amplitude, sim_offset=sim_offset, sim_noise=sim_noise,
    )

    current_pos = iface.position()
    abs_start   = current_pos + start
    abs_stop    = current_pos + stop

    if verbose:
        print(f"  Current position : {current_pos:.5g}")
        print(f"  Relative offsets : start={start:+g}  stop={stop:+g}")
        print(f"  Absolute targets : {abs_start:.5g} → {abs_stop:.5g}")

    start, stop = abs_start, abs_stop

    step_size = abs(stop - start) / (nsteps - 1)
    if extend_step_size is None:
        extend_step_size = step_size

    pos_data: list = []
    sig_data: list = []

    def _fresh_read(active_sig_data: list, timeout: float = 2.0) -> float:
        if iface._sim:
            return iface.read()
        last = active_sig_data[-1] if active_sig_data else None
        t0 = time.monotonic()
        prev = iface.read()
        while time.monotonic() - t0 < timeout:
            time.sleep(det_update_interval)
            curr = iface.read()
            if last is None or (curr != prev and curr != last):
                return curr
            prev = curr
        return iface.read()

    def _acquire(target: float) -> Optional[float]:
        ok = iface.move(target, timeout=motor_timeout)
        if not ok:
            return None
        if settle > 0:
            time.sleep(settle)
        actual = iface.position()
        if pos_data and abs(actual - pos_data[-1]) < step_size * 0.1:
            if verbose:
                print(f"  pos={actual:>13.8g}   [skipped – same encoder position]")
            return 0.0
        signal = _fresh_read(sig_data)
        pos_data.append(actual)
        sig_data.append(signal)
        if verbose:
            print(f"  pos={actual:>13.8g}   signal={signal:>14.5g}")
        return signal

    # ── Phase 1: initial sweep ────────────────────────────────────────────────
    if verbose:
        tag = " [SIM]" if (simulate or not _EPICS_AVAILABLE) else ""
        print(f"=== Smart Scan{tag}  motor={motor_label}  det={det_label}  mode={mode} ===")
        print(f"    Initial sweep: {start} → {stop}  ({nsteps} steps)")

    if not iface._sim:
        _start_target = np.linspace(start, stop, nsteps)[0]
        iface.move(_start_target, timeout=motor_timeout)
        if verbose:
            print(f"  [Flushing detector at start position…]")
        iface.read()
        time.sleep(det_update_interval)
        iface.read()

    for target in np.linspace(start, stop, nsteps):
        if _acquire(target) is None:
            return ScanResult(
                status=ScanStatus.OUT_OF_RANGE,
                positions=np.array(pos_data),
                signals=np.array(sig_data),
                message="Motor timed out during initial sweep.",
                stats=stats_peak(np.array(pos_data), np.array(sig_data), mode=mode) if pos_data else None,
            )

    positions = np.array(pos_data)
    signals   = np.array(sig_data)

    scan_descending = stop < start

    def _dedup(pos, sig):
        order = np.argsort(pos)
        pos, sig = pos[order], sig[order]
        _, unique_idx = np.unique(np.round(pos, 4), return_index=True)
        pos, sig = pos[unique_idx], sig[unique_idx]
        if scan_descending:
            pos, sig = pos[::-1], sig[::-1]
        return pos, sig

    positions, signals = _dedup(positions, signals)
    if verbose:
        print(f"  ({len(positions)} unique positions after deduplication)")

    # ── Phase 2: check for peak ───────────────────────────────────────────────
    if not _has_peak_character(signals, mode, min_prominence_ratio):
        # No clear peak above noise floor — fall back to the raw maximum
        # position from stats rather than failing completely.
        s_fallback = stats_peak(positions, signals, mode=mode)
        fallback_pos = s_fallback.get("peak_pos")
        msg = (f"No {'peak' if mode=='max' else 'dip'} detected "
               f"(signal variation below {min_prominence_ratio}× noise floor) "
               f"– using raw {'maximum' if mode=='max' else 'minimum'} at "
               f"{fallback_pos:.6g}.")
        if verbose:
            print(f"\n⚠ {msg}")
        cen_fb  = float(fallback_pos)
        sig_fb  = s_fallback.get("rms_width") or abs(stop - start) / 4.0
        if sig_fb is None or np.isnan(sig_fb):
            sig_fb = abs(stop - start) / 4.0
        result = ScanResult(
            status=ScanStatus.SUCCESS,
            positions=positions, signals=signals,
            center=cen_fb, sigma=float(sig_fb),
            amplitude=float(s_fallback.get("peak_val", 0.0)),
            offset=0.0,
            profile="stats",
            message=msg,
            stats=s_fallback,
        )
        if move_to_peak:
            if verbose:
                print(f"  → Moving to raw {'max' if mode=='max' else 'min'}: {cen_fb:.6g}")
            backlash_fb = _backlash_fwhm(result) if backlash_correction else 0.0
            _move_with_backlash(iface, cen_fb, scan_descending,
                                backlash_fb, motor_timeout, verbose)
        if plot:
            _plot(result, motor_label, det_label, mode,
                  title_prefix="Coarse Scan (no peak – raw max)")
        if verbose:
            print(result)
        return result

    # ── Phase 3: extend if near edge ─────────────────────────────────────────
    extend_count = 0
    while True:
        positions, signals = _dedup(np.array(pos_data), np.array(sig_data))
        edge = _peak_near_edge(positions, signals, mode, edge_fraction,
                               scan_descending=scan_descending)
        if edge is None:
            break

        if extend_count >= max_extend_steps:
            msg = (f"Reached max_extend_steps={max_extend_steps} "
                   f"without fully capturing the {'peak' if mode=='max' else 'dip'}.")
            if verbose:
                print(f"\n✗ {msg}")
            result = ScanResult(
                status=ScanStatus.INSUFFICIENT_DATA,
                positions=positions, signals=signals,
                message=msg,
                stats=stats_peak(np.array(pos_data), np.array(sig_data), mode=mode) if pos_data else None,
            )
            if plot:
                _plot(result, motor_label, det_label, mode)
            return result

        if scan_descending:
            if edge == "start":
                next_pos = positions.max() + extend_step_size
            else:
                next_pos = positions.min() - extend_step_size
        else:
            if edge == "start":
                next_pos = positions.min() - extend_step_size
            else:
                next_pos = positions.max() + extend_step_size

        if verbose and extend_count == 0:
            print(f"\n  Peak near '{edge}' of scan range – extending scan...")

        last_pos = np.array(pos_data)[-1] if pos_data else None
        if last_pos is not None and abs(next_pos - last_pos) < step_size * 0.1:
            if verbose:
                print(f"  ⚠ Motor resolution limit reached – using data collected so far.")
            break

        result = _acquire(next_pos)
        if result is None:
            msg = f"Motor limit reached while extending toward '{edge}' edge."
            if verbose:
                print(f"\n✗ {msg}")
            result = ScanResult(
                status=ScanStatus.OUT_OF_RANGE,
                positions=np.array(pos_data), signals=np.array(sig_data),
                message=msg,
                stats=stats_peak(np.array(pos_data), np.array(sig_data), mode=mode) if pos_data else None,
            )
            if plot:
                _plot(result, motor_label, det_label, mode)
            return result

        extend_count += 1

        if _data_sufficient(np.array(pos_data), np.array(sig_data),
                            mode, sigma_coverage):
            if verbose:
                print(f"  Data sufficient after {extend_count} extension step(s).")
            break

    # ── Phase 4: peak analysis ────────────────────────────────────────────────
    positions, signals = _dedup(np.array(pos_data), np.array(sig_data))
    s = stats_peak(positions, signals, mode=mode)

    def _stats_cen(st):
        val = st.get(stats_centre, None)
        if val is None or np.isnan(float(val)):
            val = st.get("peak_pos")
        return float(val)

    if peak_method == "stats":
        cen  = _stats_cen(s)
        sig  = s["fwhm_width"] if not np.isnan(s["fwhm_width"]) else s["rms_width"]
        amp  = s["peak_val"]
        off  = 0.0
        profile_used = "stats"
        msg  = (f"Stats: {'peak' if mode=='max' else 'valley'} at {cen:.6g}  "
                f"FWHM={s['fwhm_interp']:.5g}  RMS-σ={s['rms_width']:.5g}")
        if verbose:
            print(f"\n✓ {msg}")
            print_stats(s)
        coarse_result = ScanResult(
            status=ScanStatus.SUCCESS,
            positions=positions, signals=signals,
            center=cen, sigma=sig,
            amplitude=amp, offset=off,
            profile="stats",
            stats=s,
            message=msg,
        )

    else:
        fit_ok, cen, sig, amp, off, profile_used = _try_fit(positions, signals, mode, profile=fit_profile)

        if not fit_ok:
            msg = "Profile fit failed; raw optimum reported."
            if verbose:
                print(f"\n⚠ {msg}")
            result = ScanResult(
                status=ScanStatus.FIT_FAILED,
                positions=positions, signals=signals,
                stats=s,
                message=msg,
            )
            if plot:
                _plot(result, motor_label, det_label, mode)
            return result

        msg = (f"{'Peak' if mode=='max' else 'Valley'} found at "
               f"{cen:.5g} ± {sig:.5g} ({profile_used}).")
        if peak_method == "both":
            msg += f"  Centroid={s['centroid']:.6g}  FWHM={s['fwhm_interp']:.5g}"
        if verbose:
            print(f"\n✓ {msg}")
            if peak_method == "both":
                print_stats(s)

        if profile_used == "supergaussian":
            s["supergaussian_p"] = _last_supergaussian_p
            if verbose:
                print(f"  Super-Gaussian p = {_last_supergaussian_p:.3f}  "
                      f"({'flat-topped' if _last_supergaussian_p > 3 else 'near-Gaussian'})")

        coarse_result = ScanResult(
            status=ScanStatus.SUCCESS,
            positions=positions, signals=signals,
            center=cen, sigma=sig, amplitude=amp, offset=off,
            profile=profile_used,
            stats=s,
            message=msg,
        )

    if plot:
        _plot(coarse_result, motor_label, det_label, mode,
              title_prefix="Coarse Scan")

    # ── Phase 5: fine scan ────────────────────────────────────────────────────
    if fine_scan:
        coarse_step = abs(stop - start) / max(nsteps - 1, 1)
        if (peak_method == "stats" and s is not None
                and "peak_pos" in s and not np.isnan(s["peak_pos"])):
            centroid_drift = abs(cen - s["peak_pos"])
            if centroid_drift > coarse_step:
                if verbose:
                    print(f"  ⚠ Stats centre ({cen:.6g}) is {centroid_drift:.4g} from "
                          f"peak_pos ({s['peak_pos']:.6g}) > 1 step ({coarse_step:.4g}) "
                          f"– using peak_pos for fine scan centre")
                cen = s["peak_pos"]

        iter_cen  = cen
        iter_sig  = sig
        cen2, sig2, amp2, off2, profile_used2 = cen, sig, amp, off, profile_used
        fine_positions, fine_signals = positions, signals
        fit_ok2 = True

        if _fine_scan_start_hook is not None:
            _fine_scan_start_hook()
        for fine_iter in range(max(1, fine_scan_iter)):
            iter_half = fine_sigma_range * iter_sig

            if verbose:
                iter_label = f" (iteration {fine_iter+1}/{fine_scan_iter})" if fine_scan_iter > 1 else ""
                print(f"\n── Fine scan{iter_label}: centre={iter_cen:.6g}  ±{iter_half:.4g} ({fine_sigma_range}σ)  steps={fine_nsteps} ──")

            # On the first iteration always move to peak_pos (the measured maximum)
            # so the fine scan window is correctly centred on the signal peak.
            # move_target only controls where the motor goes AFTER the final scan,
            # not where the fine scan is centred.
            if fine_iter == 0:
                fine_centre = (coarse_result.stats["peak_pos"]
                               if coarse_result.stats and
                               not np.isnan(coarse_result.stats.get("peak_pos", float("nan")))
                               else iter_cen)
                if verbose:
                    print(f"  Fine scan centred on peak_pos ({fine_centre:.6g})")
                iter_cen = fine_centre
            iface.move(iter_cen, timeout=motor_timeout)

            fine_pos_data: list = []
            fine_sig_data: list = []

            fine_start = iter_cen + iter_half if stop < start else iter_cen - iter_half
            iface.move(fine_start, timeout=motor_timeout)
            if not iface._sim:
                if verbose:
                    print(f"  [Flushing detector at fine start position…]")
                iface.read()
                time.sleep(det_update_interval)
                iface.read()

            fine_targets = (
                np.linspace(iter_cen + iter_half, iter_cen - iter_half, fine_nsteps)
                if stop < start else
                np.linspace(iter_cen - iter_half, iter_cen + iter_half, fine_nsteps)
            )

            fine_step = abs(fine_targets[1] - fine_targets[0])

            def _fine_acquire(target):
                ok = iface.move(target, timeout=motor_timeout)
                if not ok:
                    return None
                if settle > 0:
                    time.sleep(settle)
                actual = iface.position()
                if fine_pos_data and abs(actual - fine_pos_data[-1]) < fine_step * 0.1:
                    return 0.0
                signal = _fresh_read(fine_sig_data)
                fine_pos_data.append(actual)
                fine_sig_data.append(signal)
                if verbose:
                    print(f"  pos={actual:>13.8g}   signal={signal:>14.5g}")
                return signal

            timed_out = False
            for target in fine_targets:
                if _fine_acquire(target) is None:
                    if verbose:
                        print("  ⚠ Fine scan motor timed out – using previous result.")
                    timed_out = True
                    break

            if timed_out or len(fine_pos_data) < 5:
                if verbose and not timed_out:
                    print("  ⚠ Too few fine scan points – using previous result.")
                break

            _fp, _fs = _dedup(np.array(fine_pos_data), np.array(fine_sig_data))

            # When peak_method="stats" skip curve fitting entirely –
            # use model-free statistics for centre and width.
            if peak_method == "stats":
                _s_iter = stats_peak(_fp, _fs, mode=mode)
                _iter_cen = _stats_cen(_s_iter)
                _iter_sig = (_s_iter["fwhm_width"]
                             if not np.isnan(_s_iter["fwhm_width"])
                             else _s_iter["rms_width"])
                if _iter_sig is None or np.isnan(_iter_sig):
                    if verbose:
                        print("  ⚠ Fine stats width unavailable – using previous result.")
                    break
                prev_sig      = iter_sig
                iter_cen      = _iter_cen
                iter_sig      = _iter_sig
                amp2, off2, profile_used2 = _s_iter["peak_val"], 0.0, "stats"
                fine_positions, fine_signals = _fp, _fs
                cen2, sig2, fit_ok2 = iter_cen, iter_sig, True
                if verbose:
                    shrink = prev_sig / iter_sig if iter_sig > 0 else 1.0
                    print(f"  σ: {prev_sig:.4g} → {iter_sig:.4g}  (shrink factor={shrink:.2f}x)")
                if prev_sig / max(iter_sig, 1e-30) < 2.0:
                    if verbose:
                        print(f"  σ stable – stopping iterations.")
                    break
                continue

            # For peak_method="fit" or "both": use curve fitting
            _ok, _c, _s, _a, _o, _pu = _try_fit(_fp, _fs, mode, profile=fit_profile)

            if not _ok:
                _s_fine = stats_peak(_fp, _fs, mode=mode)
                if _s_fine["peak_pos"] is not None and not np.isnan(_s_fine["peak_pos"]):
                    iter_cen  = _stats_cen(_s_fine)
                    iter_sig  = (_s_fine["fwhm_width"]
                                 if not np.isnan(_s_fine["fwhm_width"])
                                 else _s_fine["rms_width"])
                    amp2      = _s_fine["peak_val"]
                    off2      = 0.0
                    profile_used2 = "stats"
                    fine_positions, fine_signals = _fp, _fs
                    cen2, sig2, fit_ok2 = iter_cen, iter_sig, True
                    if verbose:
                        print(f"  ⚠ Fine fit failed – using stats {stats_centre}: {iter_cen:.6g}")
                else:
                    if verbose:
                        print("  ⚠ Fine fit failed and stats unavailable – using previous result.")
                break

            prev_sig = iter_sig
            if peak_method == "stats":
                _s_iter = stats_peak(_fp, _fs, mode=mode)
                iter_cen  = _stats_cen(_s_iter)
                iter_sig  = (_s_iter["fwhm_width"]
                             if not np.isnan(_s_iter["fwhm_width"])
                             else _s_iter["rms_width"])
                amp2, off2, profile_used2 = _s_iter["peak_val"], 0.0, "stats"
            else:
                iter_cen, iter_sig, amp2, off2, profile_used2 = _c, _s, _a, _o, _pu
            fine_positions, fine_signals = _fp, _fs
            cen2, sig2, fit_ok2 = iter_cen, iter_sig, True

            if verbose:
                shrink = prev_sig / iter_sig if iter_sig > 0 else 1.0
                print(f"  σ: {prev_sig:.4g} → {iter_sig:.4g}  (shrink factor={shrink:.2f}x)")

            if prev_sig / max(iter_sig, 1e-30) < 2.0:
                if verbose:
                    print(f"  σ stable – stopping iterations.")
                break

        if not fit_ok2:
            if verbose:
                print("  ⚠ Fine Gaussian fit failed – returning coarse result.")
            if move_to_peak:
                target_pos = _choose_centre(coarse_result, verbose=verbose,
                                            move_target=move_target)
                if verbose:
                    print(f"  → Moving to: {target_pos:.6g}")
                backlash = _backlash_fwhm(coarse_result) if backlash_correction else 0.0
                _move_with_backlash(iface, target_pos, scan_descending,
                                   backlash, motor_timeout, verbose)
            if verbose:
                print(coarse_result)
            return coarse_result

        # Build fine ScanResult with stats populated
        fine_stats = stats_peak(fine_positions, fine_signals, mode=mode)
        if profile_used2 == "supergaussian":
            fine_stats["supergaussian_p"] = _last_supergaussian_p
            if verbose:
                print(f"  Super-Gaussian p = {_last_supergaussian_p:.3f}  "
                      f"({'flat-topped' if _last_supergaussian_p > 3 else 'near-Gaussian'})")
        fine_msg = (f"Fine scan: {'peak' if mode=='max' else 'valley'} at "
                    f"{cen2:.6g} ± {sig2:.4g} ({profile_used2})  "
                    f"[coarse: {cen:.6g} ± {sig:.4g} ({profile_used})]")
        if verbose:
            print(f"\n✓ {fine_msg}")

        result = ScanResult(
            status=ScanStatus.SUCCESS,
            positions=fine_positions, signals=fine_signals,
            center=cen2, sigma=sig2, amplitude=amp2, offset=off2,
            profile=profile_used2,
            message=fine_msg,
            stats=fine_stats,
        )

        if plot:
            _plot(result, motor_label, det_label, mode,
                  title_prefix="Fine Scan")

    else:
        result = coarse_result

    # ── Phase 6: move to chosen centre (with optional backlash correction) ────
    if move_to_peak:
        final_target = _choose_centre(result, verbose=verbose,
                                      move_target=move_target)
        if verbose:
            print(f"  → Moving to: {final_target:.6g}")
        backlash = _backlash_fwhm(result) if backlash_correction else 0.0
        _move_with_backlash(iface, final_target, scan_descending,
                            backlash, motor_timeout, verbose)

    if verbose:
        print(result)

    return result


# ── Optional debug plot ───────────────────────────────────────────────────────

def _plot(result: ScanResult, motor_label: str, det_label: str, mode: str,
          title_prefix: str = "Smart Scan"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed – skipping plot.", stacklevel=2)
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             gridspec_kw={"width_ratios": [3, 1]})
    ax, ax_t = axes

    ax.plot(result.positions, result.signals, "o-", ms=4,
            color="#4C72B0", label="Measured", zorder=3)

    if result.status == ScanStatus.SUCCESS:
        profile = getattr(result, "profile", "gaussian")

        if profile == "stats":
            # ── Stats mode: attempt a super-Gaussian fit for visual overlay ───
            # The fit is display-only and does not affect the motor move.
            s = result.stats or {}
            peak_pos = s.get("peak_pos")
            centroid = s.get("centroid")
            fwhm     = s.get("fwhm_interp")

            fit_ok, f_cen, f_sig, f_amp, f_off, f_prof = _try_fit(
                result.positions, result.signals, mode, profile="auto")

            if fit_ok:
                xs       = np.linspace(result.positions.min(),
                                       result.positions.max(), 600)
                amp_plot = f_amp if mode == "max" else -f_amp
                if f_prof == "supergaussian":
                    p_val = _last_supergaussian_p
                    ys        = _supergaussian(xs, amp_plot, f_cen,
                                               f_sig, p_val, f_off)
                    fit_label = f"Super-Gaussian (p={p_val:.2f})"
                elif f_prof == "lorentzian":
                    ys        = _lorentzian(xs, amp_plot, f_cen, f_sig, f_off)
                    fit_label = "Lorentzian fit"
                else:
                    ys        = _gaussian(xs, amp_plot, f_cen, f_sig, f_off)
                    fit_label = "Gaussian fit"
                ax.plot(xs, ys, "--", lw=1.5, color="#DD8452",
                        alpha=0.7, label=f"{fit_label} (display only)")

            if peak_pos is not None and not np.isnan(float(peak_pos)):
                ax.axvline(float(peak_pos), color="#C44E52", ls=":",  lw=1.5,
                           label=f"peak_pos = {float(peak_pos):.6g}")
            if centroid is not None and not np.isnan(float(centroid)):
                ax.axvline(float(centroid), color="#DD8452", ls="--", lw=1.5,
                           label=f"centroid = {float(centroid):.6g}")
            if (fwhm is not None and not np.isnan(float(fwhm))
                    and peak_pos is not None and not np.isnan(float(peak_pos))):
                hw = float(fwhm) / 2.0
                ax.axvspan(float(peak_pos) - hw, float(peak_pos) + hw,
                           alpha=0.08, color="#C44E52", label=f"FWHM = {float(fwhm):.4g}")

            # Motor target
            chosen = _choose_centre(result, verbose=False, move_target="auto")
            ax.axvline(chosen, color="#2ca02c", ls="-.", lw=1.5,
                       label=f"Motor target = {chosen:.6g}")

        else:
            # ── Fit mode: draw the fitted curve ──────────────────────────────
            xs       = np.linspace(result.positions.min(), result.positions.max(), 600)
            amp_plot = result.amplitude if mode == "max" else -result.amplitude

            if profile == "lorentzian":
                ys          = _lorentzian(xs, amp_plot, result.center,
                                          result.sigma, result.offset)
                fit_label   = "Lorentzian fit"
                width_label = f"±γ (HWHM={result.sigma:.4g})"
            elif profile == "supergaussian":
                p_val = (result.stats.get("supergaussian_p", 2.0)
                         if result.stats else 2.0)
                ys          = _supergaussian(xs, amp_plot, result.center,
                                             result.sigma, p_val, result.offset)
                fwhm_sg     = _supergaussian_sigma_to_fwhm(result.sigma, p_val)
                fit_label   = f"Super-Gaussian fit (p={p_val:.2f})"
                width_label = f"±σ  FWHM={fwhm_sg:.4g}"
            else:
                ys          = _gaussian(xs, amp_plot, result.center,
                                        result.sigma, result.offset)
                fit_label   = "Gaussian fit"
                width_label = "±1σ"

            ax.plot(xs, ys, "--", lw=2, color="#DD8452", label=fit_label)
            ax.axvline(result.center, color="#C44E52", ls=":", lw=1.5,
                       label=f"Centre = {result.center:.6g}")
            ax.axvspan(result.center - result.sigma,
                       result.center + result.sigma,
                       alpha=0.08, color="#C44E52", label=width_label)

            if result.stats is not None:
                chosen = _choose_centre(result, verbose=False, move_target="auto")
                ax.axvline(chosen, color="#2ca02c", ls="-.", lw=1.5,
                           label=f"Motor target = {chosen:.6g}")

    ax.axvline(result.positions[0],  color="grey", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(result.positions[-1], color="grey", ls="--", lw=0.8, alpha=0.5,
               label="Scan limits")

    ax.set_xlabel(f"Motor [{motor_label}]", fontsize=11)
    ax.set_ylabel(f"Detector [{det_label}]", fontsize=11)
    ax.set_title(f"{title_prefix}  –  {result.status.value}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Parameter table ───────────────────────────────────────────────────────
    ax_t.set_axis_off()
    if result.status == ScanStatus.SUCCESS:
        profile = getattr(result, 'profile', 'gaussian')
        s       = result.stats or {}

        if profile == "stats":
            fwhm    = s.get("fwhm_interp", float("nan"))
            rms_w   = s.get("rms_width",   float("nan"))
            peak_p  = s.get("peak_pos",    float("nan"))
            centrd  = s.get("centroid",    float("nan"))
            peak_v  = s.get("peak_val",    float("nan"))
            # Show which fit was overlaid (display only)
            fit_ok2, f_cen2, f_sig2, f_amp2, f_off2, f_prof2 = _try_fit(
                result.positions, result.signals, mode, profile="auto")
            fit_note = f_prof2 if fit_ok2 else "n/a"
            if fit_ok2 and f_prof2 == "supergaussian":
                fit_note = f"supergaussian (p={_last_supergaussian_p:.2f})"
            def _fmt(v):
                try: return f"{float(v):.6g}" if not np.isnan(float(v)) else "n/a"
                except: return "n/a"
            rows = [
                ["Parameter",    "Value"],
                ["Profile",      "stats"],
                ["Fit overlay",  fit_note],
                ["peak_pos",     _fmt(peak_p)],
                ["Centroid",     _fmt(centrd)],
                ["FWHM (interp)",_fmt(fwhm)],
                ["RMS width σ",  _fmt(rms_w)],
                ["Peak signal",  _fmt(peak_v)],
                ["Points",       str(len(result.positions))],
            ]
        else:
            w_label = 'Gamma (γ)' if profile == 'lorentzian' else 'Sigma (σ)'
            rows = [
                ["Parameter", "Value"],
                ["Profile",       profile],
                ["Centre (µ)",   f"{result.center:.6g}"],
                [w_label,        f"{result.sigma:.5g}"],
                ["Amplitude",    f"{result.amplitude:.5g}"],
                ["Offset",       f"{result.offset:.5g}"],
                ["Points",       str(len(result.positions))],
            ]

        tbl = ax_t.table(cellText=rows[1:], colLabels=rows[0],
                         loc="center", cellLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.1, 1.7)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#CCCCCC")
            if r == 0:
                cell.set_facecolor("#4C72B0")
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#EEF2F8")
    else:
        ax_t.text(0.5, 0.5, result.status.value + "\n\n" + result.message,
                  ha="center", va="center", transform=ax_t.transAxes,
                  fontsize=10, color="grey", wrap=True)

    plt.tight_layout()
    plt.show()



# =============================================================================
# FLY SCAN
# =============================================================================

import threading


class _FlyInterface:
    def __init__(self, motor, det, simulate: bool,
                 sim_center, sim_sigma, sim_amp, sim_offset, sim_noise,
                 sim_velocity):
        self._sim      = simulate or not _EPICS_AVAILABLE
        self._sim_pos  = 0.0
        self._sim_vel  = sim_velocity
        self._sim_t0   = None
        self._sim_dest = 0.0
        self._pvaxis   = None
        self._motor    = None
        self._sim_c    = sim_center
        self._sim_s    = sim_sigma
        self._sim_a    = sim_amp
        self._sim_o    = sim_offset
        self._sim_n    = sim_noise

        if not self._sim:
            if isinstance(motor, PVAxis):
                self._pvaxis = motor
                self._motor  = None
            elif isinstance(motor, str):
                print(f"    [DEBUG] fly_scan: Creating epics.Motor({motor!r})")
                self._motor = epics.Motor(motor)
                print(f"    [DEBUG] fly_scan: Motor type: {type(self._motor).__name__}")
                try:
                    _conn = self._motor.connected
                    print(f"    [DEBUG] fly_scan: Motor.connected = {_conn}")
                except AttributeError as _ae:
                    print(f"    [DEBUG] fly_scan: Motor.connected raised AttributeError: {_ae} — skipping check")
                    _conn = True
                if not _conn:
                    raise ConnectionError(f"Motor PV not connected: {motor}")
            else:
                print(f"    [DEBUG] fly_scan: Motor passed as object: {type(motor).__name__}")
                self._motor = motor

            if isinstance(det, str):
                self._det = epics.PV(det, auto_monitor=True)
                if not self._det.connect(timeout=5):
                    raise ConnectionError(f"Detector PV not connected: {det}")
            else:
                self._det = det

    def soft_limits(self) -> tuple:
        if self._sim:
            return (-1e9, 1e9)
        if self._pvaxis is not None:
            return self._pvaxis.soft_limits()
        return (self._motor.low_limit, self._motor.high_limit)

    def get_velocity(self) -> float:
        if self._sim:
            return self._sim_vel
        if self._pvaxis is not None:
            return self._pvaxis.get_velocity()
        velo_pv = self._motor.PV("VELO")
        return float(velo_pv.get(use_monitor=False))

    def set_velocity(self, v: float):
        if self._sim:
            self._sim_vel = v
        elif self._pvaxis is not None:
            self._pvaxis.set_velocity(v)
        else:
            velo_pv = self._motor.PV("VELO")
            velo_pv.put(v, wait=True)
            epics.ca.flush_io()

    def get_base_velocity(self) -> float:
        if self._sim:
            return self._sim_vel * 0.1
        if self._pvaxis is not None:
            return self._pvaxis.get_base_velocity()
        return float(self._motor.PV("VBAS").get(use_monitor=False))

    def set_base_velocity(self, v: float):
        if self._sim:
            pass
        elif self._pvaxis is not None:
            self._pvaxis.set_base_velocity(v)
        else:
            self._motor.PV("VBAS").put(v, wait=True)
            epics.ca.flush_io()

    def move_start(self, dest: float):
        if self._sim:
            self._sim_dest = dest
            self._sim_t0   = time.monotonic()
        elif self._pvaxis is not None:
            self._pvaxis.put(dest)
        else:
            self._motor.move(dest, wait=False)

    def is_moving(self) -> bool:
        if self._sim:
            if self._sim_t0 is None:
                return False
            elapsed   = time.monotonic() - self._sim_t0
            distance  = abs(self._sim_dest - self._sim_pos)
            travelled = self._sim_vel * elapsed
            return travelled < distance
        if self._pvaxis is not None:
            return self._pvaxis.is_moving()
        dmov = self._motor.PV("DMOV").get(use_monitor=False)
        return dmov == 0

    def wait_move(self, timeout: float = 120.0):
        if not self._sim:
            time.sleep(0.15)
        t0 = time.monotonic()
        while self.is_moving():
            if time.monotonic() - t0 > timeout:
                raise TimeoutError("Fly scan motor move timed out.")
            time.sleep(0.05)
        if self._sim:
            self._sim_pos  = self._sim_dest
            self._sim_t0   = None

    def read_position(self) -> float:
        if self._sim:
            if self._sim_t0 is None:
                return self._sim_pos
            elapsed   = time.monotonic() - self._sim_t0
            direction = 1.0 if self._sim_dest > self._sim_pos else -1.0
            moved     = min(self._sim_vel * elapsed,
                            abs(self._sim_dest - self._sim_pos))
            return self._sim_pos + direction * moved
        if self._pvaxis is not None:
            return self._pvaxis.get_position()
        return float(self._motor.PV("RBV").get(use_monitor=False))

    def read_detector(self) -> float:
        if self._sim:
            p = self.read_position()
            return (self._sim_a
                    * np.exp(-0.5 * ((p - self._sim_c) / self._sim_s) ** 2)
                    + self._sim_o
                    + np.random.normal(0, self._sim_n))
        return float(self._det.get())


def _sample_loop(iface: _FlyInterface, sample_interval: float,
                 pos_list: list, sig_list: list,
                 stop_event: threading.Event, verbose: bool):
    while not stop_event.is_set():
        t_next = time.monotonic() + sample_interval
        p = iface.read_position()
        s = iface.read_detector()
        pos_list.append(p)
        sig_list.append(s)
        if verbose:
            print(f"  pos={p:>13.8g}   signal={s:>14.5g}")
        remaining = t_next - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)


def fly_scan(
    motor,
    det,
    start               : float,
    stop                : float,
    sample_interval     : float = 0.1,
    nsteps              : int   = 21,
    det_update_interval : float = 0.2,
    velocity_factor     : Optional[float] = None,
    restore_velocity    : bool  = True,
    mode                : str   = "max",
    move_to_peak        : bool  = True,
    move_target         : str   = "auto",  # "auto" | "peak_pos" | "centroid"
    backlash_correction : bool  = False,
    motor_timeout       : float = 120.0,
    min_prominence_ratio: float = 5.0,
    edge_fraction       : float = 0.2,
    sigma_coverage      : float = 2.0,
    extend_scan         : bool  = True,
    max_extend_steps    : int   = 10,
    peak_method         : str   = "stats",
    stats_centre        : str   = "centroid",
    fit_profile         : str   = "auto",
    plot                : bool  = False,
    simulate            : bool  = False,
    sim_center          : float = 1.5,
    sim_sigma           : float = 1.2,
    sim_amplitude       : float = 100.0,
    sim_offset          : float = 10.0,
    sim_noise           : float = 2.0,
    sim_velocity        : float = 2.0,
    verbose             : bool  = True,
    debug               : bool  = False,
) -> ScanResult:
    """
    Fly scan with _choose_centre() applied to the move_to_peak step.
    """
    if mode not in ("max", "min"):
        raise ValueError("mode must be 'max' or 'min'.")
    if sample_interval <= 0:
        raise ValueError("sample_interval must be > 0.")

    motor_label = _safe_label(motor)
    det_label   = _safe_label(det)

    if debug:
        verbose = False

    iface = _FlyInterface(
        motor, det,
        simulate=simulate,
        sim_center=sim_center, sim_sigma=sim_sigma,
        sim_amp=sim_amplitude, sim_offset=sim_offset, sim_noise=sim_noise,
        sim_velocity=sim_velocity,
    )

    current_pos = iface.read_position()
    abs_start   = current_pos + start
    abs_stop    = current_pos + stop

    if verbose:
        print(f"  Current position : {current_pos:.5g}")
        print(f"  Relative offsets : start={start:+g}  stop={stop:+g}")
        print(f"  Absolute targets : {abs_start:.5g} → {abs_stop:.5g}")

    start, stop = abs_start, abs_stop

    llm, hlm = iface.soft_limits()
    scan_forward = stop > start

    if scan_forward:
        start_clamped = max(start, llm)
        stop_clamped  = min(stop,  hlm)
    else:
        start_clamped = min(start, hlm)
        stop_clamped  = max(stop,  llm)

    if start_clamped != start or stop_clamped != stop:
        if verbose:
            print(f"  ⚠ Limits clamped: [{start:.5g}, {stop:.5g}] → "
                  f"[{start_clamped:.5g}, {stop_clamped:.5g}]")
        start, stop = start_clamped, stop_clamped

    if start == stop or (scan_forward and start > stop) or (not scan_forward and start < stop):
        msg = "Scan range is zero or inverted after soft-limit clamping; nothing to scan."
        if verbose:
            print(f"✗ {msg}")
        return ScanResult(
            status=ScanStatus.OUT_OF_RANGE,
            positions=np.array([]), signals=np.array([]),
            message=msg,
        )

    is_pvaxis       = isinstance(motor, PVAxis)
    original_velo   = iface.get_velocity() if not is_pvaxis else float("nan")
    original_bvel   = iface.get_base_velocity() if not is_pvaxis else float("nan")
    velocity_changed = velocity_factor is not None and not is_pvaxis

    if velocity_factor is not None and is_pvaxis and verbose:
        print("  ⚠ velocity_factor ignored – PVAxis has no velocity control.")

    if velocity_changed:
        if velocity_factor <= 0:
            raise ValueError("velocity_factor must be > 0.")
        scan_velo = original_velo * velocity_factor
        scan_bvel = original_bvel * velocity_factor
        iface.set_velocity(scan_velo)
        iface.set_base_velocity(scan_bvel)
        time.sleep(0.05)
        if verbose:
            print(f"  Velocity factor  : {velocity_factor}×")
            print(f"  VELO: {original_velo:.4g} → {scan_velo:.4g} EGU/s")
            print(f"  VBAS: {original_bvel:.4g} → {scan_bvel:.4g} EGU/s")

    if verbose:
        tag = " [SIM]" if (simulate or not _EPICS_AVAILABLE) else ""
        print(f"=== Fly Scan{tag}  motor={motor_label}  det={det_label}  mode={mode} ===")
        if is_pvaxis:
            print(f"    Range: {start} → {stop}   interval={sample_interval}s   [PVAxis – no velocity]")
        else:
            print(f"    Range: {start} → {stop}   "
                  f"interval={sample_interval}s   vel={iface.get_velocity():.4g} EGU/s")

    pos_list: list = []
    sig_list: list = []

    pre_scan_pos  = current_pos
    scan_exception = None
    stop_event     = threading.Event()

    def _cleanup(interrupted: bool = False):
        stop_event.set()
        if interrupted:
            try:
                if not iface._sim:
                    iface._motor.stop()
            except Exception:
                pass
            if verbose:
                print("\n  ⚠ Interrupted – restoring state …")
        if restore_velocity and velocity_changed:
            iface.set_velocity(original_velo)
            iface.set_base_velocity(original_bvel)
            if verbose:
                print(f"  VELO restored to {original_velo:.4g} EGU/s  VBAS restored to {original_bvel:.4g} EGU/s")
        if interrupted:
            if verbose:
                print(f"  Returning to pre-scan position {pre_scan_pos:.5g} …")
            try:
                iface.move_start(pre_scan_pos)
                iface.wait_move(timeout=motor_timeout)
                if verbose:
                    print(f"  Done.")
            except Exception as e:
                if verbose:
                    print(f"  Could not return to pre-scan position: {e}")

    try:
        if is_pvaxis:
            step_targets = np.linspace(start, stop, nsteps)
            step_size    = abs(step_targets[1] - step_targets[0]) if nsteps > 1 else abs(stop - start)

            if verbose:
                print(f"  Stepped PVAxis scan: {start:.5g} → {stop:.5g}  ({nsteps} steps)")
                print(f"  {'Position':>13.13}  {'Signal':>14}")
                print("  " + "─" * 30)

            if not iface._sim:
                motor.put(step_targets[0])
                time.sleep(motor.settle_time)
                iface._det.get(use_monitor=False)
                time.sleep(det_update_interval)
                iface._det.get(use_monitor=False)

            for target in step_targets:
                if not iface._sim:
                    motor.put(target)
                    time.sleep(motor.settle_time)
                    actual = motor.get_position()
                    prev = iface._det.get(use_monitor=False)
                    t0 = time.monotonic()
                    while time.monotonic() - t0 < det_update_interval * 5:
                        time.sleep(det_update_interval * 0.1)
                        curr = iface._det.get(use_monitor=False)
                        if curr != prev:
                            prev = curr
                            break
                    t0 = time.monotonic()
                    while time.monotonic() - t0 < det_update_interval * 5:
                        time.sleep(det_update_interval * 0.1)
                        curr = iface._det.get(use_monitor=False)
                        if curr != prev:
                            signal = float(curr)
                            break
                    else:
                        signal = float(curr)
                else:
                    actual = target
                    iface._sim_pos = target
                    signal = iface.read_detector()

                pos_list.append(actual)
                sig_list.append(signal)
                if verbose:
                    print(f"  pos={actual:>13.8g}   signal={signal:>14.5g}")

        else:
            if verbose:
                print(f"  Moving to start position {start:.5g} …")
            iface.move_start(start)
            iface.wait_move(timeout=motor_timeout)

            if not iface._sim:
                if verbose:
                    print(f"  [Flushing detector at start position…]")
                iface._det.get(use_monitor=False)
                time.sleep(det_update_interval)
                iface._det.get(use_monitor=False)

            sampler = threading.Thread(
                target=_sample_loop,
                args=(iface, sample_interval, pos_list, sig_list, stop_event, verbose),
                daemon=True,
            )

            if verbose:
                print(f"  Flying {start:.5g} → {stop:.5g} …")
                print(f"  {'Position':>12}  {'Signal':>14}")
                print("  " + "─" * 30)

            sampler.start()
            iface.move_start(stop)
            iface.wait_move(timeout=motor_timeout)

            stop_event.set()
            sampler.join(timeout=sample_interval * 3)

    except KeyboardInterrupt:
        _cleanup(interrupted=True)
        raise

    except Exception as exc:
        scan_exception = exc

    finally:
        if scan_exception is not None:
            _cleanup(interrupted=False)

    if scan_exception is not None:
        raise scan_exception

    if pos_list and not is_pvaxis:
        pos_list = pos_list[1:]
        sig_list = sig_list[1:]
        if verbose:
            print("  [First point discarded – detector flush]")

    if len(pos_list) < 5:
        if restore_velocity and velocity_changed:
            iface.set_velocity(original_velo)
            iface.set_base_velocity(original_bvel)
        return ScanResult(
            status=ScanStatus.INSUFFICIENT_DATA,
            positions=np.array(pos_list), signals=np.array(sig_list),
            message=f"Only {len(pos_list)} samples collected – increase range or decrease velocity.",
            stats=stats_peak(np.array(pos_list), np.array(sig_list), mode=mode) if pos_list else None,
        )

    order     = np.argsort(pos_list)
    positions = np.array(pos_list)[order]
    signals   = np.array(sig_list)[order]
    if not scan_forward:
        positions = positions[::-1]
        signals   = signals[::-1]

    if verbose:
        print(f"\n  Collected {len(positions)} samples over "
              f"[{positions.min():.4g}, {positions.max():.4g}]")

    if not _has_peak_character(signals, mode, min_prominence_ratio):
        msg = (f"No {'peak' if mode=='max' else 'dip'} detected: "
               f"signal variation below {min_prominence_ratio}× noise floor.")
        if verbose:
            print(f"\n✗ {msg}")
        if restore_velocity and velocity_changed:
            iface.set_velocity(original_velo)
            iface.set_base_velocity(original_bvel)
        result = ScanResult(
            status=ScanStatus.NO_PEAK,
            positions=positions, signals=signals,
            message=msg,
            stats=stats_peak(np.array(pos_list), np.array(sig_list), mode=mode) if pos_list else None,
        )
        if plot:
            _plot(result, motor_label, det_label, mode)
        return result

    edge = _peak_near_edge(positions, signals, mode, edge_fraction,
                           scan_descending=not scan_forward)
    edge_warning = ""

    if edge and extend_scan and is_pvaxis:
        if verbose:
            print(f"  Peak near '{edge}' edge – extending fly scan...")
        extend_count = 0
        step_size_fly = abs(stop - start) / (nsteps - 1) if nsteps > 1 else abs(stop - start)
        while edge and extend_count < max_extend_steps:
            if (edge == "end" and scan_forward) or (edge == "start" and not scan_forward):
                next_pos = np.max(pos_list) + step_size_fly
            else:
                next_pos = np.min(pos_list) - step_size_fly

            llm_ext, hlm_ext = iface.soft_limits()
            if next_pos > hlm_ext or next_pos < llm_ext:
                if verbose:
                    print(f"  ⚠ Soft limit reached during extension.")
                break

            if not iface._sim:
                motor.put(next_pos)
                time.sleep(motor.settle_time)
            else:
                iface._sim_pos = next_pos

            actual = motor.get_position() if not iface._sim else next_pos
            signal = iface.read_detector()
            pos_list.append(actual)
            sig_list.append(signal)
            if verbose:
                print(f"  pos={actual:>13.8g}   signal={signal:>14.5g}")

            order    = np.argsort(pos_list)
            positions = np.array(pos_list)[order]
            signals   = np.array(sig_list)[order]
            if not scan_forward:
                positions = positions[::-1]
                signals   = signals[::-1]

            edge = _peak_near_edge(positions, signals, mode, edge_fraction,
                                   scan_descending=not scan_forward)
            extend_count += 1

            if _data_sufficient(positions, signals, mode, sigma_coverage):
                if verbose:
                    print(f"  Data sufficient after {extend_count} extension step(s).")
                edge = None
                break

        if edge:
            edge_warning = (f" (peak near '{edge}' edge – consider re-scanning "
                            f"with extended limits)")
            if verbose:
                print(f"  ⚠ {edge_warning.strip()}")

    elif edge:
        edge_warning = (f" (peak near '{edge}' edge – consider re-scanning "
                        f"with extended limits)")
        if verbose:
            print(f"  ⚠ Peak is near the '{edge}' edge of the scan range.")

    if peak_method == "stats":
        s_fly_stats = stats_peak(positions, signals, mode=mode)
        cen  = float(s_fly_stats.get(stats_centre) or s_fly_stats["peak_pos"])
        sig  = s_fly_stats["fwhm_width"] if not np.isnan(s_fly_stats["fwhm_width"]) else s_fly_stats["rms_width"]
        profile_used = "stats"
        fit_ok = True
        amp = s_fly_stats["peak_val"]
        off = 0.0
    else:
        fit_ok, cen, sig, amp, off, profile_used = _try_fit(positions, signals, mode, profile=fit_profile)

    if not fit_ok:
        msg = "Profile fit failed." + edge_warning
        if verbose:
            print(f"\n⚠ {msg}")
        result = ScanResult(
            status=ScanStatus.FIT_FAILED,
            positions=positions, signals=signals,
            message=msg,
            stats=stats_peak(np.array(pos_list), np.array(sig_list), mode=mode) if pos_list else None,
        )
        if plot:
            _plot(result, motor_label, det_label, mode)
        return result

    msg = (f"{'Peak' if mode=='max' else 'Valley'} found at "
           f"{cen:.5g} ± {sig:.5g} ({profile_used}).{edge_warning}")
    if verbose:
        print(f"\n✓ {msg}")

    result = ScanResult(
        status=ScanStatus.SUCCESS,
        positions=positions, signals=signals,
        center=cen, sigma=sig, amplitude=amp, offset=off,
        profile=profile_used,
        message=msg,
        stats=stats_peak(positions, signals, mode=mode),
    )

    if restore_velocity and velocity_changed:
        iface.set_velocity(original_velo)
        iface.set_base_velocity(original_bvel)
        if verbose:
            print(f"  VELO restored to {original_velo:.4g} EGU/s  "
                  f"VBAS restored to {original_bvel:.4g} EGU/s")

    if move_to_peak:
        final_target = _choose_centre(result, verbose=verbose,
                                      move_target=move_target)
        if verbose:
            print(f"  → Moving to: {final_target:.6g}")
        if backlash_correction:
            fly_backlash = _backlash_fwhm(result)
            overshoot = (final_target + fly_backlash if not scan_forward
                         else final_target - fly_backlash)
            if verbose:
                print(f"  Backlash correction: overshoot to {overshoot:.6g} "
                      f"then approach {final_target:.6g} "
                      f"from {'above' if not scan_forward else 'below'} "
                      f"(FWHM={fly_backlash:.4g})")
            iface.move_start(overshoot)
            iface.wait_move(timeout=motor_timeout)
        iface.move_start(final_target)
        iface.wait_move(timeout=motor_timeout)

    if plot:
        _plot(result, motor_label, det_label, mode,
               title_prefix="Fly Scan")

    if verbose:
        print(result)

    return result


# =============================================================================
# Energy table and beamline functions (unchanged from original)
# =============================================================================

_COL_MONO_E    = 0
_COL_HARMONIC  = 1
_COL_UND_E     = 2
_COL_ROLL2     = 3
_COL_X2        = 4

table400 = [
    [10.0,   1,         10.03,   3.7e6,   -1393],
    [12.0,   1,         12.05,   3.3e6,    -843],
    [16.0,   3,         16.06,   3.0e6,    -362],
    [20.0,   3,         20.10,   3.7e6,    -586],
    [25.0,   3,         25.10,   3.3e6,    -463],
    [30.0,   3,         30.13,   3.4e6,    -582],
]


def get_energy_row(mono_e: float, table: list = None, tol: float = 0.5) -> dict:
    if table is None:
        table = table400
    best_idx  = None
    best_diff = float("inf")
    for i, row in enumerate(table):
        diff = abs(row[_COL_MONO_E] - mono_e)
        if diff < best_diff:
            best_diff = diff
            best_idx  = i
    if best_diff > tol:
        available = [row[_COL_MONO_E] for row in table]
        raise ValueError(f"No entry within {tol} keV of {mono_e} keV. Available: {available}")
    row = table[best_idx]
    return {
        "mono_e"   : row[_COL_MONO_E],
        "harmonic" : int(row[_COL_HARMONIC]),
        "und_e"    : row[_COL_UND_E],
        "roll2"    : row[_COL_ROLL2],
        "x2"       : row[_COL_X2],
        "row_index": best_idx,
    }


def _motor_move(motor_or_name, value, label, wait, verbose):
    if isinstance(motor_or_name, str):
        m      = epics.Motor(motor_or_name)
        mlabel = motor_or_name
    else:
        m      = motor_or_name
        mlabel = _safe_label(m)
    try:
        _mc = object.__getattribute__(m, "connected")
    except AttributeError:
        _mc = True
    if not _mc:
        raise ConnectionError(f"{label} motor not connected: {mlabel}")
    m.move(value, wait=wait)
    if verbose:
        print(f"  {label:10s} → {mlabel} = {value}  [motor]")


def _pv_put(pv_or_name, value, label, wait, verbose):
    if isinstance(pv_or_name, str):
        pv    = epics.PV(pv_or_name)
        pvlabel = pv_or_name
        if not pv.connect(timeout=5):
            raise ConnectionError(f"{label} PV not connected: {pv_or_name}")
    else:
        pv      = pv_or_name
        pvlabel = _safe_label(pv)
    pv.put(value, wait=wait)
    if verbose:
        print(f"  {label:10s} → {pvlabel} = {value}")


def set_energy(mono_e, table=None, tol=0.5,
               mono_e_pv="ID15A2:mono:Energy", harmonic_pv="ID15A2:und:Harmonic",
               und_e_pv="ID15A2:und:Energy", und_start_pv="ID15A2:und:Start",
               roll2_pv="ID15A2:mono:Roll2", x2_pv="ID15A2:mono:X2",
               wait=True, settle=1.0, simulate=False, verbose=True, debug=False) -> dict:
    if debug:
        verbose = False
    row = get_energy_row(mono_e, table=table, tol=tol)
    if verbose:
        print(f"Setting energy to {row['mono_e']} keV  (matched from request {mono_e} keV)")
        print(f"  Harmonic : {row['harmonic']}")
        print(f"  UndE     : {row['und_e']} keV")
        print(f"  Roll2    : {row['roll2']:.4g}")
        print(f"  X2       : {row['x2']}")
    def _lbl(p): return _safe_label(p)
    if simulate or not _EPICS_AVAILABLE:
        if verbose:
            print("  [SIMULATION – no PVs written]")
            print(f"    caput {_lbl(harmonic_pv)} {row['harmonic']}")
            print(f"    caput {_lbl(und_e_pv)} {row['und_e']}")
            print(f"    caput {_lbl(und_start_pv)} 1")
            print(f"    Motor.move {_lbl(roll2_pv)} {row['roll2']}")
            print(f"    Motor.move {_lbl(x2_pv)} {row['x2']}")
            print(f"    Motor.move {_lbl(mono_e_pv)} {row['mono_e']}")
    else:
        _pv_put(harmonic_pv, row["harmonic"], "Harmonic", wait, verbose)
        _pv_put(und_e_pv, row["und_e"], "UndE", wait, verbose)
        _pv_put(und_start_pv, 1, "UndStart", wait=False, verbose=verbose)
        _motor_move(roll2_pv, row["roll2"], "Roll2", wait, verbose)
        _motor_move(x2_pv,    row["x2"],   "X2",    wait, verbose)
        if isinstance(mono_e_pv, str):
            mono_motor = epics.Motor(mono_e_pv)
            mono_label = mono_e_pv
        else:
            mono_motor = mono_e_pv
            mono_label = _safe_label(mono_motor)
        try:
            _mc = object.__getattribute__(mono_motor, "connected")
        except AttributeError:
            _mc = True
        if not _mc:
            raise ConnectionError(f"MonoE motor not connected: {mono_label}")
        mono_motor.move(row["mono_e"], wait=wait)
        if verbose:
            print(f"  {'MonoE':10s} → {mono_label} = {row['mono_e']}  [motor]")
        if wait:
            epics.ca.poll()
        if settle > 0:
            if verbose:
                print(f"  Waiting {settle}s for beamline to settle …")
            time.sleep(settle)
    if verbose:
        print("  Done.")
    return row


def list_energies(table: list = None) -> None:
    if table is None:
        table = table400
    print(f"{'#':>3}  {'MonoE':>8}  {'Harmonic':>9}  {'UndE':>8}  {'Roll2':>10}  {'X2':>8}")
    print("  " + "─" * 52)
    for i, row in enumerate(table):
        print(f"{i:>3}  {row[_COL_MONO_E]:>8.1f}  {int(row[_COL_HARMONIC]):>9d}  "
              f"{row[_COL_UND_E]:>8.3f}  {row[_COL_ROLL2]:>10.3g}  {row[_COL_X2]:>8.0f}")


try:
    from scipy.interpolate import PchipInterpolator, CubicSpline, interp1d
    _SCIPY_INTERP = True
except ImportError:
    _SCIPY_INTERP = False


def interpolate_energy(mono_e, table=None, method="pchip",
                       extrapolate=False, verbose=False) -> dict:
    if table is None:
        table = table400
    if not _SCIPY_INTERP:
        raise ImportError("scipy is required for interpolation.")
    mono_vals  = np.array([row[_COL_MONO_E]   for row in table], dtype=float)
    harmonic   = np.array([row[_COL_HARMONIC]  for row in table], dtype=float)
    und_vals   = np.array([row[_COL_UND_E]     for row in table], dtype=float)
    roll2_vals = np.array([row[_COL_ROLL2]     for row in table], dtype=float)
    x2_vals    = np.array([row[_COL_X2]        for row in table], dtype=float)
    e_min, e_max = mono_vals.min(), mono_vals.max()
    if not extrapolate and not (e_min <= mono_e <= e_max):
        raise ValueError(f"mono_e={mono_e} keV outside range [{e_min}, {e_max}].")
    def _make_interp(x, y):
        if method == "pchip":
            return PchipInterpolator(x, y, extrapolate=extrapolate)
        elif method == "cubic":
            return CubicSpline(x, y, extrapolate=extrapolate)
        elif method == "linear":
            fill = "extrapolate" if extrapolate else (y[0], y[-1])
            return interp1d(x, y, kind="linear", bounds_error=not extrapolate, fill_value=fill)
        else:
            raise ValueError(f"Unknown method '{method}'.")
    result = {
        "mono_e"  : float(mono_e),
        "harmonic": int(round(float(_make_interp(mono_vals, harmonic)(mono_e)))),
        "und_e"   : float(_make_interp(mono_vals, und_vals)(mono_e)),
        "roll2"   : float(_make_interp(mono_vals, roll2_vals)(mono_e)),
        "x2"      : float(_make_interp(mono_vals, x2_vals)(mono_e)),
        "method"  : method,
    }
    if verbose:
        print(f"Interpolated values at {mono_e} keV  [{method}]:")
        print(f"  Harmonic : {result['harmonic']}")
        print(f"  UndE     : {result['und_e']:.4f} keV")
        print(f"  Roll2    : {result['roll2']:.4g}")
        print(f"  X2       : {result['x2']:.2f}")
    return result


def set_energy_interpolated(mono_e, table=None, method="pchip", extrapolate=False,
                            mono_e_pv="ID15A2:mono:Energy", harmonic_pv="ID15A2:und:Harmonic",
                            und_e_pv="ID15A2:und:Energy", und_start_pv="ID15A2:und:Start",
                            roll2_pv="ID15A2:mono:Roll2", x2_pv="ID15A2:mono:X2",
                            wait=True, settle=1.0, simulate=False, verbose=True, debug=False) -> dict:
    if debug:
        verbose = False
    row = interpolate_energy(mono_e, table=table, method=method,
                             extrapolate=extrapolate, verbose=verbose)
    def _lbl(p): return _safe_label(p)
    if simulate or not _EPICS_AVAILABLE:
        if verbose:
            print(f"  [SIMULATION – no PVs written]")
            print(f"    caput {_lbl(harmonic_pv)} {row['harmonic']}")
            print(f"    caput {_lbl(und_e_pv)} {row['und_e']:.6g}")
            print(f"    caput {_lbl(und_start_pv)} 1")
            print(f"    Motor.move {_lbl(roll2_pv)} {row['roll2']:.6g}")
            print(f"    Motor.move {_lbl(x2_pv)} {row['x2']:.6g}")
            print(f"    Motor.move {_lbl(mono_e_pv)} {row['mono_e']:.6g}")
    else:
        _pv_put(harmonic_pv,  row["harmonic"], "Harmonic", wait, verbose)
        _pv_put(und_e_pv,     row["und_e"],    "UndE",     wait, verbose)
        _pv_put(und_start_pv, 1,               "UndStart", False, verbose)
        _motor_move(roll2_pv, row["roll2"], "Roll2", wait, verbose)
        _motor_move(x2_pv,    row["x2"],   "X2",    wait, verbose)
        if isinstance(mono_e_pv, str):
            mono_motor = epics.Motor(mono_e_pv)
            mono_label = mono_e_pv
        else:
            mono_motor = mono_e_pv
            mono_label = _safe_label(mono_motor)
        try:
            _mc = object.__getattribute__(mono_motor, "connected")
        except AttributeError:
            _mc = True
        if not _mc:
            raise ConnectionError(f"MonoE motor not connected: {mono_label}")
        mono_motor.move(row["mono_e"], wait=wait)
        if verbose:
            print(f"  {'MonoE':10s} → {mono_label} = {row['mono_e']:.6g}  [motor]")
        if wait:
            epics.ca.poll()
        if settle > 0:
            if verbose:
                print(f"  Waiting {settle}s for beamline to settle …")
            time.sleep(settle)
    if verbose:
        print("  Done.")
    return row


def plot_interpolation(table=None, method="pchip", n_plot=300) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed – skipping plot.")
        return
    if table is None:
        table = table400
    mono_vals = np.array([row[_COL_MONO_E] for row in table], dtype=float)
    e_fine    = np.linspace(mono_vals.min(), mono_vals.max(), n_plot)
    cols = [(_COL_UND_E, "UndE (keV)", "#4C72B0"),
            (_COL_ROLL2, "Roll2",      "#DD8452"),
            (_COL_X2,    "X2",         "#55A868")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f"Energy table interpolation  [{method}]", fontsize=12, fontweight="bold")
    for ax, (col_idx, label, color) in zip(axes, cols):
        y_table = np.array([row[col_idx] for row in table], dtype=float)
        result_fine = [interpolate_energy(e, table=table, method=method) for e in e_fine]
        key = {_COL_UND_E: "und_e", _COL_ROLL2: "roll2", _COL_X2: "x2"}[col_idx]
        y_fine = np.array([r[key] for r in result_fine])
        ax.plot(e_fine, y_fine, "-", lw=2, color=color, label=method)
        ax.plot(mono_vals, y_table, "o", ms=7, color=color,
                markeredgecolor="white", markeredgewidth=1.2, zorder=5, label="table points")
        ax.set_xlabel("MonoE (keV)", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


def _set_energy_for_row(mono_e, table, mono_e_pv, harmonic_pv, und_e_pv,
                         und_start_pv, roll2_energy_pv, x2_energy_pv, interp_method,
                         energy_settle, simulate, verbose):
    if interp_method is None:
        set_energy(mono_e, table=table, mono_e_pv=mono_e_pv, harmonic_pv=harmonic_pv,
                   und_e_pv=und_e_pv, und_start_pv=und_start_pv,
                   roll2_pv=roll2_energy_pv, x2_pv=x2_energy_pv,
                   wait=True, settle=energy_settle, simulate=simulate, verbose=verbose, debug=False)
    else:
        set_energy_interpolated(mono_e, table=table, method=interp_method,
                                mono_e_pv=mono_e_pv, harmonic_pv=harmonic_pv,
                                und_e_pv=und_e_pv, und_start_pv=und_start_pv,
                                roll2_pv=roll2_energy_pv, x2_pv=x2_energy_pv,
                                wait=True, settle=energy_settle, simulate=simulate,
                                verbose=verbose, debug=False)


def align_beamline(
    table               : list,
    detector            = None,
    brg2                = None,
    roll2_motor         = None,
    x2_motor            = None,
    pitch_pv            : str   = None,
    slit_v_pv           : str   = None,
    slit_h_pv           : str   = None,
    slit_open_v         : float = None,
    slit_open_h         : float = None,
    slit_close_v        : float = None,
    slit_close_h        : float = None,
    pitch_home          : float = 5.0,
    do_pitch_scan       : bool  = True,
    pitch_peak_method   : str   = "stats",
    brg2_start              : float = -0.005,
    brg2_stop               : float =  0.005,
    brg2_nsteps             : int   = 21,
    brg2_min_prominence     : float = 3.0,
    pitch_start         : float = -1.0,
    pitch_stop          : float =  1.0,
    pitch_nsteps        : int   = 21,
    pitch_settle        : float = 0.1,
    pitch_soft_limits   : tuple = (-10.0, 10.0),
    pitch_extend_scan   : bool  = True,
    pitch_max_extend    : int   = 10,
    roll2_start         : float = -0.005,
    roll2_stop          : float =  0.005,
    roll2_nsteps        : int   = 21,
    x2_start            : float = -0.5,
    x2_stop             : float =  0.5,
    x2_nsteps           : int   = 21,
    settle              : float = 0.3,
    det_update_interval : float = 0.2,
    peak_method         : str   = "stats",
    stats_centre        : str   = "centroid",
    fit_profile         : str   = "auto",
    mono_e_pv           : object = None,
    harmonic_pv         : object = None,
    und_e_pv            : object = None,
    und_start_pv        : object = None,
    roll2_energy_pv     : object = None,
    x2_energy_pv        : object = None,
    energy_settle       : float = 2.0,
    filename            : str   = "alignment_results.csv",
    record_pvs          : dict  = None,
    record_settle       : float = 2.0,
    config              : object = None,
    interp_method       : str   = None,
    fine_scan           : bool  = True,
    fine_sigma_range    : float = 3.0,
    fine_nsteps         : int   = 21,
    fine_scan_iter      : int   = 2,
    backlash_correction : bool  = False,
    plot                : bool  = False,
    simulate            : bool  = False,
    verbose             : bool  = True,
    debug               : bool  = False,
    step_cb                     = None,
    row_cb                      = None,
) -> list:
    """
    Run a full beamline alignment sequence for every energy row in *table*.

    For each row the sequence is:
      a) Open slits
      b) Home pitch piezo
      c) smart_scan BRG2  → move to peak_pos
      d) fly_scan pitch   → move to peak
      e) Close vertical slit
      f) smart_scan Roll2 → move to centroid
      f2) fly_scan pitch repeat
      g-pre) Close horizontal slit
      g) smart_scan X2    → move to centroid
      h) Read final RBV of Roll2, X2, and all record_pvs; write CSV row

    CSV columns (in order):
      datetime | MonoE | Harmonic | UndE | Roll2 | X2 | <record_pvs keys>
    Roll2 and X2 are the actual post-scan RBV values, not nominal table values.
    """
    if debug:
        verbose = False

    # Unpack BeamlineConfig if provided
    if config is not None:
        detector            = config.detector
        brg2                = config.brg2
        roll2_motor         = config.roll2_motor
        x2_motor            = config.x2_motor
        pitch_pv            = config.pitch_pv
        slit_v_pv           = config.slit_v_pv
        slit_h_pv           = config.slit_h_pv
        slit_open_v         = config.slit_open_v
        slit_open_h         = config.slit_open_h
        slit_close_v        = config.slit_close_v
        slit_close_h        = config.slit_close_h
        mono_e_pv           = config.mono_e_pv
        harmonic_pv         = config.harmonic_pv
        und_e_pv            = config.und_e_pv
        und_start_pv        = config.und_start_pv
        roll2_energy_pv     = config.roll2_energy_pv
        x2_energy_pv        = config.x2_energy_pv
        pitch_home          = config.pitch_home
        do_pitch_scan       = config.do_pitch_scan
        pitch_peak_method   = config.pitch_peak_method
        pitch_start         = config.pitch_start
        pitch_stop          = config.pitch_stop
        pitch_nsteps        = config.pitch_nsteps
        pitch_settle        = config.pitch_settle
        pitch_soft_limits   = config.pitch_soft_limits
        pitch_extend_scan   = config.pitch_extend_scan
        pitch_max_extend    = config.pitch_max_extend
        brg2_start              = config.brg2_start
        brg2_stop               = config.brg2_stop
        brg2_nsteps             = config.brg2_nsteps
        brg2_min_prominence     = config.brg2_min_prominence
        roll2_start         = config.roll2_start
        roll2_stop          = config.roll2_stop
        roll2_nsteps        = config.roll2_nsteps
        x2_start            = config.x2_start
        x2_stop             = config.x2_stop
        x2_nsteps           = config.x2_nsteps
        fine_scan           = config.fine_scan
        fine_sigma_range    = config.fine_sigma_range
        fine_nsteps         = config.fine_nsteps
        fine_scan_iter      = config.fine_scan_iter
        plot                = config.plot
        backlash_correction = config.backlash_correction
        settle              = config.settle
        det_update_interval = config.det_update_interval
        peak_method         = config.peak_method
        stats_centre        = config.stats_centre
        fit_profile         = config.fit_profile
        energy_settle       = config.energy_settle
        record_pvs          = config.record_pvs
        record_settle       = config.record_settle
        filename            = config.filename

    import csv, os, datetime as _dt

    results = []

    # ── Resolve PV objects ────────────────────────────────────────────────────
    def _ensure_pv(pv_or_name):
        if isinstance(pv_or_name, str):
            return epics.PV(pv_or_name)
        return pv_or_name

    if not simulate and _EPICS_AVAILABLE:
        slit_v = _ensure_pv(slit_v_pv)
        slit_h = _ensure_pv(slit_h_pv)
        pitch  = _ensure_pv(pitch_pv)
    else:
        slit_v = slit_h = pitch = None

    pitch_axis = PVAxis(
        setpoint_pv = pitch_pv,
        settle_tol  = 0.01,
        settle_time = pitch_settle,
        soft_limits = pitch_soft_limits,
    ) if not simulate and _EPICS_AVAILABLE else None

    def _write_pv(pv_obj, value, label):
        if simulate or not _EPICS_AVAILABLE:
            if verbose:
                print(f"    [SIM] {label} = {value}")
        else:
            if isinstance(pv_obj, str):
                epics.PV(pv_obj).put(value, wait=True)
            elif hasattr(pv_obj, 'move'):
                pv_obj.move(value, wait=True)
            else:
                pv_obj.put(value, wait=True)
            if verbose:
                lbl = pv_obj if isinstance(pv_obj, str) else _safe_label(pv_obj)
                print(f"    {label} ({lbl}) = {value}")

    def _read_pv(pv_name):
        if simulate or not _EPICS_AVAILABLE:
            return float("nan")
        if isinstance(pv_name, str):
            v = epics.PV(pv_name).get(use_monitor=False)
        elif hasattr(pv_name, 'get_position'):
            v = pv_name.get_position()
        else:
            v = pv_name.get(use_monitor=False)
        return float(v) if v is not None else float("nan")

    # ── CSV setup ─────────────────────────────────────────────────────────────
    # Columns: datetime | MonoE | Harmonic | UndE | Roll2 | X2 | <record_pvs>
    # Roll2 and X2 are the actual post-scan RBV values, not nominal table values.
    fieldnames = ["datetime", "MonoE", "Harmonic", "UndE", "Roll2", "X2"]
    if record_pvs:
        fieldnames += list(record_pvs.keys())

    # Merge columns with any existing CSV rather than archiving it.
    # New columns (from added record_pvs) are appended; existing rows get "0".
    # Removed PVs whose columns already exist are kept; new rows get "0" via restval.
    if os.path.exists(filename):
        try:
            with open(filename, "r", newline="") as _f:
                existing_fields = csv.DictReader(_f).fieldnames or []
        except Exception:
            existing_fields = []

        if existing_fields:
            # Union: preserve existing column order, append any brand-new columns
            merged = list(existing_fields)
            added  = []
            for col in fieldnames:
                if col not in merged:
                    merged.append(col)
                    added.append(col)

            if added:
                # Rewrite the file with the extra columns filled as "0"
                try:
                    with open(filename, "r", newline="") as _f:
                        old_rows = list(csv.DictReader(_f))
                    with open(filename, "w", newline="") as _f:
                        _w = csv.DictWriter(_f, fieldnames=merged,
                                            extrasaction="ignore", restval="0")
                        _w.writeheader()
                        for _r in old_rows:
                            _w.writerow(_r)
                    if verbose:
                        print(f"  ℹ Added column(s) to CSV: {added} "
                              f"(existing rows filled with 0)")
                except Exception as _e:
                    if verbose:
                        print(f"  ⚠ Could not migrate CSV columns: {_e}")

            fieldnames = merged  # use the merged set for all new rows

    file_exists = os.path.exists(filename)
    csv_file    = open(filename, "a", newline="")
    # restval="0" fills in 0 for any column absent from a written record
    # (handles the case where a record_pv was removed — column stays, value is 0).
    writer      = csv.DictWriter(csv_file, fieldnames=fieldnames,
                                 extrasaction="ignore", restval="0")
    if not file_exists:
        writer.writeheader()

    # ── Main loop ─────────────────────────────────────────────────────────────
    for row_idx, row in enumerate(table):
        mono_e   = row[_COL_MONO_E]
        harmonic = int(row[_COL_HARMONIC])
        und_e    = row[_COL_UND_E]
        roll2_sp = row[_COL_ROLL2]
        x2_sp    = row[_COL_X2]

        sep = "═" * 60
        if verbose:
            print(f"\n{sep}")
            print(f"  Row {row_idx+1}/{len(table)}: MonoE={mono_e} keV  "
                  f"Harmonic={harmonic}  UndE={und_e}")
            print(sep)

        # record starts with energy table values; Roll2/X2 overwritten at step h
        record = {
            "datetime": "",
            "MonoE"   : mono_e,
            "Harmonic": harmonic,
            "UndE"    : und_e,
            "Roll2"   : roll2_sp,
            "X2"      : x2_sp,
        }

        r_brg2 = r_pitch = r_roll2 = r_x2 = None
        _row_ok = True
        try:
            # ── Set energy ────────────────────────────────────────────────────
            if step_cb: step_cb("Set energy")
            if verbose:
                print(f"\n  Setting energy …")
            _set_energy_for_row(
                mono_e=mono_e, table=table,
                mono_e_pv=mono_e_pv, harmonic_pv=harmonic_pv,
                und_e_pv=und_e_pv, und_start_pv=und_start_pv,
                roll2_energy_pv=roll2_energy_pv, x2_energy_pv=x2_energy_pv,
                interp_method=interp_method,
                energy_settle=energy_settle,
                simulate=simulate, verbose=verbose,
            )

            # ── a) Open slits ─────────────────────────────────────────────────
            if step_cb: step_cb("Open slits")
            if verbose:
                print(f"\n  a) Opening slits: V={slit_open_v}  H={slit_open_h}")
            _write_pv(slit_v, slit_open_v, f"slit_v → {slit_open_v}")
            _write_pv(slit_h, slit_open_h, f"slit_h → {slit_open_h}")
            time.sleep(5.0)

            # ── b) Home pitch piezo ───────────────────────────────────────────
            if step_cb: step_cb("Home pitch")
            if verbose:
                print(f"\n  b) Setting pitch piezo to {pitch_home}")
            _write_pv(pitch, pitch_home, f"pitch → {pitch_home}")
            time.sleep(pitch_settle * 3)

            # ── c) BRG2 smart_scan → move to peak_pos ────────────────────────
            if step_cb: step_cb("BRG2 scan")
            if verbose:
                print(f"\n  c) BRG2 smart_scan  [{brg2_start:+g} … {brg2_stop:+g}  "
                      f"{brg2_nsteps} steps]")
            if not simulate:
                r_brg2 = smart_scan(
                    brg2, detector,
                    start=brg2_start, stop=brg2_stop, nsteps=brg2_nsteps,
                    mode="max", settle=settle,
                    det_update_interval=det_update_interval,
                    fit_profile=fit_profile,
                    peak_method=peak_method, stats_centre=stats_centre,
                    min_prominence_ratio=brg2_min_prominence,
                    move_to_peak=True, move_target="peak_pos",
                    fine_scan=fine_scan, fine_sigma_range=fine_sigma_range,
                    fine_nsteps=fine_nsteps, fine_scan_iter=fine_scan_iter,
                    plot=plot,
                    backlash_correction=backlash_correction,
                    simulate=False, debug=not verbose,
                )
                if verbose:
                    cen_str = f"{r_brg2.center:.6g}" if r_brg2.center is not None else "n/a"
                    print(f"    BRG2: {r_brg2.status.value}  peak={cen_str}")
            else:
                if verbose:
                    print("    [SIM] BRG2 smart_scan skipped")

            # ── d) Pitch fly_scan ─────────────────────────────────────────────
            if do_pitch_scan:
                if step_cb: step_cb("Pitch scan")
                if verbose:
                    print(f"\n  d) Pitch fly_scan  [{pitch_start:+g} … {pitch_stop:+g}  "
                          f"{pitch_nsteps} steps]")
                if not simulate:
                    r_pitch = fly_scan(
                        pitch_axis, detector,
                        start=pitch_start, stop=pitch_stop, nsteps=pitch_nsteps,
                        mode="max",
                        det_update_interval=det_update_interval,
                        fit_profile=fit_profile,
                        peak_method=pitch_peak_method,
                        stats_centre=stats_centre,
                        extend_scan=pitch_extend_scan,
                        max_extend_steps=pitch_max_extend,
                        move_to_peak=False,
                        simulate=False, debug=not verbose,
                    )
                    if pitch_peak_method == "stats" and r_pitch.stats is not None:
                        pitch_best = float(r_pitch.stats.get(stats_centre)
                                           or r_pitch.stats["peak_pos"])
                    else:
                        pitch_best = r_pitch.center if r_pitch.center else float("nan")
                    if r_pitch.status == ScanStatus.SUCCESS and not np.isnan(pitch_best):
                        if verbose:
                            print(f"    Moving pitch to {pitch_best:.6g}")
                        _write_pv(pitch, pitch_best, "pitch")
                        time.sleep(pitch_settle * 3)
                    if verbose:
                        print(f"    Pitch: {r_pitch.status.value}  peak={pitch_best:.6g}")
                else:
                    if verbose:
                        print("    [SIM] Pitch fly_scan skipped")
            else:
                if verbose:
                    print("\n  d) Pitch scan disabled – skipping")

            # ── e) Close vertical slit ────────────────────────────────────────
            if step_cb: step_cb("Close V slit")
            if verbose:
                print(f"\n  e) Closing vertical slit: V={slit_close_v}")
            _write_pv(slit_v, slit_close_v, f"slit_v → {slit_close_v}")
            time.sleep(5.0)

            # ── f) Roll2 smart_scan → move to centroid ────────────────────────
            if step_cb: step_cb("Roll2 scan")
            if verbose:
                print(f"\n  f) Roll2 smart_scan  [{roll2_start:+g} … {roll2_stop:+g}  "
                      f"{roll2_nsteps} steps]")
            if not simulate:
                r_roll2 = smart_scan(
                    roll2_motor, detector,
                    start=roll2_start, stop=roll2_stop, nsteps=roll2_nsteps,
                    mode="max", settle=settle,
                    det_update_interval=det_update_interval,
                    fit_profile=fit_profile,
                    peak_method=peak_method, stats_centre=stats_centre,
                    move_to_peak=True, move_target="centroid",
                    fine_scan=fine_scan, fine_sigma_range=fine_sigma_range,
                    fine_nsteps=fine_nsteps, fine_scan_iter=fine_scan_iter,
                    plot=plot,
                    backlash_correction=backlash_correction,
                    simulate=False, debug=not verbose,
                )
                if verbose:
                    cen_str = f"{r_roll2.center:.6g}" if r_roll2.center is not None else "n/a"
                    print(f"    Roll2: {r_roll2.status.value}  peak={cen_str}")
            else:
                if verbose:
                    print("    [SIM] Roll2 smart_scan skipped")

            # ── f2) Pitch fly_scan repeat ─────────────────────────────────────
            if do_pitch_scan:
                if step_cb: step_cb("Pitch scan 2")
                if verbose:
                    print(f"\n  f2) Pitch fly_scan  [{pitch_start:+g} … {pitch_stop:+g}  "
                          f"{pitch_nsteps} steps]")
                if not simulate:
                    r_pitch2 = fly_scan(
                        pitch_axis, detector,
                        start=pitch_start, stop=pitch_stop, nsteps=pitch_nsteps,
                        mode="max",
                        det_update_interval=det_update_interval,
                        fit_profile=fit_profile,
                        peak_method=pitch_peak_method,
                        stats_centre=stats_centre,
                        extend_scan=pitch_extend_scan,
                        max_extend_steps=pitch_max_extend,
                        move_to_peak=False,
                        simulate=False, debug=not verbose,
                    )
                    if r_pitch2.status == ScanStatus.SUCCESS:
                        pitch_best2 = (
                            float(r_pitch2.stats.get(stats_centre)
                                  or r_pitch2.stats["peak_pos"])
                            if pitch_peak_method == "stats" and r_pitch2.stats
                            else r_pitch2.center
                        )
                        if pitch_best2 is not None and not np.isnan(pitch_best2):
                            if verbose:
                                print(f"    Moving pitch to {pitch_best2:.6g}")
                            _write_pv(pitch, pitch_best2, "pitch")
                            time.sleep(pitch_settle * 3)
                    if verbose:
                        print(f"    Pitch (f2): {r_pitch2.status.value}")
                else:
                    if verbose:
                        print("    [SIM] Pitch fly_scan (f2) skipped")
            else:
                if verbose:
                    print("\n  f2) Pitch scan disabled – skipping")

            # ── g-pre) Close horizontal slit ──────────────────────────────────
            if step_cb: step_cb("Close H slit")
            if verbose:
                print(f"\n  g-pre) Closing horizontal slit: H={slit_close_h}")
            _write_pv(slit_h, slit_close_h, f"slit_h → {slit_close_h}")
            time.sleep(5.0)

            # ── g) X2 smart_scan → move to centroid ───────────────────────────
            if step_cb: step_cb("X2 scan")
            if verbose:
                print(f"\n  g) X2 smart_scan  [{x2_start:+g} … {x2_stop:+g}  "
                      f"{x2_nsteps} steps]")
            if not simulate:
                r_x2 = smart_scan(
                    x2_motor, detector,
                    start=x2_start, stop=x2_stop, nsteps=x2_nsteps,
                    mode="max", settle=settle,
                    det_update_interval=det_update_interval,
                    fit_profile=fit_profile,
                    peak_method=peak_method, stats_centre=stats_centre,
                    move_to_peak=True, move_target="centroid",
                    fine_scan=fine_scan, fine_sigma_range=fine_sigma_range,
                    fine_nsteps=fine_nsteps, fine_scan_iter=fine_scan_iter,
                    plot=plot,
                    backlash_correction=backlash_correction,
                    simulate=False, debug=not verbose,
                )
                if verbose:
                    cen_str = f"{r_x2.center:.6g}" if r_x2.center is not None else "n/a"
                    print(f"    X2: {r_x2.status.value}  peak={cen_str}")
            else:
                if verbose:
                    print("    [SIM] X2 smart_scan skipped")

        except KeyboardInterrupt:
            if verbose:
                print(f"\n  ⚠ Interrupted at MonoE={mono_e} keV – "
                      f"saving partial results and stopping.")
            # Read whatever RBVs we can before saving
            record["Roll2"] = _read_pv(roll2_motor)
            record["X2"]    = _read_pv(x2_motor)
            if record_pvs:
                for label, pv_name in record_pvs.items():
                    record[label] = _read_pv(pv_name)
            record["datetime"] = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow(record)
            csv_file.flush()
            results.append(record)
            csv_file.close()
            raise

        except Exception as exc:
            import traceback as _tb
            _row_ok = False
            if verbose:
                print(f"\n  ✗ Error at MonoE={mono_e} keV: {exc}")
                print(_tb.format_exc())

        # ── h) Read final RBV values and write CSV row ────────────────────────
        if step_cb: step_cb("Record results")
        if verbose:
            print(f"\n  h) Recording optimised values …")
        if record_settle > 0 and not simulate:
            if verbose:
                print(f"  Waiting {record_settle}s for motors/detector to settle …")
            time.sleep(record_settle)

        # Overwrite Roll2/X2 with actual post-scan RBV
        record["Roll2"] = _read_pv(roll2_motor)
        record["X2"]    = _read_pv(x2_motor)
        if verbose:
            print(f"    Roll2 (RBV) = {record['Roll2']:.6g}")
            print(f"    X2    (RBV) = {record['X2']:.6g}")

        # Read all extra PVs
        if record_pvs:
            for label, pv_name in record_pvs.items():
                val = _read_pv(pv_name)
                record[label] = val
                if verbose:
                    lbl = pv_name if isinstance(pv_name, str) else _safe_label(pv_name)
                    print(f"    {label} ({lbl}) = {val:.6g}")

        record["datetime"] = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow(record)
        csv_file.flush()
        record["_brg2_center"]  = r_brg2.center  if r_brg2  and r_brg2.center  is not None else float("nan")
        record["_roll2_center"] = r_roll2.center if r_roll2 and r_roll2.center is not None else float("nan")
        record["_x2_center"]   = r_x2.center    if r_x2    and r_x2.center    is not None else float("nan")
        record["_row_ok"]      = _row_ok
        results.append(record)
        if row_cb: row_cb(record)

        if verbose:
            print(f"\n  ✓ MonoE={mono_e} keV complete.  "
                  f"Results appended to {filename}")

    csv_file.close()

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  Alignment complete.  {len(results)} rows written to {filename}")
        print(f"{'═'*60}")

    return results

# ── Self-tests ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    np.random.seed(42)

    print("── Test 1: peak well inside range – verify _choose_centre ──────")
    r = smart_scan("IOC:m1", "IOC:det1", -4, 4, nsteps=21,
                   mode="max", settle=0, simulate=True,
                   sim_center=1.5, sim_sigma=1.2, sim_amplitude=100,
                   sim_noise=2, move_to_peak=False, fine_scan=False,
                   plot=False)
    assert r.status == ScanStatus.SUCCESS, f"Expected SUCCESS, got {r.status}"
    assert abs(r.center - 1.5) < 0.4, f"Centre off: {r.center}"
    # Verify _choose_centre returns a sensible position
    chosen = _choose_centre(r, verbose=True)
    assert r.positions.min() <= chosen <= r.positions.max(), \
        f"Chosen centre {chosen} outside scan range!"
    print("  PASS\n")

    print("── Test 2: peak near right edge → extension ─────────────────────")
    np.random.seed(7)
    r2 = smart_scan("IOC:m1", "IOC:det1", -4, 2.5, nsteps=21,
                    mode="max", settle=0, simulate=True,
                    sim_center=3.5, sim_sigma=0.8, sim_amplitude=80,
                    sim_noise=2, move_to_peak=False, plot=False)
    assert r2.status == ScanStatus.SUCCESS, f"Expected SUCCESS, got {r2.status}"
    assert abs(r2.center - 3.5) < 0.5, f"Centre off: {r2.center}"
    print("  PASS\n")

    print("── Test 3: flat signal → falls back to raw max ──────────────────")
    np.random.seed(0)
    r3 = smart_scan("IOC:m1", "IOC:det1", -4, 4, nsteps=21,
                    mode="max", settle=0, simulate=True,
                    sim_center=0, sim_sigma=1, sim_amplitude=0.5,
                    sim_noise=2, move_to_peak=False, plot=False)
    assert r3.status == ScanStatus.SUCCESS, f"Expected SUCCESS (fallback), got {r3.status}"
    assert "raw maximum" in r3.message, f"Expected fallback message, got: {r3.message}"
    assert r3.center is not None, "Expected a center position from raw max fallback"
    print("  PASS\n")

    print("── Test 4: _choose_centre logic directly ────────────────────────")
    # Symmetric peak: FWHM >> D → should pick centroid (corrected rule)
    r_sym = smart_scan("IOC:m1", "IOC:det1", -4, 4, nsteps=31,
                       mode="max", settle=0, simulate=True,
                       sim_center=0.0, sim_sigma=1.0, sim_amplitude=200,
                       sim_noise=0.1, move_to_peak=False, fine_scan=False, plot=False)
    if r_sym.status == ScanStatus.SUCCESS and r_sym.stats is not None:
        s = r_sym.stats
        fwhm = s.get("fwhm_interp", float("nan"))
        D    = abs(s.get("peak_pos", 0) - s.get("centroid", 0))
        chosen = _choose_centre(r_sym, verbose=True)
        if not np.isnan(fwhm) and fwhm > D:
            assert chosen == s["centroid"], \
                f"Expected centroid={s['centroid']:.5g}, got {chosen:.5g}"
            print(f"  Broad peak (FWHM={fwhm:.4g} > D={D:.4g}) → centroid chosen ✓")
        else:
            assert chosen == s["peak_pos"], \
                f"Expected peak_pos={s['peak_pos']:.5g}, got {chosen:.5g}"
            print(f"  Sharp peak (FWHM={fwhm:.4g} ≤ D={D:.4g}) → peak_pos chosen ✓")
    print("  PASS\n")

    print("── Test 5: move_target='peak_pos' always returns peak_pos ──────")
    np.random.seed(42)
    r5 = smart_scan("IOC:m1", "IOC:det1", -4, 4, nsteps=21,
                    mode="max", settle=0, simulate=True,
                    sim_center=1.5, sim_sigma=1.2, sim_amplitude=100,
                    sim_noise=2, move_to_peak=False, fine_scan=False, plot=False, debug=True)
    if r5.status == ScanStatus.SUCCESS and r5.stats:
        chosen = _choose_centre(r5, verbose=True, move_target="peak_pos")
        assert chosen == r5.stats["peak_pos"], f"Expected peak_pos, got {chosen}"
        print(f"  peak_pos={r5.stats['peak_pos']:.5g}  chosen={chosen:.5g}")
    print("  PASS\n")

    print("── Test 6: move_target='centroid' always returns centroid ───────")
    np.random.seed(42)
    r6 = smart_scan("IOC:m1", "IOC:det1", -4, 4, nsteps=21,
                    mode="max", settle=0, simulate=True,
                    sim_center=1.5, sim_sigma=1.2, sim_amplitude=100,
                    sim_noise=2, move_to_peak=False, fine_scan=False, plot=False, debug=True)
    if r6.status == ScanStatus.SUCCESS and r6.stats:
        chosen = _choose_centre(r6, verbose=True, move_target="centroid")
        assert chosen == r6.stats["centroid"], f"Expected centroid, got {chosen}"
        print(f"  centroid={r6.stats['centroid']:.5g}  chosen={chosen:.5g}")
    print("  PASS\n")

    print("All tests passed ✓")


# =============================================================================
# BeamlineConfig
# =============================================================================

@dataclass
class BeamlineConfig:
    """
    Holds all PV connections, motors, and scan parameters needed by
    align_beamline().  Create once with setup_beamline(), validate
    with .check(), then pass to align_beamline(config=...).
    """
    detector            : object  = None
    brg2                : object  = None
    roll2_motor         : object  = None
    x2_motor            : object  = None
    pitch_pv            : object  = None
    slit_v_pv           : object  = None
    slit_h_pv           : object  = None
    slit_open_v         : float   = 0.5
    slit_open_h         : float   = 0.5
    slit_close_v        : float   = 0.05
    slit_close_h        : float   = 0.05
    mono_e_pv           : object  = None
    harmonic_pv         : object  = None
    und_e_pv            : object  = None
    und_start_pv        : object  = None
    roll2_energy_pv     : object  = None
    x2_energy_pv        : object  = None
    pitch_home          : float   = 5.0
    do_pitch_scan       : bool    = True
    pitch_peak_method   : str     = "stats"
    pitch_start         : float   = -1.0
    pitch_stop          : float   =  1.0
    pitch_nsteps        : int     = 21
    pitch_settle        : float   = 0.1
    pitch_soft_limits   : tuple   = (-10.0, 10.0)
    pitch_extend_scan   : bool    = True
    pitch_max_extend    : int     = 10
    brg2_start              : float   = -0.005
    brg2_stop               : float   =  0.005
    brg2_nsteps             : int     = 21
    brg2_min_prominence     : float   = 3.0   # lower than default 5.0 – rocking curve is sharp
    roll2_start         : float   = -0.005
    roll2_stop          : float   =  0.005
    roll2_nsteps        : int     = 21
    x2_start            : float   = -0.5
    x2_stop             : float   =  0.5
    x2_nsteps           : int     = 21
    fine_scan           : bool    = True
    fine_sigma_range    : float   = 3.0
    fine_nsteps         : int     = 21
    fine_scan_iter      : int     = 2
    plot                : bool    = False   # show diagnostic plot after each scan
    backlash_correction : bool    = False   # overshoot by FWHM to eliminate backlash
    settle              : float   = 0.3
    det_update_interval : float   = 0.2
    peak_method         : str     = "stats"
    stats_centre        : str     = "centroid"
    fit_profile         : str     = "auto"
    energy_settle       : float   = 2.0
    record_pvs          : dict    = None
    record_settle       : float   = 2.0
    filename            : str     = "alignment_results.csv"

    def check(self, verbose: bool = True) -> bool:
        errors   = []
        warnings_list = []

        required_fields = {
            "detector"       : self.detector,
            "brg2"           : self.brg2,
            "roll2_motor"    : self.roll2_motor,
            "x2_motor"       : self.x2_motor,
            "pitch_pv"       : self.pitch_pv,
            "slit_v_pv"      : self.slit_v_pv,
            "slit_h_pv"      : self.slit_h_pv,
            "mono_e_pv"      : self.mono_e_pv,
            "harmonic_pv"    : self.harmonic_pv,
            "und_e_pv"       : self.und_e_pv,
            "roll2_energy_pv": self.roll2_energy_pv,
            "x2_energy_pv"   : self.x2_energy_pv,
        }

        for name, val in required_fields.items():
            if val is None:
                errors.append(f"  ✗ {name} is not set")

        if verbose:
            print("BeamlineConfig check:")
            for name, val in required_fields.items():
                label  = val if isinstance(val, str) else _safe_label(val)
                status = "✓" if val is not None else "✗ NOT SET"
                print(f"  {status}  {name:20s} : {label}")

        if _EPICS_AVAILABLE and not errors:
            pv_names = {
                "pitch_pv"       : self.pitch_pv,
                "slit_v_pv"      : self.slit_v_pv,
                "slit_h_pv"      : self.slit_h_pv,
                "mono_e_pv"      : self.mono_e_pv,
                "harmonic_pv"    : self.harmonic_pv,
                "und_e_pv"       : self.und_e_pv,
                "roll2_energy_pv": self.roll2_energy_pv,
                "x2_energy_pv"   : self.x2_energy_pv,
            }
            if verbose:
                print("\n  PV connection test:")
            for name, pv_or_name in pv_names.items():
                if pv_or_name is None:
                    if verbose:
                        print(f"    {'✗ NOT SET':15s}  {name}")
                    continue
                if isinstance(pv_or_name, str):
                    pv      = epics.PV(pv_or_name)
                    pvlabel = pv_or_name
                    ok      = pv.connect(timeout=3)
                else:
                    pvlabel = _safe_label(pv_or_name)
                    try:
                        ok = object.__getattribute__(pv_or_name, "connected")
                    except AttributeError:
                        ok = True
                status = "✓ connected" if ok else "✗ TIMEOUT"
                if not ok:
                    warnings_list.append(f"  ⚠ {name} ({pvlabel}) did not connect")
                if verbose:
                    print(f"    {status:15s}  {name:20s} : {pvlabel}")

        if errors:
            if verbose:
                print("\nErrors (must fix before running):")
                for e in errors:
                    print(e)
            return False

        if warnings_list:
            if verbose:
                print("\nWarnings (PV connection issues):")
                for w in warnings_list:
                    print(w)

        if verbose and not errors:
            ok = len(warnings_list) == 0
            print(f"\n{'✓ All checks passed.' if ok else '⚠ Passed with warnings.'}")

        return len(errors) == 0

    def summary(self) -> None:
        print("═" * 55)
        print("  BeamlineConfig summary")
        print("═" * 55)
        sections = [
            ("Detector", [("detector", self.detector)]),
            ("Motors / axes", [
                ("brg2",        self.brg2),
                ("roll2_motor", self.roll2_motor),
                ("x2_motor",    self.x2_motor),
                ("pitch_pv",    self.pitch_pv),
            ]),
            ("Slits", [
                ("slit_v_pv",      self.slit_v_pv),
                ("slit_h_pv",      self.slit_h_pv),
                ("slit_open_v/h",  f"{self.slit_open_v} / {self.slit_open_h}"),
                ("slit_close_v/h", f"{self.slit_close_v} / {self.slit_close_h}"),
            ]),
            ("Energy table PVs", [
                ("mono_e_pv",       self.mono_e_pv),
                ("harmonic_pv",     self.harmonic_pv),
                ("und_e_pv",        self.und_e_pv),
                ("roll2_energy_pv", self.roll2_energy_pv),
                ("x2_energy_pv",    self.x2_energy_pv),
            ]),
            ("Scan ranges (relative)", [
                ("brg2",  f"{self.brg2_start:+g} … {self.brg2_stop:+g}  ({self.brg2_nsteps} steps)"),
                ("pitch", f"{self.pitch_start:+g} … {self.pitch_stop:+g}  ({self.pitch_nsteps} steps)  home={self.pitch_home}"),
                ("roll2", f"{self.roll2_start:+g} … {self.roll2_stop:+g}  ({self.roll2_nsteps} steps)"),
                ("x2",    f"{self.x2_start:+g} … {self.x2_stop:+g}  ({self.x2_nsteps} steps)"),
            ]),
            ("Shared settings", [
                ("fine_scan",           str(self.fine_scan)),
                ("fine_sigma_range",    f"{self.fine_sigma_range} × σ"),
                ("fine_nsteps",         str(self.fine_nsteps)),
                ("settle",              f"{self.settle} s"),
                ("det_update_interval", f"{self.det_update_interval} s"),
                ("fit_profile",         self.fit_profile),
                ("energy_settle",       f"{self.energy_settle} s"),
            ]),
            ("Output", [
                ("filename",    self.filename),
                ("record_pvs",  str(self.record_pvs)),
                ("record_settle", f"{self.record_settle} s"),
            ]),
        ]
        for section, items in sections:
            print(f"\n  {section}")
            print("  " + "─" * 40)
            for label, val in items:
                v = val if isinstance(val, str) else _safe_label(val)
                print(f"    {label:25s}: {v}")
        print("═" * 55)


# =============================================================================
# setup_beamline
# =============================================================================

def setup_beamline(
    detector,
    brg2,
    roll2_motor,
    x2_motor,
    pitch_pv            : object,
    slit_v_pv           : object,
    slit_h_pv           : object,
    slit_open_v         : float,
    slit_open_h         : float,
    slit_close_v        : float,
    slit_close_h        : float,
    mono_e_pv           : object,
    harmonic_pv         : object,
    und_e_pv            : object,
    und_start_pv        : object,
    roll2_energy_pv     : object,
    x2_energy_pv        : object,
    brg2_start          : float = -0.005,
    brg2_stop           : float =  0.005,
    brg2_nsteps         : int   = 21,
    brg2_min_prominence : float = 3.0,
    pitch_home          : float = 5.0,
    do_pitch_scan       : bool  = True,
    pitch_peak_method   : str   = "stats",
    pitch_start         : float = -1.0,
    pitch_stop          : float =  1.0,
    pitch_nsteps        : int   = 21,
    pitch_settle        : float = 0.1,
    pitch_soft_limits   : tuple = (-10.0, 10.0),
    pitch_extend_scan   : bool  = True,
    pitch_max_extend    : int   = 10,
    roll2_start         : float = -0.005,
    roll2_stop          : float =  0.005,
    roll2_nsteps        : int   = 21,
    x2_start            : float = -0.5,
    x2_stop             : float =  0.5,
    x2_nsteps           : int   = 21,
    fine_scan           : bool  = True,
    fine_sigma_range    : float = 3.0,
    fine_nsteps         : int   = 21,
    fine_scan_iter      : int   = 2,
    plot                : bool  = False,
    backlash_correction : bool  = False,
    settle              : float = 0.3,
    det_update_interval : float = 0.2,
    peak_method         : str   = "stats",
    stats_centre        : str   = "centroid",
    fit_profile         : str   = "auto",
    energy_settle       : float = 2.0,
    record_pvs          : dict  = None,
    record_settle       : float = 2.0,
    filename            : str   = "alignment_results.csv",
    verbose             : bool  = True,
) -> BeamlineConfig:
    """
    Create, validate, and return a BeamlineConfig ready for align_beamline().
    """
    cfg = BeamlineConfig(
        detector=detector, brg2=brg2, roll2_motor=roll2_motor,
        x2_motor=x2_motor, pitch_pv=pitch_pv,
        slit_v_pv=slit_v_pv, slit_h_pv=slit_h_pv,
        slit_open_v=slit_open_v, slit_open_h=slit_open_h,
        slit_close_v=slit_close_v, slit_close_h=slit_close_h,
        mono_e_pv=mono_e_pv, harmonic_pv=harmonic_pv,
        und_e_pv=und_e_pv, und_start_pv=und_start_pv,
        roll2_energy_pv=roll2_energy_pv, x2_energy_pv=x2_energy_pv,
        brg2_start=brg2_start, brg2_stop=brg2_stop, brg2_nsteps=brg2_nsteps,
        brg2_min_prominence=brg2_min_prominence,
        pitch_home=pitch_home, do_pitch_scan=do_pitch_scan,
        pitch_peak_method=pitch_peak_method,
        pitch_start=pitch_start, pitch_stop=pitch_stop,
        pitch_nsteps=pitch_nsteps, pitch_settle=pitch_settle,
        pitch_soft_limits=pitch_soft_limits,
        pitch_extend_scan=pitch_extend_scan, pitch_max_extend=pitch_max_extend,
        roll2_start=roll2_start, roll2_stop=roll2_stop, roll2_nsteps=roll2_nsteps,
        x2_start=x2_start, x2_stop=x2_stop, x2_nsteps=x2_nsteps,
        fine_scan=fine_scan, fine_sigma_range=fine_sigma_range,
        fine_nsteps=fine_nsteps, fine_scan_iter=fine_scan_iter,
        plot=plot,
        backlash_correction=backlash_correction,
        settle=settle, det_update_interval=det_update_interval,
        peak_method=peak_method, stats_centre=stats_centre,
        fit_profile=fit_profile, energy_settle=energy_settle,
        record_pvs=record_pvs, record_settle=record_settle, filename=filename,
    )
    if verbose:
        cfg.summary()
        print()
        cfg.check(verbose=True)
    return cfg


# =============================================================================
# align_energy
# =============================================================================

def align_energy(
    mono_e        : float,
    config        : "BeamlineConfig",
    table         : list  = None,
    interp_method : str   = "pchip",
    extrapolate   : bool  = False,
    simulate      : bool  = False,
    verbose       : bool  = True,
    debug         : bool  = False,
) -> dict:
    """
    Run the full alignment sequence for a single energy using interpolation.
    """
    if debug:
        verbose = False
    if table is None:
        table = table400
    mono_vals = [row[_COL_MONO_E] for row in table]
    e_min, e_max = min(mono_vals), max(mono_vals)
    if not extrapolate and not (e_min <= mono_e <= e_max):
        raise ValueError(
            f"mono_e={mono_e} keV is outside the table range "
            f"[{e_min}, {e_max}] keV.  Set extrapolate=True to proceed anyway."
        )
    row = interpolate_energy(mono_e, table=table, method=interp_method,
                             extrapolate=extrapolate, verbose=verbose)
    if verbose:
        print(f"\nalign_energy: {mono_e} keV  (interpolated via {interp_method})")
    synthetic_row = [row["mono_e"], row["harmonic"], row["und_e"],
                     row["roll2"], row["x2"]]
    results = align_beamline(
        table         = [synthetic_row],
        config        = config,
        interp_method = interp_method,
        simulate      = simulate,
        verbose       = verbose,
        debug         = debug,
    )
    return results[0] if results else {}


# =============================================================================
# test_motor_compat
# =============================================================================

def test_motor_compat(
    motor,
    test_move     : bool  = False,
    move_delta    : float = 0.001,
    motor_timeout : float = 10.0,
    verbose       : bool  = True,
) -> dict:
    """
    Test that a motor object is compatible with smart_scan and fly_scan.
    """
    import traceback as _tb
    checks = {}
    label  = "unknown"

    def _record(name, passed, detail=""):
        checks[name] = {"pass": passed, "detail": detail}
        icon = "✓" if passed else "✗"
        if verbose:
            print(f"  {icon} {name:35s}  {detail}")

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  Motor compatibility test")
        print(f"{'═'*60}")

    # 1. Type detection
    if isinstance(motor, str):
        motor_type = "string (PV name)"
    elif isinstance(motor, PVAxis):
        motor_type = "PVAxis"
    elif _EPICS_AVAILABLE:
        if isinstance(motor, epics.Motor):
            motor_type = "epics.Motor"
        elif isinstance(motor, epics.PV):
            motor_type = "epics.PV"
        else:
            motor_type = type(motor).__name__
    else:
        motor_type = type(motor).__name__
    _record("1. Type detection", True, motor_type)

    # 2. Label resolution
    try:
        label = _safe_label(motor)
        ok    = bool(label) and label != 'None'
        _record("2. _safe_label()", ok, repr(label))
    except Exception as e:
        _record("2. _safe_label()", False, str(e))

    # 3. _Interface construction
    try:
        iface = _Interface(motor, "sim_det", simulate=True,
                           sim_center=0.0, sim_sigma=1.0,
                           sim_amp=100.0, sim_offset=5.0, sim_noise=1.0)
        _record("3. _Interface construction", True)
    except Exception as e:
        _record("3. _Interface construction", False, str(e))
        iface = None

    # 4. _FlyInterface construction
    try:
        fiface = _FlyInterface(motor, "sim_det", simulate=True,
                               sim_center=0.0, sim_sigma=1.0,
                               sim_amp=100.0, sim_offset=5.0,
                               sim_noise=1.0, sim_velocity=1.0)
        _record("4. _FlyInterface construction", True)
    except Exception as e:
        _record("4. _FlyInterface construction", False, str(e))
        fiface = None

    # 5. Position read
    _is_plain_pv = (_EPICS_AVAILABLE and not isinstance(motor, str)
                    and not isinstance(motor, PVAxis)
                    and isinstance(motor, epics.PV)
                    and not isinstance(motor, epics.Motor))
    if _is_plain_pv:
        _record("5. Position read", True, "N/A – plain epics.PV")
    elif iface is not None:
        try:
            pos = iface.position()
            _record("5. Position read", True, f"{pos:.6g}")
        except Exception as e:
            _record("5. Position read", False, str(e))

    # 6. Soft limits
    if _is_plain_pv:
        _record("6. Soft limits", True, "N/A – plain epics.PV")
    elif fiface is not None:
        try:
            lo, hi = fiface.soft_limits()
            _record("6. Soft limits", isinstance(lo, (int, float)), f"({lo:.4g}, {hi:.4g})")
        except Exception as e:
            _record("6. Soft limits", False, str(e))

    # 7. Velocity read
    if _is_plain_pv:
        _record("7. Velocity read", True, "N/A – plain epics.PV")
    elif fiface is not None:
        try:
            import math
            vel = fiface.get_velocity()
            ok  = isinstance(vel, (int, float))
            _record("7. Velocity read", ok,
                    f"{vel:.4g}" if not math.isnan(vel) else "nan (no VELO – expected for PVAxis)")
        except Exception as e:
            _record("7. Velocity read", False, str(e))

    # 8. Optional physical move
    if test_move and _EPICS_AVAILABLE and not isinstance(motor, str):
        try:
            if iface is not None:
                start_pos = iface.position()
                ok_fwd    = iface.move(start_pos + move_delta, timeout=motor_timeout)
                pos_fwd   = iface.position()
                ok_ret    = iface.move(start_pos, timeout=motor_timeout)
                pos_ret   = iface.position()
                ok = ok_fwd and ok_ret and abs(pos_fwd - start_pos) > move_delta * 0.1
                _record("8. Physical move", ok,
                        f"start={start_pos:.5g} → {pos_fwd:.5g} → {pos_ret:.5g}")
            else:
                _record("8. Physical move", False, "Interface not created")
        except Exception as e:
            _record("8. Physical move", False, str(e))
    else:
        _record("8. Physical move", True,
                "skipped (test_move=False)" if not test_move else "skipped (no epics)")

    # 9. smart_scan compatibility
    try:
        _seed = np.random.get_state()
        np.random.seed(0)
        r = smart_scan(motor, "sim_det", start=-1.0, stop=1.0, nsteps=11,
                       settle=0, simulate=True, sim_center=0.0, sim_sigma=0.3,
                       sim_amplitude=100, sim_noise=2, move_to_peak=False,
                       fine_scan=False, plot=False, debug=True)
        np.random.set_state(_seed)
        ok = r.status in (ScanStatus.SUCCESS, ScanStatus.NO_PEAK)
        _record("9. smart_scan (simulated)", ok,
                f"status={r.status.value}  centre={r.center}")
    except Exception as e:
        _record("9. smart_scan (simulated)", False, _tb.format_exc().splitlines()[-1])

    # 10. fly_scan compatibility
    try:
        _seed = np.random.get_state()
        np.random.seed(0)
        r2 = fly_scan(motor, "sim_det", start=-1.0, stop=1.0,
                      sample_interval=0.05, nsteps=11, simulate=True,
                      sim_center=0.0, sim_sigma=0.3, sim_amplitude=100,
                      sim_noise=2, sim_velocity=2.0, move_to_peak=False,
                      plot=False, debug=True)
        np.random.set_state(_seed)
        ok = r2.status in (ScanStatus.SUCCESS, ScanStatus.NO_PEAK)
        _record("10. fly_scan (simulated)", ok,
                f"status={r2.status.value}  centre={r2.center}")
    except Exception as e:
        _record("10. fly_scan (simulated)", False, _tb.format_exc().splitlines()[-1])

    mandatory  = [k for k in checks if not k.startswith("8.")]
    all_passed = all(checks[k]["pass"] for k in mandatory)

    if verbose:
        print(f"\n{'═'*60}")
        if all_passed:
            print(f"  ✓ {motor_type} '{label}' is fully compatible.")
        else:
            failed = [k for k in mandatory if not checks[k]["pass"]]
            print(f"  ✗ {len(failed)} check(s) failed: {', '.join(failed)}")
        print(f"{'═'*60}\n")

    return {"passed": all_passed, "checks": checks,
            "motor_label": label, "motor_type": motor_type}
