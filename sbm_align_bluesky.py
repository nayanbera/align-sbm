"""
sbm_align_bluesky.py — Bluesky plan replicating the ID15A2 SBM alignment protocol.

This plan is the Bluesky equivalent of ``align_sbm.align_beamline()``.

Device requirements
-------------------
brg2, roll2, x2 : ophyd EpicsMotor (or any Positioner with .position / .set())
pitch            : ophyd Positioner wrapping the pitch piezo setpoint + readback.
                   The original app uses a bare EpicsSignal (setpoint-only PVAxis).
                   For Bluesky you need a device that satisfies the Positioner interface.
                   A minimal wrapper is shown in ``make_pv_axis()`` below.
slit_v, slit_h   : ophyd EpicsSignal (setpoint) — no readback required.
detector         : any ophyd readable Device; only its trigger/read cycle is used.
monitor_det      : optional second readable Device; when supplied its value is used
                   to normalise the detector reading (det / monitor) in the CSV record.

Minimal usage
-------------
    from sbm_align_bluesky import sbm_align, make_pv_axis
    from ophyd import EpicsMotor, EpicsSignal
    import bluesky.plans as bp

    brg2      = EpicsMotor("ID15A2:BRG2",    name="brg2")
    roll2     = EpicsMotor("ID15A2:Roll2",   name="roll2")
    x2        = EpicsMotor("ID15A2:X2",      name="x2")
    pitch     = make_pv_axis("ID15A2:PitchPiezo:SP",
                              "ID15A2:PitchPiezo:RBV", name="pitch")
    slit_v    = EpicsSignal("ID15A2:SlitV:SP",  name="slit_v")
    slit_h    = EpicsSignal("ID15A2:SlitH:SP",  name="slit_h")
    det       = EpicsSignal("ID15A2:det:Signal", name="det")

    RE(sbm_align(table400, brg2=brg2, roll2=roll2, x2=x2,
                 pitch=pitch, slit_v=slit_v, slit_h=slit_h, detector=det))

Fly-scan note
-------------
The original app uses a continuous-motion fly_scan for the pitch steps.
Bluesky fly scans require hardware-specific Flyer objects.  This plan
substitutes a stepped rel_scan, which is equivalent at the peak-finding level.
To restore the continuous-motion behaviour, replace the pitch scan calls with
``yield from bp.fly([pitch_flyer], ...)`` and adapt ``_find_peak_from_uid``.
"""

from __future__ import annotations

import csv
import datetime
import os
from typing import Optional

import numpy as np

import bluesky.plan_stubs as bps
import bluesky.plans as bp
from bluesky.callbacks.fitting import PeakStats
from bluesky.preprocessors import subs_wrapper


# ---------------------------------------------------------------------------
# Minimal PVAxis positioner (pitch piezo or any setpoint-only axis)
# ---------------------------------------------------------------------------

def make_pv_axis(setpoint_pv: str, readback_pv: str, *, name: str,
                 settle_time: float = 0.1, timeout: float = 5.0):
    """
    Return a lightweight ophyd PseudoPositioner-style device for a
    setpoint + readback PV pair that has no DMOV signal (e.g. a piezo).

    Wraps ``ophyd.PVPositioner`` which handles set/wait via tolerance polling.
    """
    from ophyd import PVPositioner, EpicsSignal, EpicsSignalRO

    class _PVAxis(PVPositioner):
        setpoint = ophyd.Component(EpicsSignal,  setpoint_pv, kind="hinted")
        readback = ophyd.Component(EpicsSignalRO, readback_pv, kind="hinted")

        def __init__(self, *args, **kwargs):
            super().__init__(*args,
                             settle_time=settle_time,
                             timeout=timeout,
                             **kwargs)

    import ophyd
    return _PVAxis("", name=name)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _append_csv(filename: str, record: dict, fieldnames: list[str]) -> None:
    """Append one row to *filename*, creating the file and header if needed."""
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore", restval="")
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def _peak_info(ps: PeakStats) -> tuple[Optional[float], Optional[float]]:
    """Return (centroid, sigma) from a PeakStats object; None if unavailable."""
    cen   = ps.cen
    sigma = (ps.fwhm / 2.355) if ps.fwhm else None
    return cen, sigma


# ---------------------------------------------------------------------------
# Core helper: stepped scan → find peak → move motor
# ---------------------------------------------------------------------------

def _scan_and_move(
    detectors, motor,
    rel_start: float, rel_stop: float, nsteps: int,
    *,
    move_to: Optional[str] = "centroid",  # "centroid" | "peak_pos" | None
    settle: float = 0.3,
    fine_scan: bool = True,
    fine_sigma_range: float = 3.0,
    fine_nsteps: int = 21,
):
    """
    Stepped relative scan with optional coarse + fine phases.

    Phase 1 — coarse rel_scan over [rel_start, rel_stop].
    Phase 2 — (if fine_scan) fine rel_scan centred on the coarse peak,
               width = ±fine_sigma_range × sigma.

    The motor is moved to the peak position (absolute) after each phase.
    ``move_to`` chooses between "centroid" (weighted mean) and "peak_pos"
    (position of the highest point); pass None to skip the final move.
    """
    motor_name = motor.name
    det_name   = detectors[0].name

    # ── Coarse scan ────────────────────────────────────────────────────────
    ps = PeakStats(motor_name, det_name)
    yield from subs_wrapper(
        bp.rel_scan(detectors, motor, rel_start, rel_stop, nsteps),
        ps,
    )

    cen, sigma = _peak_info(ps)
    if cen is None:
        print(f"    ⚠ No peak found in coarse scan of {motor_name}")
        return

    target = cen if move_to == "centroid" else (ps.max[0] if ps.max else cen)
    if move_to is not None:
        yield from bps.mv(motor, target)
        yield from bps.sleep(settle)

    if not fine_scan or sigma is None:
        return

    # ── Fine scan centred on coarse peak ──────────────────────────────────
    half = fine_sigma_range * sigma
    ps_f = PeakStats(motor_name, det_name)
    yield from subs_wrapper(
        bp.rel_scan(detectors, motor, -half, +half, fine_nsteps),
        ps_f,
    )

    cen_f, _ = _peak_info(ps_f)
    if cen_f is None:
        print(f"    ⚠ No peak found in fine scan of {motor_name}")
        return

    if move_to is not None:
        target_f = (cen_f if move_to == "centroid"
                    else (ps_f.max[0] if ps_f.max else cen_f))
        yield from bps.mv(motor, target_f)
        yield from bps.sleep(settle)


# ---------------------------------------------------------------------------
# Main alignment plan
# ---------------------------------------------------------------------------

def sbm_align(
    table,
    *,
    # ── Devices (required) ─────────────────────────────────────────────────
    brg2,
    roll2,
    x2,
    pitch,
    slit_v,
    slit_h,
    detector,
    # ── Optional energy-setting devices ────────────────────────────────────
    mono_e        = None,   # mono energy setpoint
    harmonic_sp   = None,   # undulator harmonic setpoint
    und_e         = None,   # undulator energy setpoint
    und_start     = None,   # undulator start trigger
    roll2_e_sp    = None,   # Roll2 nominal energy setpoint
    x2_e_sp       = None,   # X2 nominal energy setpoint
    monitor_det   = None,   # monitor detector for normalisation (recorded only)
    # ── Slit positions ─────────────────────────────────────────────────────
    slit_open_v   : float = 5.0,
    slit_open_h   : float = 5.0,
    slit_close_v  : float = 0.5,
    slit_close_h  : float = 0.5,
    # ── Pitch ──────────────────────────────────────────────────────────────
    pitch_home    : float = 5.0,
    pitch_settle  : float = 0.1,
    do_pitch_scan : bool  = True,
    pitch_start   : float = -1.0,
    pitch_stop    : float =  1.0,
    pitch_nsteps  : int   = 21,
    # ── BRG2 ───────────────────────────────────────────────────────────────
    brg2_start    : float = -0.005,
    brg2_stop     : float =  0.005,
    brg2_nsteps   : int   = 21,
    # ── Roll2 ──────────────────────────────────────────────────────────────
    roll2_start   : float = -0.005,
    roll2_stop    : float =  0.005,
    roll2_nsteps  : int   = 21,
    # ── X2 ─────────────────────────────────────────────────────────────────
    x2_start      : float = -0.5,
    x2_stop       : float =  0.5,
    x2_nsteps     : int   = 21,
    # ── Scan quality ───────────────────────────────────────────────────────
    settle        : float = 0.3,
    energy_settle : float = 2.0,
    fine_scan     : bool  = True,
    fine_sigma_range: float = 3.0,
    fine_nsteps   : int   = 21,
    # ── Output ─────────────────────────────────────────────────────────────
    filename      : str   = "alignment_results.csv",
    record_settle : float = 2.0,
    md            : Optional[dict] = None,
):
    """
    Full SBM alignment Bluesky plan.

    Replicates ``align_sbm.align_beamline()``.  For each row in *table*
    (list of dicts or DataFrame rows with keys MonoE, Harmonic, UndE,
    Roll2, X2) the plan executes:

      1.  Set mono energy, undulator, Roll2/X2 nominal set-points
      2.  Open slits (V + H) → sleep 5 s
      3.  Home pitch piezo → sleep
      4.  BRG2 coarse + fine scan → move to peak_pos
      5.  [optional] Pitch scan → move to centroid
      6.  Close vertical slit → sleep 5 s
      7.  Roll2 coarse + fine scan → move to centroid
      8.  [optional] Pitch scan (repeat) → move to centroid
      9.  Close horizontal slit → sleep 5 s
     10.  X2 coarse + fine scan → move to centroid
     11.  Read Roll2 + X2 RBVs (+ monitor if supplied) → append CSV row

    Returns a list of result dicts, one per energy row.
    """
    if md is None:
        md = {}

    detectors = [detector]

    fieldnames = ["datetime", "MonoE", "Harmonic", "UndE", "Roll2", "X2"]
    if monitor_det is not None:
        fieldnames.append("Monitor")

    all_records = []

    for row_idx, row in enumerate(table):
        mono_e_val = float(row["MonoE"])
        harmonic   = int(row["Harmonic"])
        und_e_val  = float(row["UndE"])
        roll2_sp   = float(row["Roll2"])
        x2_sp      = float(row["X2"])

        sep = "═" * 60
        print(f"\n{sep}")
        print(f"  Row {row_idx + 1}/{len(table)}: "
              f"MonoE={mono_e_val} keV  Harmonic={harmonic}  UndE={und_e_val}")
        print(sep)

        # ── Step 1: Set energy ────────────────────────────────────────────
        print("\n  1) Setting energy …")
        moves = []
        if mono_e      is not None: moves += [mono_e,      mono_e_val]
        if harmonic_sp is not None: moves += [harmonic_sp, harmonic]
        if und_e       is not None: moves += [und_e,       und_e_val]
        if roll2_e_sp  is not None: moves += [roll2_e_sp,  roll2_sp]
        if x2_e_sp     is not None: moves += [x2_e_sp,     x2_sp]
        if moves:
            yield from bps.mv(*moves)
        if und_start is not None:
            # Trigger undulator to start moving to the new energy
            yield from bps.mv(und_start, 1)
        yield from bps.sleep(energy_settle)

        # ── Step 2: Open slits ────────────────────────────────────────────
        print(f"\n  2) Opening slits: V={slit_open_v}  H={slit_open_h}")
        yield from bps.mv(slit_v, slit_open_v, slit_h, slit_open_h)
        yield from bps.sleep(5.0)

        # ── Step 3: Home pitch ────────────────────────────────────────────
        print(f"\n  3) Homing pitch → {pitch_home}")
        yield from bps.mv(pitch, pitch_home)
        yield from bps.sleep(pitch_settle * 3)

        # ── Step 4: BRG2 scan → move to peak_pos ─────────────────────────
        print(f"\n  4) BRG2 scan  [{brg2_start:+g} … {brg2_stop:+g}  "
              f"{brg2_nsteps} steps]")
        yield from _scan_and_move(
            detectors, brg2, brg2_start, brg2_stop, brg2_nsteps,
            move_to="peak_pos", settle=settle,
            fine_scan=fine_scan,
            fine_sigma_range=fine_sigma_range,
            fine_nsteps=fine_nsteps,
        )

        # ── Step 5: Pitch scan (d) → move to centroid ─────────────────────
        if do_pitch_scan:
            print(f"\n  5) Pitch scan  [{pitch_start:+g} … {pitch_stop:+g}  "
                  f"{pitch_nsteps} steps]")
            yield from _scan_and_move(
                detectors, pitch, pitch_start, pitch_stop, pitch_nsteps,
                move_to="centroid", settle=pitch_settle,
                fine_scan=False,  # pitch uses single-pass scan
            )

        # ── Step 6: Close vertical slit ───────────────────────────────────
        print(f"\n  6) Closing V slit → {slit_close_v}")
        yield from bps.mv(slit_v, slit_close_v)
        yield from bps.sleep(5.0)

        # ── Step 7: Roll2 scan → move to centroid ─────────────────────────
        print(f"\n  7) Roll2 scan  [{roll2_start:+g} … {roll2_stop:+g}  "
              f"{roll2_nsteps} steps]")
        yield from _scan_and_move(
            detectors, roll2, roll2_start, roll2_stop, roll2_nsteps,
            move_to="centroid", settle=settle,
            fine_scan=fine_scan,
            fine_sigma_range=fine_sigma_range,
            fine_nsteps=fine_nsteps,
        )

        # ── Step 8: Pitch scan (f2) → move to centroid ────────────────────
        if do_pitch_scan:
            print(f"\n  8) Pitch scan (repeat)  [{pitch_start:+g} … {pitch_stop:+g}  "
                  f"{pitch_nsteps} steps]")
            yield from _scan_and_move(
                detectors, pitch, pitch_start, pitch_stop, pitch_nsteps,
                move_to="centroid", settle=pitch_settle,
                fine_scan=False,
            )

        # ── Step 9: Close horizontal slit ─────────────────────────────────
        print(f"\n  9) Closing H slit → {slit_close_h}")
        yield from bps.mv(slit_h, slit_close_h)
        yield from bps.sleep(5.0)

        # ── Step 10: X2 scan → move to centroid ───────────────────────────
        print(f"\n  10) X2 scan  [{x2_start:+g} … {x2_stop:+g}  "
              f"{x2_nsteps} steps]")
        yield from _scan_and_move(
            detectors, x2, x2_start, x2_stop, x2_nsteps,
            move_to="centroid", settle=settle,
            fine_scan=fine_scan,
            fine_sigma_range=fine_sigma_range,
            fine_nsteps=fine_nsteps,
        )

        # ── Step 11: Record RBVs and write CSV ────────────────────────────
        yield from bps.sleep(record_settle)

        roll2_rbv = roll2.position
        x2_rbv    = x2.position

        record = {
            "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "MonoE"   : mono_e_val,
            "Harmonic": harmonic,
            "UndE"    : und_e_val,
            "Roll2"   : roll2_rbv,
            "X2"      : x2_rbv,
        }
        if monitor_det is not None:
            reading = yield from bps.rd(monitor_det)
            record["Monitor"] = reading

        _append_csv(filename, record, fieldnames)
        all_records.append(record)

        print(f"\n  ✓ Row done: Roll2={roll2_rbv:.6g}  X2={x2_rbv:.6g}")

    print(f"\n{'═'*60}")
    print(f"  Alignment complete — {len(all_records)} row(s) written to {filename}")
    print('═'*60)
    return all_records


# ---------------------------------------------------------------------------
# Convenience: iterate the plan over multiple loop passes
# ---------------------------------------------------------------------------

def sbm_align_loop(table, n_loops: int = 0, **kwargs):
    """
    Run sbm_align repeatedly.

    Parameters
    ----------
    n_loops : int
        Number of complete passes.  0 = run once (no looping).
        N > 0 = run exactly N times.
    **kwargs : passed verbatim to sbm_align.
    """
    passes = n_loops if n_loops > 0 else 1
    all_records = []
    for i in range(passes):
        if passes > 1:
            print(f"\n{'━'*60}")
            print(f"  Loop {i + 1}/{passes}")
        records = yield from sbm_align(table, **kwargs)
        all_records.extend(records)
    return all_records
