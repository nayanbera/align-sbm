# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commit rule
**Never** add `Co-Authored-By:` trailers to commit messages.

## Running

```bash
pip install -e .
python -m align_sbm.gui         # launch the PyQt6 GUI
python -m align_sbm.smart_scan_functions  # run built-in self-tests
```

## Architecture

```
align_sbm/
  smart_scan_functions.py   — all scan/alignment backend logic (no Qt)
  gui.py                    — PyQt6 main window (to be created)
  gui/                      — per-tab widgets (to be created)
```

### Backend (`smart_scan_functions.py`)

- **`smart_scan(motor, det, start, stop, …)`** — stepped scan, coarse+fine phases, automatic peak finding, extends range if peak is at edge.
- **`fly_scan(motor, det, start, stop, …)`** — continuous-motion scan with a background sampling thread; supports `PVAxis` (no DMOV).
- **`align_beamline(table, config)`** — full multi-step SBM alignment protocol:
  1. Open slits → 2. Home pitch → 3. BRG2 smart_scan → 4. Pitch fly_scan
  5. Close V slit → 6. Roll2 smart_scan → 6b. Pitch fly_scan → 7. Close H slit
  8. X2 smart_scan → 9. Record RBVs to CSV
- **`BeamlineConfig`** dataclass — all motors, PVs, scan parameters; validated by `.check()`.
- **`PVAxis`** — setpoint+readback PV wrapper (no velocity control, no DMOV).
- **`ScanResult`** — status, positions, signals, center, sigma, amplitude, profile, stats.
- Simulation mode activates automatically when `pyepics` is not installed; all scans use Gaussian + noise.

### Energy table (`table400`)
Columns: `[MonoE, Harmonic, UndE, Roll2, X2]`  
Default PV prefix: `ID15A2:`

## Key design constraints

- All EPICS I/O is in `smart_scan_functions.py` only — Qt widgets must never import `epics` directly.
- Simulation mode must work without any beamline connection (CI-safe).
- GUI must never block the Qt event loop; scans run in a `QThread`.
