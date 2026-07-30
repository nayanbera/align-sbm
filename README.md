# align-sbm

PyQt6 GUI for the Side-Bounce Monochromator (SBM) alignment protocol at ID15A2.

## Overview

`align-sbm` automates the multi-step SBM alignment sequence across a user-defined energy table. It wraps a fully-featured scan backend (`smart_scan`, `fly_scan`) that supports both live EPICS hardware and a built-in simulation mode — no beamline connection required for testing.

## Requirements

- Python ≥ 3.10
- PyQt6 ≥ 6.4
- pyqtgraph ≥ 0.13
- numpy ≥ 1.24
- scipy ≥ 1.10
- pyepics ≥ 3.5 *(optional — simulation mode works without it)*

## Installation

```bash
git clone <repo-url> align-sbm
cd align-sbm
pip install -e .
```

## Launch

```bash
python -m align_sbm.gui
```

Or after installation:

```bash
align-sbm
```

---

## GUI Layout

The GUI has three tabs.

### Setup

Two inner tabs:

**Motors & PVs**

| Field | Description |
|---|---|
| PV Prefix | Prefix applied to all PVs when you click *Auto-fill all PVs* |
| BRG2 / Roll2 / X2 motor | EPICS motor record base PV (e.g. `ID15A2:BRG2`) |
| Detector | Scalar readback PV |
| Pitch piezo SP | Setpoint PV for the pitch piezo (`PVAxis`) |
| Slit V / H SP | Vertical and horizontal slit setpoint PVs |
| Mono energy | Monochromator energy setpoint PV |
| Undulator harmonic / energy / start | Undulator control PVs |
| Roll2 / X2 energy set | Nominal encoder setpoint PVs written before each energy row |

All PV names are saved and restored automatically between sessions.

**Scan Parameters**

Editable scan ranges, step counts, slit open/close values, fine-scan settings, settle times, peak-finding method, and output CSV filename. Defaults:

| Motor | Start | Stop | Steps |
|---|---|---|---|
| BRG2 | −0.005 | +0.005 | 21 |
| Pitch | −1.0 | +1.0 | 21 |
| Roll2 | −0.005 | +0.005 | 21 |
| X2 | −0.5 | +0.5 | 21 |

---

### Energy Table

A five-column table with rows of the form:

| MonoE (keV) | Harmonic | UndE (eV) | Roll2 (mdeg) | X2 (μm) |
|---|---|---|---|---|
| 10.0 | 1 | 10.03 | 3.7 × 10⁶ | −1393 |
| 12.0 | 1 | 12.05 | 3.3 × 10⁶ | −843 |
| 16.0 | 3 | 16.06 | 3.0 × 10⁶ | −362 |
| 20.0 | 3 | 20.10 | 3.7 × 10⁶ | −586 |
| 25.0 | 3 | 25.10 | 3.3 × 10⁶ | −463 |
| 30.0 | 3 | 30.13 | 3.4 × 10⁶ | −582 |

Buttons: **Add Row**, **Remove Row**, **Load CSV**, **Save CSV**, **Reset to Defaults**.

The table is saved and restored between sessions. To load an external CSV the file must have header columns: `MonoE`, `Harmonic`, `UndE`, `Roll2`, `X2`.

---

### Alignment

**Left panel — controls**

- **Simulation** checkbox — when checked, scans and motor moves are fully simulated (no EPICS). Enabled by default.
- **Energy rows to align** — multi-select list populated from the Energy Table tab. Use *All* / *None* / *Refresh* to manage the selection.
- **Start Alignment** — builds the EPICS/simulation context from the Setup tab and runs `align_beamline()` in a background thread for the selected rows.
- **Abort** — terminates the background thread immediately.
- **Demo Scan (sim)** — runs a single simulated `smart_scan` using the BRG2 range from Setup and immediately shows the data and fit curve in the plot. Useful for verifying the GUI without a beamline.

**Right panel — output**

- **Plot** (top) — updates after each scan with the raw signal data (blue dots) and the fitted curve (red line). A dashed vertical line marks the peak/centroid centre.
- **Log** (bottom) — live stdout capture of all backend output, including per-step scan tables, fit results, and `[SIM]` tags in simulation mode.

---

## Alignment Sequence

For each selected energy row `align_beamline()` runs these steps in order:

| Step | Action |
|---|---|
| a | Open slits to configured open positions |
| b | Home pitch piezo to `pitch_home` |
| c | `smart_scan` BRG2 → move to peak position |
| d | `fly_scan` pitch → move to peak |
| e | Close vertical slit |
| f | `smart_scan` Roll2 → move to centroid |
| f2 | `fly_scan` pitch (repeat) |
| g-pre | Close horizontal slit |
| g | `smart_scan` X2 → move to centroid |
| h | Record RBVs and write CSV row |

The CSV output file (default `alignment_results.csv`) grows one row per energy, with columns: `datetime`, `MonoE`, `Harmonic`, `UndE`, `Roll2`, `X2`.

### Peak-finding logic

After each scan the motor moves to a position chosen by `_choose_centre()`:

```
FWHM > |peak_pos − centroid|  →  move to peak_pos
otherwise                      →  move to centroid
```

With `peak_method="stats"` (default) all metrics are model-free (centroid, RMS width, FWHM by linear interpolation). With `peak_method="fit"` the code fits a Gaussian, Lorentzian, or super-Gaussian and picks the lowest-residual model.

---

## Backend API

The backend is self-contained in `align_sbm/smart_scan_functions.py` and has no Qt dependency.

```python
from align_sbm import smart_scan, fly_scan, align_beamline, BeamlineConfig, table400

# Simulate a single scan
result = smart_scan(
    motor="IOC:m1", det="IOC:det",
    start=-1.0, stop=1.0, nsteps=21,
    simulate=True, sim_center=0.1, sim_sigma=0.3,
    sim_amplitude=1000, sim_noise=10,
)
print(result.center, result.sigma)

# Run the full alignment protocol (simulation)
results = align_beamline(
    table=table400,
    simulate=True,
    detector="ID15A2:det:Signal",
    brg2="ID15A2:BRG2",
    roll2_motor="ID15A2:Roll2",
    x2_motor="ID15A2:X2",
    pitch_pv="ID15A2:PitchPiezo:SP",
    slit_v_pv="ID15A2:SlitV:SP",
    slit_h_pv="ID15A2:SlitH:SP",
    mono_e_pv="ID15A2:mono:Energy",
    harmonic_pv="ID15A2:und:Harmonic",
    und_e_pv="ID15A2:und:Energy",
    und_start_pv="ID15A2:und:Start",
    roll2_energy_pv="ID15A2:Roll2:EnergySet",
    x2_energy_pv="ID15A2:X2:EnergySet",
    filename="alignment_results.csv",
)
```

### Key types

| Class / function | Purpose |
|---|---|
| `smart_scan(motor, det, start, stop, …)` | Stepped scan with coarse + fine phases; extends range if peak is at edge |
| `fly_scan(motor, det, start, stop, …)` | Continuous-motion scan with threaded sampling |
| `align_beamline(table, …)` | Full multi-step alignment for a list of energy rows |
| `BeamlineConfig` | Dataclass holding all motors, PVs, and scan parameters |
| `PVAxis` | Setpoint + readback PV wrapper (no EPICS motor record required) |
| `ScanResult` | Dataclass: status, positions, signals, center, sigma, amplitude, profile, stats |
| `ScanStatus` | Enum: SUCCESS, NO_PEAK, OUT_OF_RANGE, FIT_FAILED, INSUFFICIENT_DATA |
| `stats_peak(positions, signals)` | Model-free peak estimators: centroid, RMS width, FWHM, weighted median |

### Simulation mode

When `pyepics` is not installed, or when `simulate=True` is passed, all motor moves and detector reads are replaced by Gaussian + noise signal generation. The scan logic (coarse sweep, peak-finding, fine scan, extend-if-edge) runs identically.

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

Linting:

```bash
ruff check align_sbm/
```
