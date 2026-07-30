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
- matplotlib ≥ 3.7 *(optional — required for the Predict from CSV plot)*

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

The GUI has three tabs: **Setup**, **Energy Table**, and **Alignment**.

---

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

Each PV field shows a live readback value next to it (updated via CA monitor). All PV names are saved and restored automatically between sessions.

**Scan Parameters**

Editable scan ranges, step counts, slit open/close values, fine-scan settings, settle times, peak-finding method, and output CSV filename. Defaults:

| Motor | Start | Stop | Steps |
|---|---|---|---|
| BRG2 | −0.005 | +0.005 | 21 |
| Pitch | −1.0 | +1.0 | 21 |
| Roll2 | −0.005 | +0.005 | 21 |
| X2 | −0.5 | +0.5 | 21 |

All numeric fields accept typed values of any magnitude (no spinner clamping).

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

**Buttons:**

- **Add Row / Remove Row** — append or delete rows.
- **Load CSV / Save CSV** — import or export the table. The CSV must have header columns `MonoE`, `Harmonic`, `UndE`, `Roll2`, `X2`.
- **Reset to Defaults** — restore the built-in `table400` values.
- **Predict from CSV…** — open the prediction dialog (see below).

The table is saved and restored between sessions.

#### Predict from CSV

This dialog fits Roll2 and X2 as a function of MonoE using the alignment history stored in the output CSV, then lets you predict values for new intermediate energies and add them directly to the energy table.

**Workflow:**

1. The dialog opens with the last alignment CSV pre-loaded (the same file shown in the Alignment → CSV tab). Use *Browse…* to pick a different file.
2. Choose a regression model:
   - **Polynomial degree 1–4** — uses `numpy.polyfit`; R² is shown for both Roll2 and X2. The maximum available degree is capped at (number of data points − 1).
   - **Cubic spline** — uses `scipy.interpolate.UnivariateSpline` (requires scipy and ≥ 4 data points); interpolates exactly through every measured point.
3. A side-by-side plot shows the measured data (blue circles) and the fitted curve (orange line) for Roll2 and X2.
4. In the prediction table, type a MonoE value in the first column — Roll2 and X2 are filled in automatically. **Harmonic** and **UndE** are highlighted amber to indicate they must be provided manually.
5. Cells with **orange** background indicate the MonoE is outside the training range (extrapolation — use with caution).
6. Click **Add to Energy Table** to append all completed rows.

---

### Alignment

#### Left panel — controls

**Mode**

- **Simulation** checkbox — when checked, all scans and motor moves are fully simulated (Gaussian + noise, no EPICS). Enabled by default.

**Energy rows to align**

Multi-select list populated from the Energy Table tab. Use **All** / **None** / **Refresh** to manage the selection.

**Run**

- **Start Alignment** — builds the EPICS/simulation context from the Setup tab and runs `align_beamline()` in a background thread for every selected energy row.
- **Abort** — stops the running scan immediately (the background thread is terminated). If looping, no further iterations are started.
- **Demo Scan (sim)** — runs a single simulated BRG2 `smart_scan` and animates its data points live in the BRG2 plot tab. Useful for verifying the GUI without a beamline.
- **Loop** checkbox + **Iterations** field — when *Loop* is checked, the full alignment sequence repeats automatically after each pass.
  - `0` (default) — runs indefinitely until **Abort** is pressed.
  - `N > 0` — runs exactly N times then stops.
  - The log marks each iteration with `Loop N/total` headers. The results table accumulates across all loops.

#### Right panel — output

**Plot tabs (BRG2 / Pitch / Roll2 / X2)**

Each motor has its own tab. During a scan, data points appear in real time (blue dots). On scan completion the fitted curve (red line) and peak marker (dashed vertical line) are overlaid. A parameter annotation in the top-right corner shows Profile, Center, FWHM, Sigma, Amplitude, Offset (and the super-Gaussian *p* exponent when applicable).

**Bottom tabs**

| Tab | Contents |
|---|---|
| Results | Summary table — one row per completed energy row: MonoE, BRG2 centre, Roll2 RBV, X2 RBV, pass/fail tick. |
| Log | Live stdout from the backend — per-step scan tables, fit results, warnings, and `[SIM]` tags. |
| CSV | Live view of the output CSV file. Refreshes automatically after each completed energy row. |

**CSV tab controls:**

- **Open CSV…** — load an existing CSV file to append new results to. The column layout must match the current setup (record PVs). Adopts the file as the new output target.
- **Delete Row(s)** — permanently remove selected rows from the CSV file (confirmation required; file is rewritten in place).
- **Refresh** — manually reload the CSV view.
- The path of the last opened or written CSV is remembered and the file is loaded automatically the next time the application starts.

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

---

## Scan Details

### Coarse + fine scan phases

`smart_scan` runs in two phases:

1. **Coarse sweep** — scans from `start` to `stop` in `nsteps` steps. If the peak lands near the scan edge, the range is extended automatically until the peak is fully captured.
2. **Fine scan** — centres a narrower window (default ±3σ, 21 steps) on the coarse peak and refines the position. The motor moves to the start of the fine range *before* the detector flush, so no wasted traversal occurs between phases.

### Peak-finding logic

After each scan the motor moves to a position chosen by `_choose_centre()`:

```
FWHM > |peak_pos − centroid|  →  move to centroid
otherwise                      →  move to peak_pos
```

With `peak_method="stats"` (default) all metrics are model-free (centroid, RMS width, FWHM by linear interpolation). With `peak_method="fit"` the code fits a Gaussian, Lorentzian, or super-Gaussian and picks the lowest-residual model.

### Backlash correction

When `backlash_correction=True`, the motor overshoots the target by one FWHM in the direction opposite to the scan, then approaches from the scan direction — ensuring a consistent final approach.

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
