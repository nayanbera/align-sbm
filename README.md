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
- matplotlib ≥ 3.7

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

## Configuration files

All settings (PV names, scan parameters, energy table, hold conditions) are saved automatically to an OS-native store and restored on the next launch — no manual action needed for normal use.

To move a configuration to another machine or keep named presets, use the **File** menu:

| Action | Description |
|---|---|
| **Save Config** (`Ctrl+S`) | Flush the current UI state to the local settings store immediately |
| **Save Config As…** | Export *all* settings to a portable `.ini` file you can copy to any machine |
| **Load Config…** | Import a `.ini` file — overwrites the local settings store and refreshes every tab live (no restart needed) |

The `.ini` file is plain text and can be version-controlled or diff'd to compare beamline configurations.

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
| Roll1 motor | EPICS motor record base PV for Roll1; when set, Roll1 is moved to the table value at every energy change and alignment step |
| Detector | Scalar readback PV |
| Monitor (normalize) | Optional monitor detector PV — when set, all scan signals are divided by this value before fitting, statistics, and plotting. Leave blank to disable. |
| Pitch piezo SP | Setpoint PV for the pitch piezo (`PVAxis`) |
| Slit V / H SP | Vertical and horizontal slit setpoint PVs |
| Mono energy | Monochromator energy setpoint PV |
| Undulator harmonic / energy / start | Undulator control PVs |
| Roll2 / X2 energy set | Nominal encoder setpoint PVs written before each energy row |

Each PV field shows a live readback value next to it (updated via CA monitor). All PV names are saved and restored automatically between sessions.

**Scan Parameters**

Editable scan ranges, step counts, slit open/close values, fine-scan settings, settle times, peak-finding method, and output CSV filename. All numeric fields and dropdowns use compact, fixed-width widgets that do not expand horizontally when the window is resized. Mouse-wheel scrolling is disabled on all spinboxes and dropdowns to prevent accidental value changes.

Each scan group (BRG2, Pitch, Roll2, X2) includes a **Normalize with monitor PV** checkbox. When checked, the detector signal for that scan is divided by the Monitor PV value at each point. The Monitor PV is set in the Motors & PVs tab.

Defaults:

| Motor | Start | Stop | Steps |
|---|---|---|---|
| BRG2 | −0.005 | +0.005 | 21 |
| Pitch | −1.0 | +1.0 | 21 |
| Roll2 | −0.005 | +0.005 | 21 |
| X2 | −0.5 | +0.5 | 21 |

All numeric fields accept typed values of any magnitude (no spinner clamping).

**Fine Scan group** additionally exposes:

| Field | Default | Description |
|---|---|---|
| Sigma range | 3.0 | Fine window half-width in σ units |
| N steps | 21 | Steps in the fine sweep |
| Max iterations | 2 | How many successive fine refinements to run |
| DMOV delay (s) | 0.25 | Wait this many seconds after issuing a motor move before polling the DMOV done flag. Increase if the motor record is slow to clear DMOV (typical: 0.2–0.5 s). |

**Output CSV row** includes an **Autosave** checkbox. When checked (default), results are appended to the CSV after every completed alignment. Uncheck to run without writing to disk. In Simulation mode the CSV is never written regardless of this setting.

---

### Energy Table

An eight-column table:

| MonoE (keV) | Harmonic | UndE (eV) | Roll2 (mdeg) | X2 (μm) | Roll1 (mdeg) | Crystal | Updated |
|---|---|---|---|---|---|---|---|
| 10.0 | 1 | 10.03 | 3.7 × 10⁶ | −1393 | 0.0 | Si 111 | 2026-09-23 10:04:11 |
| 12.0 | 1 | 12.05 | 3.3 × 10⁶ | −843 | 0.0 | Si 400 | |

**Columns:**

| Column | Description |
|---|---|
| MonoE (keV) | Monochromator target energy |
| Harmonic | Undulator harmonic |
| UndE (eV) | Undulator energy |
| Roll2 (mdeg) | Roll2 motor target; updated automatically after each alignment |
| X2 (μm) | X2 motor target; updated automatically after each alignment |
| Roll1 (mdeg) | Roll1 motor target; auto-filled from current Roll1 RBV when a row is added |
| Crystal | Crystal name for this energy (free text, e.g. `Si 400`); used to enforce crystal matching before alignment or Move to Energy |
| Updated | Timestamp of the last successful alignment at this energy; read-only |

**Sorting:** Click any column header to sort by that column. Numeric columns (MonoE through Roll1) sort by value; Crystal and Updated sort alphabetically. An arrow indicator shows the active sort column and direction. Changing the sort order is immediately reflected in the Energy rows list on the Alignment tab.

**Row colors:** Each row's background is automatically colored using the crystal's configured color (set via the Crystal Status **⚙** button on the Alignment tab). Colors update live when a Crystal cell is edited or when crystal color mappings are changed. Rows with no crystal set have no background color.

**Buttons:**

- **Add Row / Remove Row** — append or delete rows. Roll1 and Crystal are auto-filled from the current Setup readback and crystal status when a row is added.
- **Load CSV / Save CSV** — import or export the table. The CSV must have header columns `MonoE`, `Harmonic`, `UndE`, `Roll2`, `X2`.
- **Reset to Defaults** — restore the built-in `table400` values.
- **Predict from CSV…** — open the prediction dialog (see below).

The table is saved and restored between sessions (including Roll1, Crystal, and Updated fields).

#### Predict from CSV

This dialog fits Roll2 and X2 as a function of MonoE using the alignment history stored in the output CSV, then lets you predict values for new intermediate energies and add them directly to the energy table.

**Workflow:**

1. The dialog opens with the last alignment CSV pre-loaded. Use *Browse…* to pick a different file.
2. Choose a regression model:
   - **Polynomial degree 1–4** — uses `numpy.polyfit`; R² is shown for both Roll2 and X2.
   - **Cubic spline** — uses `scipy.interpolate.UnivariateSpline` (requires ≥ 4 data points); interpolates exactly through every measured point.
3. A side-by-side plot shows the measured data (blue circles) and the fitted curve (orange line) for Roll2 and X2.
4. In the prediction table, type a MonoE value — Roll2 and X2 are filled in automatically. **Harmonic** and **UndE** are highlighted amber to indicate they must be provided manually.
5. Cells with **orange** background indicate extrapolation beyond the training range — use with caution.
6. Click **Add to Energy Table** to append all completed rows.

---

### Alignment

#### Left panel — controls

**Mode**

- **Simulation** checkbox — when checked, all scans and motor moves are fully simulated (Gaussian + noise, no EPICS). The energy table is never modified in simulation mode.

**Crystal Status**

A color-coded chip shows the currently active crystal read from the configured EPICS PV. Click **⚙** to set the PV name and define raw-value → display-name + color mappings. The chip updates live via a CA monitor.

Each crystal mapping has a configurable color. The color picker includes an **opacity/transparency slider** (alpha channel), so colors can be fully opaque or semi-transparent. The same color is used in the energy table row backgrounds, the energy row list items, and the crystal chip.

**Energy rows to align**

Multi-select list populated from the Energy Table. Each item is color-coded to match its crystal's configured color. A `"N of M selected"` count is shown above the list.

- Rows whose Crystal field is set to a **different** crystal than the current one are **grayed out and non-selectable** — they cannot be included in an alignment run until the crystal is switched.
- Rows with no Crystal field set are always selectable.
- The list updates automatically when: the energy table changes, the sort order changes, the crystal PV value changes, or the crystal mappings are edited.
- Use **All** / **None** / **Refresh** buttons to manage the selection. **All** only selects rows that are currently enabled (not grayed out).

**Crystal mismatch enforcement:**

- **Start Alignment** — blocked with an error dialog if any selected row's Crystal field doesn't match the current crystal.
- **Move to Energy** — blocked with an error dialog if the selected row's Crystal field doesn't match the current crystal.

**Run**

- **Start Alignment** — builds the EPICS/simulation context from the Setup tab and runs `align_beamline()` in a background thread for every selected energy row.
- **Abort** — stops the running scan immediately.
- **Demo Scan (sim)** — runs a single simulated BRG2 `smart_scan` and animates its data points live.
- **Loop** checkbox + **Iterations** field — when *Loop* is checked, the full alignment sequence repeats automatically. Unchecking mid-run stops looping after the current pass completes. `0` = run indefinitely; `N > 0` = run exactly N times.
- The **per-energy repeat** spinbox is disabled while alignment is running and re-enabled when finished or aborted.
- The **currently running row** is highlighted in amber with bold text. Highlighting is cleared when the run finishes or is aborted.

#### Right panel — output

**Plot tabs (BRG2 / Pitch / Roll2 / X2)**

Each motor has its own tab. During a scan, data points appear in real time:

| Colour | Meaning |
|---|---|
| Blue (`#4fc3f7`) | Coarse sweep points |
| Green (`#66bb6a`) | Fine scan points |

On completion: fitted curve (red line), peak marker (dashed orange vertical line), and a parameter annotation (Profile, Center, FWHM, Sigma, Amplitude, Offset).

**Bottom tabs**

| Tab | Contents |
|---|---|
| Results | Summary table — one row per completed energy row: MonoE, BRG2 centre, Roll2 RBV, X2 RBV, pass/fail. Sortable by any column. |
| Log | Live stdout from the backend. |
| CSV | Live view of the output CSV file. Sortable by any column. Refreshes automatically after each completed energy row. |

**CSV tab controls:**

- **Open CSV…** — load an existing CSV file to append new results to.
- **Delete Row(s)** — permanently remove selected rows (confirmation required).
- **Refresh** — manually reload the CSV view.
- **Analyze…** — open the statistical analysis dialog.
- **Add Column…** — add a new column to the CSV file backed by an EPICS PV.
- The path of the last opened or written CSV is remembered and auto-loaded on next launch.

**CSV color coding** — a **Color by:** selector and **Edit Rules…** button appear in the CSV tab header. Rules are evaluated against the chosen column and the first matching rule wins. Each rule has an Operator, Value(s), an optional Label, and a Color (with alpha/opacity support). Legend chips above the table show the active rules.

#### Statistical Analysis Dialog

Opened via **Analyze…** in the CSV tab.

A **Date range** filter re-runs all plots, the report, and model training — only rows within the selected range are included.

A **Crystal** filter combobox sits below the date range. It is populated automatically from the energy table's MonoE → crystal mapping (matched within 0.001 keV). Selecting a crystal limits all statistics, plots, drift analysis, correlation matrices, and model training to rows belonging to that crystal's energies. **All crystals** (default) shows everything. The combobox is disabled when no crystal information is available in the energy table.

**Plots tab** — 3 × 2 matplotlib grid: Roll2/X2 time series, mean±std bar charts, Pearson correlation heatmap, Roll2 vs X2 scatter.

**Report tab** — six sections: Descriptive Statistics, Per-Energy Group Statistics (CV% color-coded), Pearson Correlation, Spearman Correlation, Drift Analysis, Strongest Pairwise Correlations.

**Predict tab** — fits Roll2 and X2 vs MonoE; LOO/k-fold CV; uncertainty bands; "Send selected to Energy Table".

---

## Alignment Sequence

For each selected energy row `align_beamline()` runs these steps:

| Step | Action |
|---|---|
| a | Open slits to configured open positions |
| b | Home pitch piezo |
| c | `smart_scan` BRG2 → move to peak |
| d | `fly_scan` pitch → move to peak |
| e | Close vertical slit |
| f | `smart_scan` Roll2 → move to centroid |
| f2 | `fly_scan` pitch (repeat) |
| g-pre | Close horizontal slit |
| g | `smart_scan` X2 → move to centroid |
| h | Record RBVs, write CSV row, update Energy Table (Roll2, X2, timestamp) |

Roll1 is moved to the table value at the energy-change step (before scanning) when a Roll1 motor is configured.

---

## Scan Details

### Coarse + fine scan phases

`smart_scan` runs in two phases:

1. **Coarse sweep** (blue dots) — scans from `start` to `stop` in `nsteps` steps. If the peak lands near the scan edge, the range is extended automatically.
2. **Fine scan** (green dots) — centres a narrower window (default ±3σ, 21 steps) on the coarse peak and refines the position.

### Peak-finding logic

```
FWHM > |peak_pos − centroid|  →  move to centroid
otherwise                      →  move to peak_pos
```

With `peak_method="stats"` (default) all metrics are model-free. With `peak_method="fit"` the code fits a Gaussian, Lorentzian, or super-Gaussian and picks the lowest-residual model.

### Backlash correction

When `backlash_correction=True`, the motor overshoots the target by one FWHM then approaches from the scan direction.

### DMOV delay

The `dmov_delay` parameter (default **0.25 s**) inserts a fixed sleep between the move command and the start of DMOV polling. Configurable from the Fine Scan group in Setup.

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
    monitor_pv="IOC:monitor",   # optional normalization
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
    roll1_motor="ID15A2:Roll1",   # optional
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
    monitor_pv="ID15A2:monitor",
    monitor_brg2=True, monitor_pitch=True,
    monitor_roll2=True, monitor_x2=True,
)
```

### Key types

| Class / function | Purpose |
|---|---|
| `smart_scan(motor, det, start, stop, …)` | Stepped scan with coarse + fine phases |
| `fly_scan(motor, det, start, stop, …)` | Continuous-motion scan with threaded sampling |
| `align_beamline(table, …)` | Full multi-step alignment for a list of energy rows |
| `BeamlineConfig` | Dataclass holding all motors, PVs, and scan parameters |
| `PVAxis` | Setpoint + readback PV wrapper (no EPICS motor record required) |
| `ScanResult` | Dataclass: status, positions, signals, center, sigma, amplitude, profile, stats |
| `ScanStatus` | Enum: SUCCESS, NO_PEAK, OUT_OF_RANGE, FIT_FAILED, INSUFFICIENT_DATA |
| `stats_peak(positions, signals)` | Model-free peak estimators: centroid, RMS width, FWHM, weighted median |

### Simulation mode

When `pyepics` is not installed, or when `simulate=True` is passed, all motor moves and detector reads are replaced by Gaussian + noise signal generation. The scan logic runs identically.

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
