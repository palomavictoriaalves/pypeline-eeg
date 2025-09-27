# EEG Pipeline (BIDS) — PRE/POST & EO/EC

A complete pipeline to **preprocess**, **quantify**, and **visualize** EEG power in studies with **PRE/POST** sessions and **eyes-open (EO) / eyes-closed (EC)** conditions.  
Inputs follow **BIDS** (BrainVision); outputs are **publication-ready** CSVs and figures.

---

## Repository layout

```
.
├─ code/
│  ├─ preprocess.py                         # BIDS-aware preprocessing for BrainVision
│  ├─ calc_power.py                         # builds wide power tables (ABS/REL)
│  ├─ calc_timeseries_power.py              # (optional) time-series data assembly
│  ├─ plot_timeseries_all_EC_EO.py          # EO/EC time-series per band × ROI (2×3 panels)
│  ├─ plot_topomaps_grid_EO_EC.py           # topographic grids (EO vs EC, etc.)
│  ├─ plot_heat_all_bands_timeseries.py     # heatmaps of temporal power by band
│  ├─ plot_mirror_EO_EC_points.py           # mirror/bar/point plots (EO vs EC)
│  ├─ calc_stats_power.py                   # minimal paired stats from wide power tables
│  └─ config.py                             # study parameters and paths
├─ data/                                    # BIDS input (read-only)
│  └─ sub-XX[/ses-YY]/eeg/*.vhdr + .eeg + .vmrk (+ .json)
├─ results/                                 # derivatives (generated)
│  ├─ processed/                            # preprocessed FIFs (concat, EO, EC + manifest)
│  ├─ power/                                # wide power tables (abs/rel)
│  ├─ timeseries/                           # per-ROI×band temporal data (optional)
│  └─ plots/                                # figures and CSVs
├─ requirements.txt
└─ LICENSE.txt
```

> **Note**: `config.py` lives in `code/`. Scripts import it as `import config` when run from the project root (e.g., `python code/preprocess.py`).

---

## Requirements

- **Python 3.9+** (3.10–3.12 recommended)
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
  Or manually:
  ```bash
  pip install mne mne-icalabel numpy pandas matplotlib scipy
  ```

---

## Data (BIDS)

**Expected input under `data/`:**
```
data/
├─ sub-01/
│  ├─ ses-pre/
│  │  └─ eeg/
│  │     ├─ sub-01_ses-pre_task-rest_eeg.vhdr
│  │     ├─ sub-01_ses-pre_task-rest_eeg.eeg
│  │     └─ sub-01_ses-pre_task-rest_eeg.vmrk
│  └─ ses-post/
│     └─ eeg/
│        └─ sub-01_ses-post_task-rest_eeg.vhdr ...
└─ sub-02/ ...
```

> Each `.vhdr` **must** correctly reference its `.eeg`/`.vmrk`. If links are broken, fix them (e.g., with a helper like `fix_brainvision_links.py`) before preprocessing.

---

## Configuration (`code/config.py`)

`config.py` is inside `code/`. Because of that, `PROJECT_ROOT` should be the **parent of `code/`**:

```python
from pathlib import Path

SCRIPT_PATH   = Path(__file__).resolve()
PROJECT_ROOT  = SCRIPT_PATH.parent.parent      # <- repo root
DATA_DIR      = PROJECT_ROOT / "data"
RESULTS_DIR   = PROJECT_ROOT / "results"
PLOTS_DIR     = RESULTS_DIR / "plots"
POWER_DIR     = RESULTS_DIR / "power"
TS_DIR        = RESULTS_DIR / "timeseries"
PROCESSED_DIR = RESULTS_DIR / "processed"
```

Other key settings you will likely edit:

- **Filtering / PSD**
  ```python
  FILTER_LOW = 0.5
  FILTER_HIGH = 50.0
  NOTCH_HZ = 60        # (50 in EU)

  WELCH_SEG_SEC = 4.0
  WELCH_OVERLAP = 0.5
  PSD_FMIN      = 0.5
  PSD_FMAX      = 50.0
  ```

- **Blocks (EO/EC)** used for segmentation during preprocessing (seconds):
  ```python
  BLOCKS_WITH_STATE = [
      ("EO", (15, 135)),
      ("EC", (150, 270)),
      ("EO", (285, 405)),
      ("EC", (420, 540)),
  ]
  ```

- **Bands**
  ```python
  BANDS = {
      "Delta": (0.1, 3.5),
      "Theta": (4.0, 7.9),
      "Alpha": (8.0, 12.9),
      "Beta":  (13.0, 30.0),
      "Gamma": (30.1, 50.0),
  }
  ```

- **Groups & ordering**
  ```python
  GROUP_ACTIVE  = {'01','05','07','10','15','16','19'}
  GROUP_PASSIVE = {'03','04','06','08','11','13','21'}
  GROUP_CONTROL = {'02','09','12','14','17','18','22'}

  GROUPS_ORDER = ["Active","Passive","Control"]
  VS_ORDER     = ["EO","EC"]       # visual states
  ```

- **ROIs** — used by the time-series plots (expose as `ROI_CHANNELS`):
  ```python
  REGIONS = {
      "Prefrontal":      ["Fp1","Fp2"],
      "Frontal":         ["F7","F3","Fz","F4","F8"],
      "Frontocentral":   ["FC5","FC1","FC2","FC6"],
      "Central":         ["C3","Cz","C4"],
      "Temporo-parietal":["FT9","T7","T8","FT10","TP9","TP10"],
      "Centro-parietal": ["CP5","CP1","CP2","CP6"],
      "Parietal":        ["P7","P3","Pz","P4","P8"],
      "Occipital":       ["O1","Oz","O2"],
  }
  ROI_CHANNELS = REGIONS
  ```

- **Time-series settings**
  ```python
  TS_WIN_SEC        = 4.0
  TS_STEP_SEC       = 1.0
  TS_FDR_ALPHA      = 0.05
  TS_MARK_SIG       = True
  TS_GENERATE_PLOTS = True
  ```

---

## Quick start (from project root)

1) **Preprocess** all `.vhdr` (BIDS-aware):
```bash
python code/preprocess.py
```

2) **Wide power tables** (ABS/REL):
```bash
python code/calc_power.py
```
Expected outputs:
```
results/power/
  power_wide_rel_EO_EC.csv
  power_wide_abs_EO_EC.csv
```

3) **EO/EC time series** (band × ROI; 2×3 PRE/POST × group panels):
```bash
python code/plot_timeseries_all_EC_EO.py
```
Outputs:
```
results/plots/timeseries_all/
├─ csv/
│  └─ timeseries_all_bands_rois_relabs.csv
└─ figs/
   └─ timeseries_<Band>_<metric>_<ROI>.png
```

4) **Minimal paired stats** (EC vs EO, POST vs PRE) from wide tables:
```bash
python code/calc_stats_power.py
```
Outputs:
```
results/stats/
├─ rel/
│  ├─ stats_rel_EOvsEC.csv
│  └─ stats_rel_POSTvsPRE.csv
└─ abs/
   ├─ stats_abs_EOvsEC.csv
   └─ stats_abs_POSTvsPRE.csv
```

**Optional plots:**
```bash
python code/plot_topomaps_grid_EO_EC.py
python code/plot_heat_all_bands_timeseries.py
python code/plot_mirror_EO_EC_points.py
```

---

## Script summaries

### `code/preprocess.py`
- Recursively finds `*.vhdr` under `data/**/eeg/`.
- Parses BIDS entities (sub, ses, task, run) from file names.
- Processing: band-pass + notch, average reference, EO/EC segmentation via `BLOCKS_WITH_STATE`, ICA (Infomax extended), **ICLabel**-based IC rejection, EO/EC annotations.
- Saves per subject/session:
  - concatenated: `*_desc-preproc_clean_raw.fif`
  - EO-only:      `*_desc-preproc_EO_clean_raw.fif`
  - EC-only:      `*_desc-preproc_EC_clean_raw.fif`
  - block manifest: `*_desc-preproc_blocks_manifest.csv`

### `code/plot_timeseries_all_EC_EO.py`
- Recursively finds `*_clean_raw.fif` under `results/processed/**/eeg/`.
- Parses subject, session (PRE/POST), and state (EO/EC) from filenames (BIDS + legacy tolerated).
- Computes sliding-window PSD (Welch) per band; aggregates by **ROI** (`ROI_CHANNELS`).
- Aligns EO/EC per subject → mean ± 95% CI; optional BH-FDR marking across time.
- Saves long CSV and 2×3 panel figures (PRE/POST × groups).

### `code/calc_power.py` & `code/calc_timeseries_power.py`
- Build wide or long tables for power metrics (ABS and REL), feeding `results/power/` and/or `results/timeseries/`.

### `code/calc_stats_power.py`
- Reads wide power tables, reshapes to long, and runs paired tests:
  - **EC vs EO** within session
  - **POST vs PRE** within visual state
- Chooses paired t-test vs Wilcoxon based on Shapiro–Wilk normality of differences; reports Cohen’s d_z and significance stars.

### `code/plot_topomaps_grid_EO_EC.py`, `code/plot_heat_all_bands_timeseries.py`, `code/plot_mirror_EO_EC_points.py`
- Additional visualization options (topographies, temporal heatmaps, and EO/EC mirror/point plots).

---

## Units & definitions

- **Absolute power (ABS)**: band-averaged PSD in **V²/Hz**. Values are often small; consider exporting with higher precision or scaling to **µV²/Hz** (`× 1e6`) for readability.
- **Relative power (REL)**: `(band power) / (total power)` with total spanning `[PSD_FMIN, min(PSD_FMAX, Nyquist-1 Hz)]`.
- **Time-series**: windows of `TS_WIN_SEC` with step `TS_STEP_SEC`; Welch params from `WELCH_SEG_SEC` and `WELCH_OVERLAP`.

---

## Troubleshooting

- **“No data extracted” in time-series**
  - Confirm `config.PROJECT_ROOT = SCRIPT_PATH.parent.parent` (since `config.py` is in `code/`).
  - Verify that FIF names include `*_clean_raw.fif` and encode `ses-pre/ses-post` and `EO/EC` (as produced by `preprocess.py`).
  - If EEG channel **types** are missing, the time-series script falls back to **name-based** selection using `ROI_CHANNELS`.

- **Deprecation warnings about `pick_types`**
  - The scripts use the modern API (`raw.pick("eeg")`, `raw.pick(<list>)`). Update any leftovers accordingly.

- **ABS means appear as `0.0` in stats**
  - Increase rounding precision when writing CSVs or scale ABS to µV²/Hz. This is a display/precision issue, not “all-zero” data.

---

## Reproducibility notes

- Record `python`, `mne`, and `mne-icalabel` versions.
- Keep `config.py` in version control (document changes to filters, bands, windows).
- Preserve `*_blocks_manifest.csv` for audit trail of EO/EC segmentation.
- Fixed randomness: `ICA(..., random_state=97)`.

---

## Citation

If this pipeline contributes to a publication, please cite **MNE-Python** and **ICLabel** and reference this repository.

---

## License

Licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.  
See [`LICENSE.txt`](LICENSE.txt).

---

**Questions or issues?** Open an issue with the command used, a short description of your data layout, and the full error output (traceback).
