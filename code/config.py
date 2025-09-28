"""
Global configuration
"""

from pathlib import Path

# === Paths ====================================================================
SCRIPT_PATH   = Path(__file__).resolve()
PROJECT_ROOT  = SCRIPT_PATH.parent.parent
DATA_DIR      = PROJECT_ROOT / "data"
RESULTS_DIR   = PROJECT_ROOT / "results"
PLOTS_DIR     = RESULTS_DIR / "plots"
POWER_DIR     = RESULTS_DIR / "power"
TS_DIR        = RESULTS_DIR / "timeseries"
PROCESSED_DIR = RESULTS_DIR / "processed"

# === EEG processing ===========================================================
FILTER_LOW  = 0.5      # Hz
FILTER_HIGH = 50.0     # Hz
NOTCH_HZ    = 60       # Hz

# EO/EC blocks (seconds from recording start)
BLOCKS_WITH_STATE = [
    ("EO", (15, 135)),
    ("EC", (150, 270)),
    ("EO", (285, 405)),
    ("EC", (420, 540)),
]

# === Welch / PSD ==============================================================
WELCH_SEG_SEC = 4.0    # s
WELCH_OVERLAP = 0.5
PSD_FMIN      = 0.5    # Hz
PSD_FMAX      = 50.0   # Hz

# === Power export =============================================================
EXPORT_RELATIVE          = True
STANDARDIZE_DURATION_SEC = None  # e.g. 120.0

# === Study design =============================================================
GROUP_ACTIVE  = {'01', '05', '07', '10', '15', '16', '19'}
GROUP_PASSIVE = {'03', '04', '06', '08', '11', '13', '21'}
GROUP_CONTROL = {'02', '09', '12', '14', '17', '18', '22'}

SUBJECTS_ORDER     = sorted(GROUP_ACTIVE | GROUP_PASSIVE | GROUP_CONTROL)
GROUP_ACTIVE_ORDER = sorted(GROUP_ACTIVE)
GROUP_PASSIVE_ORDER = sorted(GROUP_PASSIVE)
GROUP_CONTROL_ORDER = sorted(GROUP_CONTROL)

# Channels 
ACTIVE_CHANNELS = {
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "C3", "Cz", "C4",
    "FT9", "T7", "T8", "FT10", "TP9", "TP10",
    "CP5", "CP1", "CP2", "CP6",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
}

# === Bands & ROIs =============================================================
BANDS = {
    "Delta": (0.1, 3.5),
    "Theta": (4.0, 7.9),
    "Alpha": (8.0, 12.9),
    "Beta":  (13.0, 30.0),
    "Gamma": (30.1, 50.0),
}

REGIONS = {
    "Prefrontal":       ["Fp1", "Fp2"],
    "Frontal":          ["F7", "F3", "Fz", "F4", "F8"],
    "Frontocentral":    ["FC5", "FC1", "FC2", "FC6"],
    "Central":          ["C3", "Cz", "C4"],
    "Temporo-parietal": ["FT9", "T7", "T8", "FT10", "TP9", "TP10"],
    "Centro-parietal":  ["CP5", "CP1", "CP2", "CP6"],
    "Parietal":         ["P7", "P3", "Pz", "P4", "P8"],
    "Occipital":        ["O1", "Oz", "O2"],
}

# === Canonical orders / palettes =============================================
GROUPS        = ["Active", "Passive", "Control"]
SESSIONS      = ["PRE", "POST"]
VISUAL_STATES = ["EO", "EC"]

BANDS_ORDER  = list(BANDS.keys())
ROIS_ORDER   = list(REGIONS.keys())
GROUPS_ORDER = GROUPS
SESS_ORDER   = SESSIONS
VS_ORDER     = VISUAL_STATES

PALETTE_SESS = {"PRE": "#72B2E7", "POST": "#003366"}
PALETTE_VS   = {"EO": "#7FB3D5", "EC": "#1F618D"}

# === Time-series ==============================================================
TS_WIN_SEC        = 4.0
TS_STEP_SEC       = 1.0
TS_FDR_ALPHA      = 0.05
TS_MARK_SIG       = True
TS_GENERATE_PLOTS = True

ROI_CHANNELS = REGIONS