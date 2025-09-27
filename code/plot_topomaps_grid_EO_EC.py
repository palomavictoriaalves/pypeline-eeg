# 2×3 panel (EO/EC × Active/Passive/Control) per band, POST−PRE topomaps.
# Exports REL and ABS versions.

from pathlib import Path
import re
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config

PROCESSED_DIR = config.PROCESSED_DIR  
OUT_REL = config.PLOTS_DIR / "topomaps_grid_rel"
OUT_ABS = config.PLOTS_DIR / "topomaps_grid_abs"
OUT_REL.mkdir(parents=True, exist_ok=True)
OUT_ABS.mkdir(parents=True, exist_ok=True)

BANDS         = config.BANDS
PSD_FMIN      = config.PSD_FMIN
PSD_FMAX      = config.PSD_FMAX
WELCH_SEG_SEC = config.WELCH_SEG_SEC
WELCH_OVERLAP = config.WELCH_OVERLAP

GROUPS = config.GROUPS_ORDER             
STATES = config.VS_ORDER                

# --- subject→group map --------------------------------------------------------
def infer_group(sub: str) -> str:
    sid = str(sub).zfill(2)
    act = set(getattr(config, "GROUP_ACTIVE", set()))
    pas = set(getattr(config, "GROUP_PASSIVE", set()))
    ctl = set(getattr(config, "GROUP_CONTROL", set()))
    if sid in act: return config.GROUPS_ORDER[0]  # "Active"
    if sid in pas: return config.GROUPS_ORDER[1]  # "Passive"
    if sid in ctl: return config.GROUPS_ORDER[2]  # "Control"
    return "Unknown"

_BIDS_BASE_RE = re.compile(
    r"(?P<sub>sub-\d+)"
    r"(?:_(?P<ses>ses-[a-z0-9]+))?"
    r"_task-(?P<task>[^_]+)"
    r"(?:_run-(?P<run>[^_]+))?"
    r"_desc-(?P<desc>[^_]+)"
    r"_(?P<what>clean_raw|EO_clean_raw|EC_clean_raw|blocks_manifest)\.(?P<ext>fif|csv)$",
    re.IGNORECASE
)

def _norm_session_label(ses: str | None) -> str:
    if not ses:
        return ""
    s = ses.lower()
    if s.endswith("pre"):  return "PRE"
    if s.endswith("post"): return "POST"
    return s.upper()

def _subject_number(sub: str) -> str:
    m = re.search(r"sub-(\d+)", sub, re.IGNORECASE)
    return m.group(1).zfill(2) if m else sub

def parse_bids_from_name(path: Path):
    m = _BIDS_BASE_RE.search(path.name)
    if not m:
        return None
    d = m.groupdict()
    bits = [d["sub"]]
    if d.get("ses"): bits.append(d["ses"])
    bits.append(f"task-{d['task']}")
    if d.get("run"): bits.append(f"run-{d['run']}")
    base = "_".join(bits) + f"_desc-{d['desc']}"
    return {
        "sub": d["sub"],
        "ses": d.get("ses"),
        "task": d["task"],
        "run": d.get("run"),
        "desc": d["desc"],
        "what": d["what"],
        "ext": d["ext"],
        "base": base
    }

def open_state_raw(dir_eeg: Path, base: str, state: str):
    """
    Open <base>_{EO|EC}_clean_raw.fif if present; otherwise crop from
    <base>_clean_raw.fif using annotations visual_state:EO/EC.
    """
    direct = dir_eeg / f"{base}_{state}_clean_raw.fif"
    if direct.exists():
        try:
            return mne.io.read_raw_fif(direct, preload=True, verbose="ERROR").pick("eeg")
        except Exception as e:
            print(f"  > WARNING: failed to open {direct.name}: {e}")

    concat = dir_eeg / f"{base}_clean_raw.fif"
    if concat.exists():
        try:
            raw_all = mne.io.read_raw_fif(concat, preload=True, verbose="ERROR")
            if not raw_all.annotations or len(raw_all.annotations) == 0:
                return None
            sf = float(raw_all.info.get("sfreq", 250.0)); eps = 1.0 / sf
            pieces = []
            for desc, onset, dur in zip(raw_all.annotations.description,
                                        raw_all.annotations.onset,
                                        raw_all.annotations.duration):
                if desc == f"visual_state:{state}" and dur > 0:
                    t0 = float(onset); t1 = min(float(onset + dur), float(raw_all.times[-1]) - eps)
                    if t1 > t0:
                        pieces.append(raw_all.copy().crop(tmin=t0, tmax=t1))
            if not pieces:
                return None
            raw = mne.concatenate_raws(pieces, verbose="WARNING") if len(pieces) > 1 else pieces[0]
            return raw.pick("eeg")
        except Exception as e:
            print(f"  > WARNING: failed to open/crop {concat.name}: {e}")
    return None

# --- figure layout (script-local) ---------------------------------------------
FIGSIZE = (13.0, 7.8)
GRID_WSPACE = 0.12
GRID_HSPACE = 0.30
CBAR_WIDTH  = 0.045
TITLE_Y     = 0.99
MARGINS     = dict(left=0.08, right=0.95, top=0.90, bottom=0.06)
COL_TITLE_PAD = 10

# --- PSD per-channel for a band ----------------------------------------------
def psd_band_by_channel(raw, fmin_band, fmax_band):
    raw = raw.copy().pick("eeg")
    sf = float(raw.info["sfreq"])
    fmax_eff = min(PSD_FMAX, sf/2.0 - 1.0)
    n_per_seg = max(16, int(round(WELCH_SEG_SEC * sf)))
    n_overlap = int(round(WELCH_OVERLAP * n_per_seg))
    spec = raw.compute_psd(
        method="welch",
        fmin=PSD_FMIN, fmax=fmax_eff,
        n_fft=n_per_seg, n_per_seg=n_per_seg, n_overlap=n_overlap,
        picks="eeg", verbose="ERROR"
    )
    psds, freqs = spec.get_data(return_freqs=True)
    band_mask  = (freqs >= fmin_band) & (freqs <= fmax_band)
    total_mask = (freqs >= PSD_FMIN)  & (freqs <= fmax_eff)
    abs_band = psds[:, band_mask].mean(axis=1)
    total    = psds[:, total_mask].mean(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_band = abs_band / np.where(total > 0, total, np.nan)
    return abs_band, rel_band, raw.ch_names

def make_info(ch_names, sfreq=250.0):
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage("standard_1020")
    return info

# --- collect PRE/POST per (subject, state) using BIDS derivatives -------------
# find all concatenated derivatives to define available (sub, ses) bases
concat_files = sorted(PROCESSED_DIR.rglob("*_desc-preproc_clean_raw.fif"))

bank = {}  # (sub_num, state) -> {"PRE": (dir_eeg, base), "POST": (...)}
for f in concat_files:
    info = parse_bids_from_name(f)
    if not info:
        continue
    dir_eeg = f.parent
    base    = info["base"]
    sub     = _subject_number(info["sub"])
    sess    = _norm_session_label(info.get("ses"))
    if sess not in {"PRE","POST"}:
        continue
    # estados possíveis: EO, EC — testamos existência ou possibilidade de crop
    for state in STATES:
        have = (dir_eeg / f"{base}_{state}_clean_raw.fif").exists() or True  # sempre dá para tentar crop
        if have:
            bank.setdefault((sub, state), {})[sess] = (dir_eeg, base)

pairs = [((sub, state), sesmap) for (sub, state), sesmap in bank.items() if "PRE" in sesmap and "POST" in sesmap]
print(f"Valid PRE/POST pairs: {len(pairs)}")

# --- precompute per subject/state/band (abs & rel) ----------------------------
# cache[(sub, state, sess, band)] = {"abs": array(n_ch), "rel": array(n_ch), "ch": ch_names}
cache = {}
for (sub, state), sesmap in pairs:
    for sess in ("PRE","POST"):
        dir_eeg, base = sesmap[sess]
        raw = open_state_raw(dir_eeg, base, state)
        if raw is None:
            cache[(sub, state, sess, None)] = None
            continue
        try:
            raw.set_montage("standard_1020", on_missing="ignore")
        except Exception:
            pass
        for band, (fmin_b, fmax_b) in BANDS.items():
            abs_v, rel_v, chs = psd_band_by_channel(raw, fmin_b, fmax_b)
            cache[(sub, state, sess, band)] = {"abs": abs_v, "rel": rel_v, "ch": chs}

# --- build POST−PRE deltas ----------------------------------------------------
def build_results(metric: str):
    results = {st: {b: [] for b in BANDS} for st in STATES}
    for (sub, state), sesmap in pairs:
        grp = infer_group(sub)
        for band in BANDS.keys():
            pre  = cache.get((sub, state, "PRE",  band))
            post = cache.get((sub, state, "POST", band))
            if pre is None or post is None:
                continue
            ch_pre, ch_post = pre["ch"], post["ch"]
            common = [ch for ch in ch_pre if ch in ch_post]
            if len(common) < 8:
                continue
            idx_pre  = [ch_pre.index(ch)  for ch in common]
            idx_post = [ch_post.index(ch) for ch in common]
            delta = (post["rel"][idx_post] - pre["rel"][idx_pre]) if metric == "rel" \
                    else (post["abs"][idx_post] - pre["abs"][idx_pre])
            results[state][band].append({"sub": sub, "group": grp, "ch": common, "delta": delta})
    return results

# --- plot 2×3 grid for a band -------------------------------------------------
FIGSIZE = (13.0, 7.8)
GRID_WSPACE = 0.12
GRID_HSPACE = 0.30
CBAR_WIDTH  = 0.045
TITLE_Y     = 0.99
MARGINS     = dict(left=0.08, right=0.95, top=0.90, bottom=0.06)
COL_TITLE_PAD = 10

def plot_grid_for_band(band: str, results, metric: str, out_dir: Path):
    data_rows, ch_common = {}, {}
    for state in STATES:
        lst = results[state][band]
        if not lst:
            data_rows[state] = None
            ch_common[state] = []
            continue
        inter = set(lst[0]["ch"])
        for d in lst[1:]:
            inter &= set(d["ch"])
        inter = [ch for ch in lst[0]["ch"] if ch in inter]
        ch_common[state] = inter

        grid_vals = {}
        for grp in GROUPS:
            ds = [d for d in lst if d["group"] == grp]
            if not ds or len(inter) < 8:
                grid_vals[grp] = None
                continue
            mats = []
            for d in ds:
                idx = [d["ch"].index(ch) for ch in inter]
                mats.append(d["delta"][idx])
            grid_vals[grp] = np.nanmean(np.vstack(mats), axis=0)
        data_rows[state] = grid_vals

    # shared vlim across both rows (EO/EC) and all groups
    all_vals = []
    for state in STATES:
        gv = data_rows.get(state)
        if not gv:
            continue
        for grp in GROUPS:
            if gv and gv.get(grp) is not None:
                all_vals.append(np.abs(gv[grp]))
    if not all_vals:
        print(f"[{band}] Not enough data.")
        return
    vmax = float(np.nanpercentile(np.concatenate(all_vals), 95))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = float(np.nanmax(np.concatenate(all_vals)))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1e-6
    vlim = (-vmax, vmax)

    fig = plt.figure(figsize=FIGSIZE)
    gs = gridspec.GridSpec(
        nrows=2, ncols=4, figure=fig,
        width_ratios=[1, 1, 1, CBAR_WIDTH / (3 + CBAR_WIDTH)],
        wspace=GRID_WSPACE, hspace=GRID_HSPACE
    )
    axes = np.empty((2,3), dtype=object)
    for i in range(2):
        for j in range(3):
            axes[i, j] = fig.add_subplot(gs[i, j])
    cax = fig.add_subplot(gs[:, 3])

    for j, grp in enumerate(GROUPS):
        axes[0, j].set_title(grp, fontsize=13, pad=COL_TITLE_PAD)

    im = None
    for i, state in enumerate(STATES):
        for j, grp in enumerate(GROUPS):
            ax = axes[i, j]
            vec = None if (data_rows.get(state) is None) else data_rows[state].get(grp)
            inter = ch_common.get(state, [])
            if vec is None or len(inter) < 8:
                ax.axis("off")
                continue
            info = make_info(inter, sfreq=250.0)
            im, _ = mne.viz.plot_topomap(
                vec, info, cmap="RdBu_r", vlim=vlim,
                contours=0, sensors=True, axes=ax, show=False
            )

    fig.text(0.06, 0.72, STATES[0], va="center", ha="left", fontsize=13)
    fig.text(0.06, 0.26, STATES[1], va="center", ha="left", fontsize=13)

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Increase (red)  /  Decrease (blue)")

    metric_title = "Relative Difference (POST − PRE)" if metric == "rel" else "Absolute Difference (POST − PRE)"
    tag = "relDiff" if metric == "rel" else "absDiff"
    fig.suptitle(f"Topomaps — {band}  |  {metric_title}", fontsize=17, y=TITLE_Y)

    plt.subplots_adjust(**MARGINS)
    fname = out_dir / f"topogrid_{tag}_{band}.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"🖼️ saved: {fname}")

# --- run (REL & ABS) ----------------------------------------------------------
results_rel = build_results(metric="rel")
results_abs = build_results(metric="abs")

for band in BANDS:
    plot_grid_for_band(band, results_rel, metric="rel", out_dir=OUT_REL)
    plot_grid_for_band(band, results_abs, metric="abs", out_dir=OUT_ABS)

print("\n🎉 Topomaps generated (REL and ABS).")