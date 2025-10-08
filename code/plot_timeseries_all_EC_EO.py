"""
Time series (EO/EC) per band × ROI, split by PRE/POST and group.
Saves a long CSV and 2×3 panels (PRE/POST × groups) with means ± 95% CI and optional FDR marks.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import t, wilcoxon

import config

PROCESSED_DIR = config.PROCESSED_DIR
PLOTS_DIR     = config.PLOTS_DIR
OUT_DIR       = PLOTS_DIR / "timeseries_all"
CSV_DIR       = OUT_DIR / "csv"
FIG_DIR       = OUT_DIR / "figs"
CSV_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

BANDS         = config.BANDS
PSD_FMIN      = config.PSD_FMIN
PSD_FMAX      = config.PSD_FMAX
WELCH_SEG_SEC = config.WELCH_SEG_SEC
WELCH_OVERLAP = config.WELCH_OVERLAP

GROUPS_ORDER  = config.GROUPS_ORDER
VS_ORDER      = config.VS_ORDER
ROI_CHANNELS  = config.ROI_CHANNELS

TS_WIN_SEC    = config.TS_WIN_SEC
TS_STEP_SEC   = config.TS_STEP_SEC

ALPHA_FDR      = config.TS_FDR_ALPHA
MARK_SIG       = config.TS_MARK_SIG
GENERATE_PLOTS = config.TS_GENERATE_PLOTS

GROUP_ACTIVE   = set(config.GROUP_ACTIVE)
GROUP_PASSIVE  = set(config.GROUP_PASSIVE)
GROUP_CONTROL  = set(config.GROUP_CONTROL)


def parse_subject_session_state(stem: str):
    s = stem.upper()
    m = re.search(r"SUB[-_]?(\d{2,})", s) or re.search(r"(\d{2,})", s)
    sub = (m.group(1) if m else stem).zfill(2)
    if re.search(r"(SES[-_]?PRE|[-_]PRE)(?:[-_]|$)", s):
        sess = "PRE"
    elif re.search(r"(SES[-_]?POST|[-_](POST|POS))(?:[-_]|$)", s):
        sess = "POST"
    else:
        sess = ""
    if re.search(r"(^|[-_])EC($|[-_])", s):
        state = "EC"
    elif re.search(r"(^|[-_])EO($|[-_])", s):
        state = "EO"
    else:
        state = ""
    return sub, sess, state


def map_subject_to_group(sub: str) -> str:
    sid = str(sub).zfill(2)
    if sid in GROUP_ACTIVE:  return GROUPS_ORDER[0]
    if sid in GROUP_PASSIVE: return GROUPS_ORDER[1]
    if sid in GROUP_CONTROL: return GROUPS_ORDER[2]
    return "Unknown"


def rois_from_channels(ch_names):
    rois = {}
    for name, arr in ROI_CHANNELS.items():
        got = [c for c in arr if c in ch_names]
        if len(got) >= 2:
            rois[name] = got
    rois["All"] = list(ch_names)
    return rois


def mean_and_ci95(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    m = float(np.mean(x))
    if x.size == 1:
        return m, np.nan, np.nan
    sd = float(np.std(x, ddof=1))
    se = sd / np.sqrt(x.size)
    tcrit = t.ppf(0.975, x.size - 1)
    return m, float(m - tcrit * se), float(m + tcrit * se)


def fdr_bh_mask(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    n = p.size
    if n == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresh
    if not np.any(passed):
        return np.zeros_like(p, dtype=bool)
    k = np.max(np.where(passed)[0])
    crit = thresh[k]
    return p <= crit


def compute_series_for_raw(raw, band_tuple):
    fmin_band, fmax_band = band_tuple
    raw = raw.copy().pick("eeg")
    sf = float(raw.info["sfreq"])
    fmax_eff = min(PSD_FMAX, sf / 2.0 - 1.0)

    overlap = TS_WIN_SEC - TS_STEP_SEC
    overlap = max(0.0, min(overlap, TS_WIN_SEC - 1e-3))
    epochs = mne.make_fixed_length_epochs(
        raw, duration=TS_WIN_SEC, overlap=overlap, preload=True, verbose="ERROR"
    )
    onsets = epochs.events[:, 0] / sf
    times = onsets + TS_WIN_SEC / 2.0

    nseg = int(round(WELCH_SEG_SEC * sf))
    spec = epochs.compute_psd(
        method="welch", fmin=PSD_FMIN, fmax=fmax_eff,
        n_fft=nseg, n_per_seg=nseg, n_overlap=int(round(WELCH_OVERLAP * nseg)),
        verbose="ERROR"
    )
    psds, freqs = spec.get_data(return_freqs=True)

    band_mask  = (freqs >= fmin_band) & (freqs <= fmax_band)
    total_mask = (freqs >= PSD_FMIN)  & (freqs <= fmax_eff)

    abs_band = psds[:, :, band_mask].mean(axis=2)
    total    = psds[:, :, total_mask].mean(axis=2)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_band = abs_band / np.where(total > 0, total, np.nan)
    return times, abs_band, rel_band, epochs.ch_names


def collect_all_series():
    use_first_block = getattr(config, "TS_USE_FIRST_BLOCK_ONLY", False)
    fifs = sorted(PROCESSED_DIR.rglob("*_block1_raw.fif" if use_first_block else "*_clean_raw.fif"))
    print(f"Files found{ ' (first block only)' if use_first_block else '' }: {len(fifs)}")

    rows = []
    bank = {}
    for f in fifs:
        sub, sess, state = parse_subject_session_state(f.stem)
        if state in VS_ORDER and sess in {"PRE", "POST"}:
            bank[(sub, state, sess)] = f
    print(f"Valid (subject, state, session) triplets: {len(bank)}")

    for (sub, state, sess), path in bank.items():
        grp = map_subject_to_group(sub)
        print(f"Reading: {path.name}")
        raw = mne.io.read_raw_fif(path, preload=True, verbose="WARNING")
        try:
            raw.set_montage("standard_1020", on_missing="ignore")
        except Exception:
            pass
        for band_name, band_tuple in BANDS.items():
            times, abs_by_ch, rel_by_ch, chs = compute_series_for_raw(raw, band_tuple)
            if times.size == 0 or abs_by_ch.size == 0 or rel_by_ch.size == 0:
                print("  Skipping: no data after pick/PSD")
                continue
            rois = rois_from_channels(chs)
            for roi_name, roi_list in rois.items():
                idx = [chs.index(ch) for ch in roi_list if ch in chs]
                if len(idx) < 2 and roi_name != "All":
                    continue
                abs_roi = abs_by_ch[:, idx].mean(axis=1)
                rel_roi = rel_by_ch[:, idx].mean(axis=1)
                for tval, a, r in zip(times, abs_roi, rel_roi):
                    rows.append(dict(
                        subject=sub, group=grp, session=sess, visual_state=state,
                        time_s=float(tval), band=band_name, metric="abs", roi=roi_name, value=float(a)
                    ))
                    rows.append(dict(
                        subject=sub, group=grp, session=sess, visual_state=state,
                        time_s=float(tval), band=band_name, metric="rel", roi=roi_name, value=float(r)
                    ))
    return pd.DataFrame.from_records(rows)


def align_sessions_per_subject(df_panel):
    subs = sorted(df_panel.subject.astype(str).unique(), key=lambda s: int(s))
    ts_common = None
    PRE_list, POST_list = [], []
    for sub in subs:
        dsub = df_panel[df_panel.subject.astype(str) == sub]
        d_pre  = dsub[dsub.session == "PRE"].sort_values("time_s")
        d_post = dsub[dsub.session == "POST"].sort_values("time_s")
        n = int(min(len(d_pre), len(d_post)))
        if n == 0:
            continue
        tt = 0.5 * (d_pre.time_s.values[:n] + d_post.time_s.values[:n])
        pre  = d_pre.value.values[:n]
        post = d_post.value.values[:n]
        if ts_common is None:
            ts_common = tt
            PRE_list  = [pre]
            POST_list = [post]
        else:
            m = min(len(ts_common), len(tt))
            ts_common = ts_common[:m]
            PRE_list  = [x[:m] for x in PRE_list]
            POST_list = [y[:m] for y in POST_list]
            PRE_list.append(pre[:m])
            POST_list.append(post[:m])
    if ts_common is None:
        return None, None, None
    return ts_common, np.vstack(PRE_list), np.vstack(POST_list)


def plot_timeseries_grid(df_all, band, metric, roi):
    fig = plt.figure(figsize=(13.2, 8.0), facecolor="white")
    gs = gridspec.GridSpec(nrows=2, ncols=3, figure=fig, wspace=0.24, hspace=0.34)
    axes = np.empty((2, 3), dtype=object)
    for i in range(2):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.7)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            axes[i, j] = ax

    for j, grp in enumerate(GROUPS_ORDER):
        axes[0, j].set_title(grp, fontsize=13, pad=12)

    bbox0 = axes[0, 0].get_position()
    bbox1 = axes[1, 0].get_position()
    fig.text(0.065, (bbox0.y0 + bbox0.y1) / 2, "EO",  va="center", ha="right", fontsize=14)
    fig.text(0.065, (bbox1.y0 + bbox1.y1) / 2, "EC",  va="center", ha="right", fontsize=14)

    states = ["EO", "EC"]
    legend_handles, legend_labels = None, None
    row_ymins = [np.inf, np.inf]
    row_ymaxs = [-np.inf, -np.inf]

    time_limits = getattr(config, "TS_FIXED_X_WINDOWS", {"EO": (0, 40), "EC": (0, 40)})

    for i, state in enumerate(states):
        for j, grp in enumerate(GROUPS_ORDER):
            ax = axes[i, j]
            d = df_all[
                (df_all.visual_state == state) & (df_all.group == grp) &
                (df_all.band == band) & (df_all.metric == metric) & (df_all.roi == roi)
            ]
            if d.empty:
                ax.axis("off"); continue

            ts, PRE, POST = align_sessions_per_subject(d)
            if ts is None:
                ax.axis("off"); continue

            t_min, t_max = time_limits[state]
            mask = (ts >= t_min) & (ts <= t_max)
            ts_f    = ts[mask]
            PRE_f   = PRE[:, mask]
            POST_f  = POST[:, mask]
            if ts_f.size == 0:
                ax.axis("off"); continue

            mean_pre  = np.nanmean(PRE_f, axis=0)
            mean_post = np.nanmean(POST_f, axis=0)
            ci_pre  = np.array([mean_and_ci95(PRE_f[:, k])  for k in range(PRE_f.shape[1])])
            ci_post = np.array([mean_and_ci95(POST_f[:, k]) for k in range(POST_f.shape[1])])

            h1, = ax.plot(ts_f, mean_pre,  label="PRE",  linewidth=2.0)
            ax.fill_between(ts_f, ci_pre[:, 1],  ci_pre[:, 2],  alpha=0.18)
            h2, = ax.plot(ts_f, mean_post, label="POST", linewidth=2.0)
            ax.fill_between(ts_f, ci_post[:, 1], ci_post[:, 2], alpha=0.18)

            if legend_handles is None:
                legend_handles = [h1, h2]; legend_labels = ["PRE", "POST"]

            ax.set_xlabel("Time (s)", fontsize=11)
            if j == 0:
                ax.set_ylabel("Relative Power" if metric == "rel" else "Absolute Power", fontsize=11)
            ax.set_xlim(t_min, t_max)

            if MARK_SIG:
                pvals = []
                for k in range(PRE_f.shape[1]):
                    a = POST_f[:, k]; b = PRE_f[:, k]
                    m = np.isfinite(a) & np.isfinite(b)
                    if np.sum(m) < 3:
                        pvals.append(1.0); continue
                    try:
                        p = wilcoxon(a[m] - b[m], zero_method="wilcox").pvalue
                    except Exception:
                        p = 1.0
                    pvals.append(float(p))
                sigmask = fdr_bh_mask(np.array(pvals), alpha=ALPHA_FDR)

                y_top = np.nanmax([ci_pre[:, 2], ci_post[:, 2]])
                dy = (np.nanmax([mean_pre, mean_post]) - np.nanmin([mean_pre, mean_post]))
                y_mark = y_top + (0.06 if np.isfinite(dy) else 0.06) * (dy if np.isfinite(dy) and dy > 0 else 1.0)
                ax.plot(ts_f, np.where(sigmask, y_mark, np.nan), linewidth=3.0, color="green", solid_capstyle="butt")

                y_min = np.nanmin([ci_pre[:, 1], ci_post[:, 1]])
                y_max = max(y_mark, np.nanmax([ci_pre[:, 2], ci_post[:, 2]]))
            else:
                y_min = np.nanmin([ci_pre[:, 1], ci_post[:, 1]])
                y_max = np.nanmax([ci_pre[:, 2], ci_post[:, 2]])

            row_ymins[i] = min(row_ymins[i], y_min)
            row_ymaxs[i] = max(row_ymaxs[i], y_max)

    for i in range(2):
        ymin, ymax = row_ymins[i], row_ymaxs[i]
        pad = 0.04 * (ymax - ymin if np.isfinite(ymax - ymin) and (ymax - ymin) > 0 else 1.0)
        for j in range(3):
            ax = axes[i, j]
            if ax.has_data():
                ax.set_ylim(ymin - pad, ymax + pad)

    fig.suptitle(f"Time Series — {band} — {metric.upper()} — ROI: {roi}", fontsize=16, y=0.97)
    plt.subplots_adjust(left=0.12, right=0.86, bottom=0.10, top=0.90)

    if legend_handles is not None:
        fig.legend(legend_handles, legend_labels,
                   loc="upper right", bbox_to_anchor=(0.98, 0.98),
                   frameon=True, fontsize=11)

    fname = FIG_DIR / f"timeseries_{band}_{metric}_{roi.replace(' ','_')}.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


def main():
    print("Scanning derivatives…", PROCESSED_DIR.resolve())
    use_first_block = getattr(config, "TS_USE_FIRST_BLOCK_ONLY", False)
    pat = "*_block1_raw.fif" if use_first_block else "*_clean_raw.fif"
    print(f"Files matching {pat}:", len(list(PROCESSED_DIR.rglob(pat))))

    print("Extracting time series for all bands/ROIs (relative and absolute)…")
    df = collect_all_series()
    if df.empty:
        print("No data extracted. Check PROCESSED_DIR and filenames.")
        return

    csv_all = CSV_DIR / "timeseries_all_bands_rois_relabs.csv"
    df.to_csv(csv_all, index=False)
    print(f"Saved CSV: {csv_all}")

    if GENERATE_PLOTS:
        bands = [b for b in BANDS.keys() if b in df.band.unique()]
        metrics = ["rel", "abs"]
        rois = sorted(df.roi.unique())
        for band in bands:
            for metric in metrics:
                for roi in rois:
                    plot_timeseries_grid(df, band, metric, roi)

    print("Done.")


if __name__ == "__main__":
    main()
