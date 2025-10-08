"""Heatmaps of band×ROI power over time by participant.
Rows: EO-PRE, EO-POST, EC-PRE, EC-POST; Columns: Active, Passive, Control.
Input : results/timeseries/ts_power_long.csv
Output: results/plots/timeseries_heatmaps_prepost/*.png (+ optional CSV summaries)
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.integrate import trapezoid
import config

# Paths / orders
TS_CSV  = config.TS_DIR / "ts_power_long.csv"
OUT_DIR = config.PLOTS_DIR / "timeseries_heatmaps_prepost"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUPS      = config.GROUPS_ORDER
STATES      = config.VS_ORDER
SESSIONS    = config.SESS_ORDER
BANDS_ORDER = config.BANDS_ORDER
ROIS_ORDER  = config.ROIS_ORDER

ROW_ORDER = [
    (STATES[0], SESSIONS[0]),  # EO-PRE
    (STATES[0], SESSIONS[1]),  # EO-POST
    (STATES[1], SESSIONS[0]),  # EC-PRE
    (STATES[1], SESSIONS[1]),  # EC-POST
]

# --------------------------- helpers ------------------------------------------
def compute_value_range(x, p_lo=5, p_hi=95):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (0.0, 1.0)
    vmin = float(np.percentile(x, p_lo))
    vmax = float(np.percentile(x, p_hi))
    if not np.isfinite(vmin): vmin = float(np.nanmin(x))
    if not np.isfinite(vmax): vmax = float(np.nanmax(x))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        span = 1.0 if not np.isfinite(vmax) else max(1e-12, abs(vmax) * 0.1)
        return (float(vmax - span), float(vmax + span))
    return (vmin, vmax)

def format_subject_ids(sids, mode="z2"):
    if mode == "S":
        return [f"S{str(s).zfill(2)}" for s in sids]
    if mode == "z2":
        return [str(s).zfill(2) for s in sids]
    return [str(s) for s in sids]

def order_subjects_for_pair(df_pair, sort_mode, value_col):
    sids = sorted(df_pair["subject"].astype(str).unique(), key=lambda s: int(s))
    if sort_mode == "none":
        return sids
    metrics = {}
    for sid, d_sid in df_pair.groupby("subject"):
        sid = str(sid)
        m_pre  = float(d_sid.loc[d_sid["session"] == "PRE",  value_col].mean())
        m_post = float(d_sid.loc[d_sid["session"] == "POST", value_col].mean())
        metrics[sid] = {
            "mean_pre": m_pre,
            "mean_post": m_post,
            "mean_post_minus_pre": (m_post - m_pre),
            "grand_mean": float(d_sid[value_col].mean()),
        }.get(sort_mode, np.nan)
    sids.sort(key=lambda s: (metrics.get(s, np.nan), int(s)))
    return sids

def detect_time_block_boundaries(tt):
    tt = np.asarray(tt, float)
    if tt.size < 3:
        return []
    dt = np.diff(tt)
    hop = np.median(dt)
    if not np.isfinite(hop) or hop <= 0:
        return []
    return list(tt[1:][dt > 1.3 * hop])

def get_fixed_window(state, session):
    """Return (x_lo, x_hi) from config.TS_FIXED_X_WINDOWS; supports keys 'EO'/'EC' or (state, session)."""
    win_cfg = getattr(config, "TS_FIXED_X_WINDOWS", None)
    if not win_cfg:
        return None
    key_ss = (str(state), str(session))
    if key_ss in win_cfg:
        lo, hi = win_cfg[key_ss]; return float(lo), float(hi)
    if str(state) in win_cfg:
        lo, hi = win_cfg[str(state)]; return float(lo), float(hi)
    return None

# --------------------------- panel --------------------------------------------
def render_heatmap_panel(df, band, roi, args, scale_cache, value_col):
    sub = df[(df["band"] == band) & (df["region"] == roi) & (df["group"].isin(GROUPS))].copy()
    if sub.empty:
        print(f"[WARN] No data for {band} / {roi}.")
        return None, None

    # Color limits
    if value_col == "power_rel":
        vmin, vmax = 0.0, 1.0
    else:
        if args.vmin is not None and args.vmax is not None:
            vmin, vmax = float(args.vmin), float(args.vmax)
        else:
            if args.scale == "band":
                key = ("band", band)
                if key not in scale_cache:
                    scale_cache[key] = compute_value_range(df.loc[df["band"] == band, value_col].values, 5, 95)
                vmin, vmax = scale_cache[key]
            elif args.scale == "global":
                key = ("global",)
                if key not in scale_cache:
                    scale_cache[key] = compute_value_range(df[value_col].values, 5, 95)
                vmin, vmax = scale_cache[key]
            else:
                vmin, vmax = compute_value_range(sub[value_col].values, 5, 95)

    fig, axes = plt.subplots(4, 3, figsize=(14, 12), sharex=False, sharey=False)
    summary_rows = []

    for row_idx, (state, session) in enumerate(ROW_ORDER):
        for col_idx, group in enumerate(GROUPS):
            ax = axes[row_idx, col_idx]
            d_group = sub[(sub["visual_state"] == state) & (sub["group"] == group)]
            if d_group.empty:
                ax.set_axis_off()
                continue

            d_pair = sub[(sub["visual_state"] == state) & (sub["group"] == group) &
                         (sub["session"].isin(SESSIONS))].copy()
            sids = order_subjects_for_pair(d_pair, args.sort, value_col)

            d_this = d_group[d_group["session"] == session]
            if d_this.empty:
                ax.set_axis_off()
                continue
            # Apply fixed window, if configured
            x_window = get_fixed_window(state, session)
            if x_window is not None:
                x_lo, x_hi = x_window
                d_this = d_this[(d_this["t_sec"] >= x_lo) & (d_this["t_sec"] <= x_hi)].copy()
                if d_this.empty:
                    ax.set_axis_off()
                    continue
                tt = np.sort(d_this["t_sec"].round(2).unique())
                if tt.size < 2:
                    step = getattr(config, "TS_STEP_SEC", 1.0)
                    n = max(2, int(round((x_hi - x_lo) / max(step, 1e-9))) + 1)
                    tt = np.linspace(x_lo, x_hi, n)
                extent_x = (x_lo, x_hi)
            else:
                tt = np.sort(d_this["t_sec"].round(2).unique())
                extent_x = (float(tt.min()), float(tt.max()))

            M = np.full((len(sids), len(tt)), np.nan, float)
            sid_index = {s: i for i, s in enumerate(sids)}
            t_index   = {t: j for j, t in enumerate(tt)}
            for (sid, t0), dd in d_this.groupby(["subject", d_this["t_sec"].round(2)]):
                M[sid_index[str(sid)], t_index[float(t0)]] = np.nanmean(dd[value_col].values)

            ax.imshow(
                M, aspect="auto", origin="lower", interpolation="nearest",
                cmap=args.cmap, norm=Normalize(vmin=vmin, vmax=vmax),
                extent=[extent_x[0], extent_x[1], 0, len(sids)]
            )
            if x_window is not None:
                ax.set_xlim(extent_x)
            yticks = np.arange(len(sids)) + 0.5
            ax.set_yticks(yticks)
            ax.set_yticklabels(format_subject_ids(sids, args.id_format), fontsize=9)

            ax.set_ylabel(f"{state}-{session}" if col_idx == 0 else "", fontsize=12)
            if row_idx == 0:
                ax.set_title(group, fontsize=12, pad=10)
            ax.set_xlabel("Time (s)")

            if args.mark_blocks == "on":
                for xc in detect_time_block_boundaries(tt):
                    ax.axvline(xc, color="k", ls="--", lw=0.8, alpha=0.25)

            for sid in sids:
                d_sid = d_this[d_this["subject"].astype(str) == sid]
                m = float(d_sid[value_col].mean()) if len(d_sid) else np.nan
                auc = float(trapezoid(d_sid[value_col].values, x=d_sid["t_sec"].values)) if len(d_sid) else np.nan
                summary_rows.append(dict(
                    band=band, region=roi, group=group, state=state, session=session,
                    subject=str(sid), mean=m, auc=auc
                ))

    cax = fig.add_axes([0.92, 0.12, 0.02, 0.78])
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=args.cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    unit = "Relative Power (proportion)" if value_col == "power_rel" else r"Absolute Power ($\mu V^2/Hz$)"
    cb.set_label(f"{band} — {unit}", rotation=90)

    fig.suptitle(f"Time × Participant Heatmap — {band} — {roi}", fontsize=15, y=0.99)
    plt.subplots_adjust(left=0.08, right=0.90, top=0.95, bottom=0.07, wspace=0.18, hspace=0.35)

    return fig, pd.DataFrame(summary_rows)

# ----------------------------- main -------------------------------------------
def main():
    p = argparse.ArgumentParser(description="PRE vs POST heatmaps (EO/EC) by group.")
    p.add_argument("--metric", choices=["power_rel", "power_abs", "both"], default="both")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--rois",  nargs="*", default=None)
    p.add_argument("--bands", nargs="*", default=None)
    p.add_argument("--sort", choices=["none","mean_pre","mean_post","mean_post_minus_pre","grand_mean"],
                   default="none")
    p.add_argument("--id-format", choices=["plain","z2","S"], default="z2")
    p.add_argument("--mark-blocks", choices=["on","off"], default="on")
    p.add_argument("--export-summary", choices=["on","off"], default="on")
    p.add_argument("--scale", choices=["panel","band","global"], default="panel")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    args = p.parse_args()

    df = pd.read_csv(TS_CSV)

    need = {"subject","group","session","visual_state","region","band","t_sec"}
    miss = need.difference(df.columns)
    if miss:
        raise SystemExit(f"Missing base columns in CSV: {miss}")

    # Enforce canonical category orders for deterministic grouping/labels
    df["band"]         = pd.Categorical(df["band"], BANDS_ORDER, ordered=True)
    df["region"]       = pd.Categorical(df["region"], ROIS_ORDER,  ordered=True)
    df["group"]        = pd.Categorical(df["group"], GROUPS,       ordered=True)
    df["session"]      = df["session"].astype(str).str.upper().map({"PRE":"PRE","POS":"POST","POST":"POST"})
    df["session"]      = pd.Categorical(df["session"], SESSIONS,   ordered=True)
    df["visual_state"] = pd.Categorical(df["visual_state"].astype(str).str.upper(), STATES, ordered=True)
    df["subject"]      = df["subject"].astype(str).str.zfill(2)

    # Resolve band/ROI selection
    bands_all = [b for b in BANDS_ORDER if b in df["band"].cat.categories]
    rois_all  = [r for r in ROIS_ORDER  if r in df["region"].cat.categories]
    bands = args.bands or bands_all
    rois  = args.rois  or rois_all

    metrics = ["power_rel","power_abs"] if args.metric == "both" else [args.metric]
    for m in metrics:
        if m not in df.columns:
            raise SystemExit(f"Column {m} not found in {TS_CSV}")

    print(f"Bands: {bands}")
    print(f"ROIs : {rois}")
    print(f"Metrics: {metrics} | Sort: {args.sort} | ID-format: {args.id_format} | mark_blocks: {args.mark_blocks}")

    scale_cache = {}
    for m in metrics:
        for band in bands:
            for roi in rois:
                fig, df_sum = render_heatmap_panel(df, band, roi, args, scale_cache, value_col=m)
                if fig is None:
                    continue
                tag = "REL" if m == "power_rel" else "ABS"
                stem = f"heat_PREPOST_{band}_{roi.replace(' ','_').replace('/','-')}_{tag}_{args.sort}.png"
                out_png = OUT_DIR / stem
                fig.savefig(out_png, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved: {out_png}")

                if args.export_summary == "on" and df_sum is not None and not df_sum.empty:
                    piv = df_sum.pivot_table(index=["band","region","group","state","subject"],
                                             columns="session", values=["mean","auc"])
                    piv.columns = [f"{a}_{b}" for a,b in piv.columns]
                    piv = piv.reset_index()
                    if "mean_PRE" in piv and "mean_POST" in piv:
                        piv["delta_mean"] = piv["mean_POST"] - piv["mean_PRE"]
                    if "auc_PRE" in piv and "auc_POST" in piv:
                        piv["delta_auc"] = piv["auc_POST"] - piv["auc_PRE"]

                    if "delta_mean" in piv and piv["delta_mean"].std(ddof=1) > 0:
                        z = (piv["delta_mean"] - piv["delta_mean"].mean()) / piv["delta_mean"].std(ddof=1)
                        piv["flag_outlier"] = np.where(np.abs(z) >= 2, 1, 0)
                    else:
                        piv["flag_outlier"] = 0

                    out_csv = OUT_DIR / f"summary_{band}_{roi.replace(' ','_').replace('/','-')}_{tag}.csv"
                    piv.to_csv(out_csv, index=False)
                    print(f"Saved summary: {out_csv}")

if __name__ == "__main__":
    main()