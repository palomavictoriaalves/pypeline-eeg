# code/plot_heat_prepost_timeseries.py
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

# Helpers
def robust_limits(x, p_lo=5, p_hi=95):
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

def format_ids(sids, mode="z2"):
    if mode == "S":
        return [f"S{str(s).zfill(2)}" for s in sids]
    if mode == "z2":
        return [str(s).zfill(2) for s in sids]
    return [str(s) for s in sids]

def subject_order_for_pair(df_pair, sort_mode, value_col):
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

def detect_block_cuts(tt):
    tt = np.asarray(tt, float)
    if tt.size < 3:
        return []
    dt = np.diff(tt)
    hop = np.median(dt)
    if not np.isfinite(hop) or hop <= 0:
        return []
    return list(tt[1:][dt > 1.3 * hop])

# Panel
def make_panel(df, band, roi, args, scale_cache, value_col):
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
                    scale_cache[key] = robust_limits(df.loc[df["band"] == band, value_col].values, 5, 95)
                vmin, vmax = scale_cache[key]
            elif args.scale == "global":
                key = ("global",)
                if key not in scale_cache:
                    scale_cache[key] = robust_limits(df[value_col].values, 5, 95)
                vmin, vmax = scale_cache[key]
            else:
                vmin, vmax = robust_limits(sub[value_col].values, 5, 95)

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
            sids = subject_order_for_pair(d_pair, args.sort, value_col)

            d_this = d_group[d_group["session"] == session]
            if d_this.empty:
                ax.set_axis_off()
                continue
            tt = np.sort(d_this["t_sec"].round(2).unique())

            M = np.full((len(sids), len(tt)), np.nan, float)
            sid_index = {s: i for i, s in enumerate(sids)}
            t_index   = {t: j for j, t in enumerate(tt)}
            for (sid, t0), dd in d_this.groupby(["subject", d_this["t_sec"].round(2)]):
                M[sid_index[str(sid)], t_index[float(t0)]] = np.nanmean(dd[value_col].values)

            ax.imshow(
                M, aspect="auto", origin="lower", interpolation="nearest",
                cmap=args.cmap, norm=Normalize(vmin=vmin, vmax=vmax),
                extent=[tt.min(), tt.max(), 0, len(sids)]
            )
            yticks = np.arange(len(sids)) + 0.5
            ax.set_yticks(yticks)
            ax.set_yticklabels(format_ids(sids, args.id_format), fontsize=9)

            ax.set_ylabel(f"{state}-{session}" if col_idx == 0 else "", fontsize=12)
            if row_idx == 0:
                ax.set_title(group, fontsize=12, pad=10)
            ax.set_xlabel("Time (s)")

            if args.mark_blocks == "on":
                for xc in detect_block_cuts(tt):
                    ax.axvline(xc, color="k", ls="--", lw=0.8, alpha=0.25)

            from numpy import trapz
            for sid in sids:
                d_sid = d_group[(d_group["session"] == session) & (d_group["subject"].astype(str) == sid)]
                m = float(d_sid[value_col].mean()) if len(d_sid) else np.nan
                auc = float(trapz(d_sid[value_col].values, x=d_sid["t_sec"].values)) if len(d_sid) else np.nan
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

# Main
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

    bands_all = list(df["band"].dropna().unique())
    if BANDS_ORDER:
        order_idx = {b:i for i,b in enumerate(BANDS_ORDER)}
        bands_all.sort(key=lambda b: order_idx.get(b, 999))
    rois_all = list(df["region"].dropna().unique())
    if ROIS_ORDER:
        roi_idx = {r:i for i,r in enumerate(ROIS_ORDER)}
        rois_all.sort(key=lambda r: roi_idx.get(r, 999))

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
                fig, df_sum = make_panel(df, band, roi, args, scale_cache, value_col=m)
                if fig is None:
                    continue
                tag = "REL" if m == "power_rel" else "ABS"
                stem = f"heat_PREPOST_{band}_{roi.replace(' ','_').replace('/','-')}_{tag}_{args.sort}.png"
                out_png = OUT_DIR / stem
                fig.savefig(out_png, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"🖼️ saved: {out_png}")

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
                    print(f"📄 summary saved: {out_csv}")

if __name__ == "__main__":
    main()