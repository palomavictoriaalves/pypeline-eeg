# calc_stats_power.py
# Minimal paired stats per ROI__Band column.
# Outputs columns:
# mean_A, mean_B, delta, normality_test, normality_p,
# test_used (t_paired or wilcoxon), p_value, sig (asterisks), d (Cohen's dz)

from pathlib import Path
import re
import numpy as np
import pandas as pd
from math import sqrt
from scipy import stats

import config  # requires POWER_DIR and RESULTS_DIR in config

ALPHA = 0.05  # normality threshold

POWER_DIR = config.POWER_DIR
RESULTS_DIR = config.RESULTS_DIR
OUT_ROOT = (RESULTS_DIR / "stats").resolve()
OUT_ROOT.mkdir(parents=True, exist_ok=True)

REL_PATH = POWER_DIR / "power_wide_rel_EO_EC.csv"
ABS_PATH = POWER_DIR / "power_wide_abs_EO_EC.csv"

# ---------- helpers ----------
def cohen_dz(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]; b = b[mask]
    if a.size < 3:
        return np.nan
    d = a - b
    sd = float(np.std(d, ddof=1))
    return float(np.mean(d) / sd) if sd > 0 else np.nan

def shapiro_p(diff):
    diff = np.asarray(diff, float)
    diff = diff[np.isfinite(diff)]
    if diff.size < 3:
        return np.nan
    try:
        return float(stats.shapiro(diff).pvalue)
    except Exception:
        return np.nan

def _wilcoxon_p(a, b):
    try:
        d = a - b
        return float(stats.wilcoxon(d, zero_method='wilcox', alternative='two-sided').pvalue)
    except Exception:
        return np.nan

def choose_test(a, b, p_shapiro, alpha=ALPHA):
    """
    If p_shapiro >= alpha => differences are approximately normal => use paired t-test.
    Else => use Wilcoxon signed-rank test.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.isnan(p_shapiro):
        pval = _wilcoxon_p(a, b)
        return "wilcoxon", pval
    if p_shapiro >= alpha:
        try:
            t_stat, pval = stats.ttest_rel(a, b, nan_policy="omit")
            return "t_paired", float(pval)
        except Exception:
            return "t_paired", np.nan
    else:
        pval = _wilcoxon_p(a, b)
        return "wilcoxon", pval

def p_to_stars(p):
    try:
        if p is None or np.isnan(p): return ''
        if p < 1e-4: return '****'
        if p < 1e-3: return '***'
        if p < 1e-2: return '**'
        if p < 5e-2: return '*'
        return 'ns'
    except Exception:
        return ''

def discover_roi_band_columns(df, bands=None, regions=None):
    out = []
    pat = re.compile(r"^(.+?)__([A-Za-z]+)$")
    for c in df.columns:
        m = pat.match(c)
        if m:
            reg, band = m.group(1), m.group(2)
            if (regions is None or reg in regions) and (bands is None or band in bands):
                out.append((c, reg, band))
    return out

def normalize_base_columns(df):
    if "session" in df.columns:
        df["session"] = (df["session"].astype(str).str.upper()
                         .replace({"POS": "POST"}))
    if "visual_state" in df.columns:
        df["visual_state"] = df["visual_state"].astype(str).str.upper()
    df["subject"] = df["subject"].astype(str).str.zfill(2)
    if "group" not in df.columns:
        df["group"] = "ALL"
    return df

def longify(df, bands=None, regions=None):
    metrics = discover_roi_band_columns(df, bands=bands, regions=regions)
    if not metrics:
        return None
    value_cols = [m[0] for m in metrics]
    idv = ["subject","group","session","visual_state"]
    if "base_file" in df.columns:
        idv.append("base_file")
    long = df.melt(id_vars=idv, value_vars=value_cols,
                   var_name="roi_band", value_name="value")
    rb = long["roi_band"].str.split("__", n=1, expand=True)
    long["region"] = rb[0]; long["band"] = rb[1]
    return long

def paired_summary(df_block, pair_col, level_A, level_B):
    piv = df_block.pivot_table(index="subject", columns=pair_col,
                               values="value", aggfunc="mean")
    if level_A not in piv.columns or level_B not in piv.columns:
        return None
    sub = piv.dropna(subset=[level_A, level_B], how="any")
    if sub.empty:
        return None

    A = sub[level_A].values
    B = sub[level_B].values

    mean_A = float(np.mean(A)) if A.size else np.nan
    mean_B = float(np.mean(B)) if B.size else np.nan
    delta  = float(mean_A - mean_B) if np.isfinite(mean_A) and np.isfinite(mean_B) else np.nan

    # Normality of differences
    d = A - B
    p_norm = shapiro_p(d)

    # Automatically chosen test
    test_used, p_val = choose_test(A, B, p_norm, alpha=ALPHA)

    # Effect size (Cohen's dz)
    dz = cohen_dz(A, B)

    return dict(
        mean_A=mean_A,
        mean_B=mean_B,
        delta=delta,
        normality_test="Shapiro-Wilk",
        normality_p=p_norm,
        test_used=test_used,
        p_value=p_val,
        sig=p_to_stars(p_val),
        d=dz
    )

# ---------- execution ----------
def run_for_table(path_csv: Path, tag: str):
    if not path_csv.exists():
        print(f"[{tag}] file not found: {path_csv.name} — skipping.")
        return
    df = pd.read_csv(path_csv)
    df = normalize_base_columns(df)
    long = longify(df)
    if long is None:
        print(f"[{tag}] no ROI__Band columns found in {path_csv.name}.")
        return

    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # EC vs EO (fix session)
    rows = []
    for (g, sess, reg, band), d in long.groupby(["group","session","region","band"], dropna=False):
        res = paired_summary(d, pair_col="visual_state", level_A="EC", level_B="EO")
        if res is None:
            continue
        row = dict(group=g, session=sess, region=reg, band=band, compare="EC-EO", **res)
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        num_cols = [c for c in out.columns if c not in {"group","session","region","band","compare","normality_test","test_used","sig"}]
        out[num_cols] = out[num_cols].apply(pd.to_numeric, errors="coerce").round(15)
        p_out = out_dir / f"stats_{tag}_EOvsEC.csv"
        out.to_csv(p_out, index=False)
        print(f"[{tag}] ✅ EC vs EO saved to: {p_out}")
    else:
        print(f"[{tag}] ⚠️ EC vs EO: no results.")

    # POST vs PRE (fix visual_state)
    rows = []
    for (g, state, reg, band), d in long.groupby(["group","visual_state","region","band"], dropna=False):
        res = paired_summary(d, pair_col="session", level_A="POST", level_B="PRE")
        if res is None:
            continue
        row = dict(group=g, visual_state=state, region=reg, band=band, compare="POST-PRE", **res)
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        num_cols = [c for c in out.columns if c not in {"group","visual_state","region","band","compare","normality_test","test_used","sig"}]
        out[num_cols] = out[num_cols].apply(pd.to_numeric, errors="coerce").round(15)
        p_out = out_dir / f"stats_{tag}_POSTvsPRE.csv"
        out.to_csv(p_out, index=False)
        print(f"[{tag}] ✅ POST vs PRE saved to: {p_out}")
    else:
        print(f"[{tag}] ⚠️ POST vs PRE: no results.")

# Run (REL and ABS)
run_for_table(REL_PATH, tag="rel")
run_for_table(ABS_PATH, tag="abs")
print("\n🎉 Minimal statistics completed for REL and ABS.")