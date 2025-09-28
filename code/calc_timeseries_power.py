"""Sliding-window ROI×Band power time series (ABS/REL) for EO/EC and PRE/POST."""

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import mne

import config

# ---- paths / params ----------------------------------
PROCESSED_DIR = config.PROCESSED_DIR
OUT_DIR       = config.TS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

BANDS          = config.BANDS
REGIONS        = config.REGIONS
BANDS_ORDER    = config.BANDS_ORDER
ROIS_ORDER     = config.ROIS_ORDER
SESS_ORDER     = config.SESS_ORDER
VS_ORDER       = config.VS_ORDER

PSD_FMIN       = config.PSD_FMIN
PSD_FMAX       = config.PSD_FMAX
WELCH_SEG_SEC  = config.WELCH_SEG_SEC  # used to set Welch params for PSD

TS_WIN_SEC     = config.TS_WIN_SEC
TS_STEP_SEC    = config.TS_STEP_SEC

GROUP_ACTIVE   = set(config.GROUP_ACTIVE)
GROUP_PASSIVE  = set(config.GROUP_PASSIVE)
GROUP_CONTROL  = set(config.GROUP_CONTROL)
ACTIVE_CHANNELS = set(config.ACTIVE_CHANNELS)

# ---- helpers -----------------------------------------------------------------
# Matches: sub-01[_ses-pre]_task-rest[_run-01]_desc-preproc_(clean_raw|EO_clean_raw|EC_clean_raw|blocks_manifest).(fif|csv)
_BIDS_DERIV_RE = re.compile(
    r"(?P<sub>sub-\d+)"
    r"(?:_(?P<ses>ses-[a-z0-9]+))?"
    r"_task-(?P<task>[^_]+)"
    r"(?:_run-(?P<run>[^_]+))?"
    r"_desc-(?P<desc>[^_]+)"
    r"_(?P<what>clean_raw|EO_clean_raw|EC_clean_raw|blocks_manifest)\.(?P<ext>fif|csv)$",
    re.IGNORECASE,
)

def session_label_prepost(ses: str | None) -> str:
    if not ses:
        return ""
    s = ses.lower()
    if s.endswith("pre"):
        return "PRE"
    if s.endswith("post"):
        return "POST"
    return s.upper()

def subject_two_digit(sub_tag: str) -> str:
    m = re.search(r"sub-(\d+)", sub_tag, re.IGNORECASE)
    return m.group(1).zfill(2) if m else sub_tag

def parse_bids_entities_from_filename(path: Path):
    m = _BIDS_DERIV_RE.search(path.name)
    if not m:
        return None
    d = m.groupdict()
    parts = [d["sub"]]
    if d.get("ses"):
        parts.append(d["ses"])
    parts.append(f"task-{d['task']}")
    if d.get("run"):
        parts.append(f"run-{d['run']}")
    base = "_".join(parts) + f"_desc-{d['desc']}"
    return {
        "sub": d["sub"],
        "ses": d.get("ses"),
        "task": d["task"],
        "run": d.get("run"),
        "desc": d["desc"],
        "what": d["what"],
        "ext": d["ext"],
        "base": base,
    }

def infer_group_from_subject(subj_num: str) -> str:
    s = str(subj_num).zfill(2)
    if s in GROUP_ACTIVE:
        return "Active"
    if s in GROUP_PASSIVE:
        return "Passive"
    if s in GROUP_CONTROL:
        return "Control"
    return "Unknown"

def pick_roi_channel_indices(raw: mne.io.BaseRaw, ch_list):
    inc = [ch for ch in ch_list if (not ACTIVE_CHANNELS or ch in ACTIVE_CHANNELS)]
    return mne.pick_channels(raw.ch_names, include=inc, ordered=False)

def load_state_raw(dir_eeg: Path, base: str, state: str):
    direct = dir_eeg / f"{base}_{state}_clean_raw.fif"
    if direct.exists():
        try:
            return mne.io.read_raw_fif(direct, preload=True, verbose="ERROR")
        except Exception as e:
            print(f"  WARNING: cannot open {direct.name}: {e}")

    concat = dir_eeg / f"{base}_clean_raw.fif"
    if concat.exists():
        try:
            raw_all = mne.io.read_raw_fif(concat, preload=True, verbose="ERROR")
            if not raw_all.annotations or len(raw_all.annotations) == 0:
                return None
            sf = float(raw_all.info.get("sfreq", 250.0))
            eps = 1.0 / sf
            pieces = []
            for desc, onset, dur in zip(
                raw_all.annotations.description,
                raw_all.annotations.onset,
                raw_all.annotations.duration,
            ):
                if desc == f"visual_state:{state}" and dur > 0:
                    t0 = float(onset)
                    t1 = min(float(onset + dur), float(raw_all.times[-1]) - eps)
                    if t1 > t0:
                        pieces.append(raw_all.copy().crop(tmin=t0, tmax=t1))
            if not pieces:
                return None
            return (
                mne.concatenate_raws(pieces, verbose="ERROR")
                if len(pieces) > 1
                else pieces[0]
            )
        except Exception as e:
            print(f"  WARNING: cannot crop {concat.name}: {e}")
    return None

# ---- collect bases from concatenated derivatives -----------------------------
concat_files = sorted(PROCESSED_DIR.rglob("*_desc-preproc_clean_raw.fif"))

bases = []   # {dir_eeg, base, sub_num, sess_pp}
seen = set()
for f in concat_files:
    ent = parse_bids_entities_from_filename(f)
    if not ent:
        continue
    dir_eeg = f.parent
    sub_num = subject_two_digit(ent["sub"])
    sess_pp = session_label_prepost(ent.get("ses"))
    base    = ent["base"]
    key     = (dir_eeg, base)
    if key in seen:
        continue
    seen.add(key)
    bases.append({"dir_eeg": dir_eeg, "base": base, "sub_num": sub_num, "sess_pp": sess_pp})

if not bases:
    print("No *_desc-preproc_clean_raw.fif found under results/processed/**/eeg/")
    raise SystemExit

# Deterministic order: by subject, then session order
_sess_rank = {s: i for i, s in enumerate(SESS_ORDER)}
bases.sort(key=lambda x: (x["sub_num"], _sess_rank.get(x["sess_pp"], 99)))

# ---- main --------------------------------------------------------------------
rows = []

for entry in bases:
    dir_eeg, base = entry["dir_eeg"], entry["base"]
    sub, sess = entry["sub_num"], entry["sess_pp"]
    grp = infer_group_from_subject(sub)

    for state in VS_ORDER:
        raw = load_state_raw(dir_eeg, base, state)
        if raw is None:
            print(f"Missing {state} for {base}; skipping.")
            continue

        raw.pick("eeg")
        sf = float(raw.info["sfreq"])
        fmax_eff = min(float(PSD_FMAX), sf / 2.0 - 1.0)

        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=float(TS_WIN_SEC),
            overlap=max(0.0, float(TS_WIN_SEC) - float(TS_STEP_SEC)),
            preload=True,
            verbose="ERROR",
        )
        if len(epochs) == 0:
            print(f"Not enough windows for {base} {state}; skipping.")
            continue

        n_per_seg = max(16, int(round(float(TS_WIN_SEC) * sf)))
        spec = epochs.compute_psd(
            method="welch",
            fmin=float(PSD_FMIN),
            fmax=fmax_eff,
            n_fft=n_per_seg,
            n_per_seg=n_per_seg,
            n_overlap=int(round((float(TS_WIN_SEC) - float(TS_STEP_SEC)) * sf)),
            verbose="ERROR",
        )
        psds, freqs = spec.get_data(return_freqs=True)  # [n_epochs, n_ch, n_freqs]

        total_mask = (freqs >= float(PSD_FMIN)) & (freqs <= fmax_eff)
        band_masks = {b: ((freqs >= lo) & (freqs <= hi)) for b, (lo, hi) in BANDS.items()}

        starts  = epochs.events[:, 0] / sf
        centers = starts + float(TS_WIN_SEC) / 2.0

        for roi in ROIS_ORDER:
            chs = REGIONS[roi]
            picks = pick_roi_channel_indices(raw, chs)
            if picks.size == 0:
                continue

            total = psds[:, picks][:, :, total_mask].mean(axis=2)
            total_mean = np.where(np.isfinite(total), total, np.nan).mean(axis=1)

            for band in BANDS_ORDER:
                mask = band_masks[band]
                band_abs = psds[:, picks][:, :, mask].mean(axis=2)
                band_abs_mean = np.nanmean(band_abs, axis=1)

                with np.errstate(divide="ignore", invalid="ignore"):
                    band_rel = np.divide(band_abs_mean, total_mean)

                for t, p_abs, p_rel in zip(centers, band_abs_mean, band_rel):
                    rows.append({
                        "subject": sub,
                        "group": grp,
                        "session": sess,
                        "visual_state": state,
                        "region": roi,
                        "band": band,
                        "t_sec": float(t),
                        "power_abs": float(p_abs) if np.isfinite(p_abs) else np.nan,
                        "power_rel": float(p_rel) if np.isfinite(p_rel) else np.nan,
                        "source_file": f"{base}_{state}_clean_raw",
                    })

# ---- save --------------------------------------------------------------------
df = pd.DataFrame(rows).sort_values(
    ["band", "region", "group", "subject", "visual_state", "session", "t_sec"]
)
out_csv = OUT_DIR / "ts_power_long.csv"
df.to_csv(out_csv, index=False)

params = dict(
    BANDS=BANDS,
    REGIONS=REGIONS,
    TS_WIN_SEC=float(TS_WIN_SEC),
    TS_STEP_SEC=float(TS_STEP_SEC),
    PSD_FMIN=float(PSD_FMIN),
    PSD_FMAX=float(PSD_FMAX),
    WELCH_SEG_SEC=float(WELCH_SEG_SEC),
)
(OUT_DIR / "readme_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

print(f"\nTime series saved to: {out_csv}")