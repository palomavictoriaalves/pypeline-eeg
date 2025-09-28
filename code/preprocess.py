"""End-to-end EEG preprocessing (BIDS-aware)."""

from pathlib import Path
import re
import numpy as np
import mne
import mne_icalabel
import config

# Paths / params
DATA_DIR      = config.DATA_DIR
PROCESSED_DIR = config.PROCESSED_DIR
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FILTER_LOW        = config.FILTER_LOW
FILTER_HIGH       = config.FILTER_HIGH
NOTCH_HZ          = config.NOTCH_HZ
BLOCKS_WITH_STATE = config.BLOCKS_WITH_STATE

print(f"BIDS root       : {DATA_DIR}")
print(f"Derivatives dir : {PROCESSED_DIR}")

# --- helpers ------------------------------------------------------------------
def _normalize_label(lbl: str) -> str:
    l = (lbl or "").strip().lower()
    mapping = {
        "brain": "brain",
        "muscle": "muscle", "muscle artifact": "muscle",
        "eye": "eye", "eog": "eye", "eye blink": "eye",
        "heart": "heart", "heart beat": "heart", "ecg": "heart",
        "line_noise": "line_noise", "line noise": "line_noise",
        "ch_noise": "channel_noise", "channel noise": "channel_noise",
        "other": "other",
    }
    return mapping.get(l, l)

def _get_iclabel_labels_probs(raw_for_label, ica):
    try:
        ret = mne_icalabel.label_components(raw_for_label, ica, method="iclabel")
    except Exception:
        return None, None
    labels, probs = None, None
    if isinstance(ret, tuple) and len(ret) == 2:
        labels, probs = ret
    elif isinstance(ret, dict):
        labels = ret.get("labels") or ret.get("label") or ret.get("y_pred")
        probs  = ret.get("probabilities") or ret.get("probas") or ret.get("y_pred_proba")
    else:
        labels = ret
    if labels is not None and not isinstance(labels, (list, tuple)):
        labels = list(labels)
    return labels, probs

def _choose_exclusions_by_label_and_prob(labels, probs):
    if labels is None:
        return []
    norm = [_normalize_label(l) for l in labels]
    class_thr = {
        "eye": 0.60, "muscle": 0.70, "heart": 0.70, "line_noise": 0.70, "channel_noise": 0.70,
        "brain": 1.10, "other": 1.10,
    }
    if probs is None:
        return [k for k, lbl in enumerate(norm) if lbl in ("eye","muscle","heart","line_noise","channel_noise")]

    import numpy as _np
    if isinstance(probs, _np.ndarray) and probs.ndim == 2 and probs.shape[0] == len(norm):
        classes = ["brain","muscle","eye","heart","line_noise","channel_noise","other"]
        excl = []
        for k, lbl in enumerate(norm):
            thr = class_thr.get(lbl, 1.10)
            if thr < 1.0:
                try:
                    p = float(probs[k, classes.index(lbl)])
                except Exception:
                    p = float(_np.max(probs[k])) if probs.shape[1] > 0 else 0.0
                if p >= thr:
                    excl.append(k)
        return excl

    if isinstance(probs, (list, tuple)) and len(probs) == len(norm) and isinstance(probs[0], dict):
        return [k for k, lbl in enumerate(norm)
                if class_thr.get(lbl, 1.10) < 1.0 and float(probs[k].get(lbl, 0.0)) >= class_thr.get(lbl, 1.10)]

    if isinstance(probs, _np.ndarray) and probs.ndim == 1 and probs.shape[0] == len(norm):
        return [k for k, lbl in enumerate(norm)
                if class_thr.get(lbl, 1.10) < 1.0 and float(probs[k]) >= class_thr.get(lbl, 1.10)]

    return [k for k, lbl in enumerate(norm) if lbl in ("eye","muscle","heart","line_noise","channel_noise")]

def _save_blocks_manifest(csv_path, manifest_rows):
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "visual_state", "block_idx", "onset_s", "end_s", "duration_s", "used"])
        for row in manifest_rows:
            w.writerow(row)
    print(f"Block manifest saved: {csv_path}")

# --- BIDS discovery/output helpers -------------------------------------------
files = sorted(p for p in DATA_DIR.rglob("*.vhdr") if p.parent.name == "eeg")
print(f"\nFound {len(files)} BrainVision header(s) under {DATA_DIR}/**/eeg/.")

BIDS_VHDR_RE = re.compile(
    r"(?P<sub>sub-[^_/]+)"
    r"(?:_(?P<ses>ses-[^_/]+))?"
    r"_task-(?P<task>[^_/]+)"
    r"(?:_run-(?P<run>[^_/]+))?"
    r"_eeg\.vhdr$",
    re.IGNORECASE,
)

def parse_bids_entities(path: Path):
    m = BIDS_VHDR_RE.search(path.name)
    if not m:
        return {"sub": "sub-UNK", "ses": None, "task": "rest", "run": None}
    d = m.groupdict()
    return {
        "sub": d.get("sub") or "sub-UNK",
        "ses": d.get("ses") or None,
        "task": d.get("task") or "rest",
        "run": d.get("run") or None,
    }

def out_paths_for(vhdr_path: Path, desc: str = "preproc"):
    ent = parse_bids_entities(vhdr_path)
    sub, ses, task, run = ent["sub"], ent["ses"], ent["task"], ent["run"]

    outdir = PROCESSED_DIR.joinpath(*([sub] + ([ses] if ses else []) + ["eeg"]))
    outdir.mkdir(parents=True, exist_ok=True)

    bits = [sub] + ([ses] if ses else []) + [f"task-{task}"] + ([f"run-{run}"] if run else [])
    base = "_".join(bits)

    return {
        "entities": ent,
        "base": base,
        "dir": outdir,
        "concat":   outdir / f"{base}_desc-{desc}_clean_raw.fif",
        "eo":       outdir / f"{base}_desc-{desc}_EO_clean_raw.fif",
        "ec":       outdir / f"{base}_desc-{desc}_EC_clean_raw.fif",
        "manifest": outdir / f"{base}_desc-{desc}_blocks_manifest.csv",
    }

# --- main ---------------------------------------------------------------------
for vhdr_path in files:
    name = vhdr_path.stem
    ent = parse_bids_entities(vhdr_path)
    ent_str = f"[{ent['sub']}{'/' + ent['ses'] if ent['ses'] else ''} | task-{ent['task']}{' | run-' + ent['run'] if ent['run'] else ''}]"
    print(f"\nProcessing: {name}  {ent_str}")

    try:
        # Import and montage
        raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose="WARNING")
        raw.set_montage("standard_1020")

        # Filters (Hz)
        raw.filter(l_freq=FILTER_LOW, h_freq=FILTER_HIGH, fir_design="firwin")
        try:
            raw.notch_filter(freqs=[NOTCH_HZ])
        except Exception as e:
            print(f"  > WARNING: notch {NOTCH_HZ} Hz not applied: {e}")

        # Reference
        raw.set_eeg_reference(ref_channels="average")

        # Segmentation (seconds)
        blocks, manifest = [], []
        sfreq = float(raw.info.get("sfreq", 250.0))
        tmax = float(raw.times[-1])
        eps = 1.0 / sfreq

        opaths = out_paths_for(vhdr_path, desc="preproc")
        for i, (state, (t0, t1)) in enumerate(BLOCKS_WITH_STATE, start=1):
            t1_safe = min(float(t1), tmax - eps)
            if float(t0) < t1_safe:
                seg = raw.copy().crop(tmin=float(t0), tmax=float(t1_safe))
                blocks.append((state, seg))
                manifest.append([opaths["base"], state, i, float(t0), float(t1_safe),
                                 round(float(t1_safe) - float(t0), 3), 1])
            else:
                print(f"  > WARNING: unusable block {i} {state} [{t0},{t1}] (file ends at {tmax:.3f}s)")
                manifest.append([opaths["base"], state, i, float(t0), float(t1), 0.0, 0])

        raw_concat = mne.concatenate_raws([seg for _, seg in blocks], verbose="WARNING") if blocks else raw.copy()

        # ICA: fit on 1–40 Hz copy, extended Infomax, deterministic seed
        ica = mne.preprocessing.ICA(n_components=0.99, method="infomax",
                                    fit_params=dict(extended=True), random_state=97)
        raw_ica = raw_concat.copy().filter(l_freq=1.0, h_freq=40.0, fir_design="firwin")
        ica.fit(raw_ica)

        # ICLabel-based exclusion
        exclude_idx = []
        sf_full = float(raw.info.get("sfreq", 250.0))
        h_ic = float(min(100.0, max(30.0, sf_full / 2.0 - 1.0)))
        raw_ic = raw_concat.copy().filter(l_freq=1.0, h_freq=h_ic, fir_design="firwin")

        labels_ic, probs = _get_iclabel_labels_probs(raw_ic, ica)
        if labels_ic is not None:
            exclude_idx = _choose_exclusions_by_label_and_prob(labels_ic, probs)
            if exclude_idx:
                print(f"  > Removing {len(exclude_idx)} IC(s): {exclude_idx}")
            else:
                print("  > No ICs auto-excluded.")
        else:
            print("  > NOTE: could not obtain ICLabel labels; skipping auto-exclusion.")

        ica.exclude = exclude_idx
        if exclude_idx:
            ica.apply(raw_concat)

        # Annotate concatenated stream with visual_state labels
        annotations, cursor = [], 0.0
        for _, state, _, _, _, dur, used in manifest:
            if used and dur > 0:
                annotations.append((cursor, float(dur), f"visual_state:{state}"))
                cursor += float(dur)
        if annotations:
            raw_concat.set_annotations(mne.Annotations(
                onset=[o for o, _, _ in annotations],
                duration=[d for _, d, _ in annotations],
                description=[desc for *_, desc in annotations],
            ))

        # Save concatenated + per-state derivatives
        raw_concat.save(str(opaths["concat"]), overwrite=True)
        print(f"  Saved concatenated: {opaths['concat']}")

        def _extract_state(raw_in, state):
            if not raw_in.annotations:
                return None
            sf_local = float(raw_in.info.get("sfreq", 250.0))
            eps_local = 1.0 / sf_local
            pieces = []
            for desc, onset, duration in zip(raw_in.annotations.description,
                                             raw_in.annotations.onset,
                                             raw_in.annotations.duration):
                if desc == f"visual_state:{state}" and duration > 0:
                    t0 = float(onset)
                    t1 = min(float(onset + duration), float(raw_in.times[-1]) - eps_local)
                    if t1 > t0:
                        pieces.append(raw_in.copy().crop(tmin=t0, tmax=t1))
            if not pieces:
                return None
            return mne.concatenate_raws(pieces, verbose="WARNING") if len(pieces) > 1 else pieces[0]

        raw_EO = _extract_state(raw_concat, "EO")
        raw_EC = _extract_state(raw_concat, "EC")

        if raw_EO is not None:
            raw_EO.save(str(opaths["eo"]), overwrite=True)
            print(f"  Saved EO: {opaths['eo']}")
        else:
            print("  > NOTE: no EO segment to save.")

        if raw_EC is not None:
            raw_EC.save(str(opaths["ec"]), overwrite=True)
            print(f"  Saved EC: {opaths['ec']}")
        else:
            print("  > NOTE: no EC segment to save.")

        _save_blocks_manifest(opaths["manifest"], manifest)

    except Exception as e:
        print(f"  ERROR processing {name}: {e}")

print("\nPreprocessing finished.")