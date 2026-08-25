#!/usr/bin/env python3
"""
entrypoint.py -- BraTS 2026 Task 1 test-phase container entrypoint. TWO-PASS design.

Contract (Synapse wiki, syn74274097, Submission Instructions -> Containerized Algorithm
+ Execution Pipeline, both fetched/re-verified 2026-07-30, quoted in CONTAINER.md):
`/input` is mounted read-only, ALL case folders present simultaneously in one invocation
("Your model must iterate through each folder in /input..." -- confirmed verbatim,
re-fetched live before this design was implemented, not assumed from memory). Write one
flat <case_id>.nii.gz per case directly to /output, no subfolders. Zero network access.
NVIDIA A10G (24GB VRAM), --memory=48G --shm-size=16G, 12 hour total inference budget.

WHY TWO-PASS: the site-proxy `is_ucsd()` gates the RC hard-zero, and its ID-range
fallback logic was tuned on the training corpus's patient-ID numbering, never confirmed
against real test-phase IDs (configs/FROZEN_TESTPHASE.yaml Part A.2). Failure is silent
in both directions (proxy flags nobody -> RC F1=0 everywhere; proxy flags everybody ->
reverts to a measured -0.0853 RC cost) and is NOT observable from a single case, but IS
observable in aggregate: Task 1's own Data wiki page states 213 of 303 test cases are
UCSD (70.3%) -- a real, sourced expectation, not a guess.

Pass 1 (cheap, ID-string-only): run is_ucsd() on every discovered case ID, count how many
are flagged. If that count falls inside a plausible band around 213, the ID logic
transferred -- use it. If not, the ID logic did not transfer; fall back to an image-only
classifier (trained with ZERO patient-ID/institution dependency, AUC 0.869 on training-
corpus CV) and RANK-select the top 213 cases by score, not by a probability threshold --
rank selection is immune to calibration drift on a shifted test distribution, which is
the specific weakness of a 0.869-AUC classifier that a fixed threshold would not survive.

The per-case classifier features require the 5-fold ensembled probability maps, which do
not exist until after inference runs -- so inference always runs for every case
regardless of which path is chosen (no wasted GPU time either way), and the fallback
classifier is only ever SCORED (not what decides whether to invoke it) after inference
completes. The decision of WHICH path to use is made up front from case IDs alone, before
any GPU work, since that's all it needs.

Cases are pre-validated (all 4 modalities present, readable, geometrically consistent)
BEFORE being handed to segmenter.py, so one malformed case cannot crash the whole
5-fold, ~303-case run.

CHUNKED, STREAMING assembly (added after a reported memory/disk failure on larger case
counts): cases are processed CHUNK_SIZE at a time, not all ~303 at once. Each fold's
raw per-case probability output only ever exists on disk for one chunk's worth of cases
at a time (deleted once that chunk's per-case means are cached), and only a small
per-case scalar feature set -- never a full-resolution probability volume -- is held in
memory for the whole cohort. A previous version built one dict holding every case's
full 4-channel probability volume simultaneously for the entire run, which scaled
linearly with case count and was invisible on a 20-case smoke test but real at
179-303 cases. The math is unchanged (same files, same averaging, same thresholds) --
only when data is computed, cached, and freed changed.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime

import joblib
import nibabel as nib
import numpy as np
from scipy.ndimage import label as cc_label

ALGO_ROOT = "/opt/algorithm"
sys.path.insert(0, os.path.join(ALGO_ROOT, "scripts"))

from regions_to_labelmap import combine_region_masks, assert_channel_order  # noqa: E402
from postproc_filters import wt_no_et_filter, rim_test_filter, hysteresis_threshold  # noqa: E402
from score_case import labeled_components_mm3  # noqa: E402

FOLDS = [0, 1, 2, 3, 4]
MODALITY_SUFFIXES = {"t1n": "-t1n.nii.gz", "t1c": "-t1c.nii.gz", "t2w": "-t2w.nii.gz", "t2f": "-t2f.nii.gz"}
# T2W-optional risk: the challenge Data wiki page states T2W became non-mandatory in
# BraTS-METS starting 2025. This model was trained assuming all 4 channels are always
# present. A genuinely T2W-less test case is treated as a missing-modality error (clear
# error, not a silent partial result) rather than guessed -- see Open Questions.

ET_THRESHOLD, TC_THRESHOLD, WT_THRESHOLD = 0.25, 0.45, 0.5
MIN_VOLUME_MM3 = 15.0
RC_FLOOD, RC_SEED = 0.5, 0.97
WT_NO_ET_MIN_ET_VOXELS = 20
RC_CONFLICT = "higher_prob"
REGION_LABELS = {"wt": (1, 2, 3), "tc": (1, 3), "et": (3,), "rc": (4,)}

UCSD_IDS_PATH = os.path.join(ALGO_ROOT, "configs", "ucsd_patient_ids.json")
IMAGE_GATE_PATH = os.path.join(ALGO_ROOT, "configs", "rc_image_only_gate.joblib")

# Expected UCSD count on the real 303-case test set (Task 1 Data wiki page, verbatim
# table row: UCSD | 646 | 91 | 213 -- re-fetched live and confirmed Task-1-specific
# before this was written, not carried over from memory). Band is +-40 (170-250) around
# 213 as pre-registered by the task -- wide enough to absorb ordinary case-count
# variation (this container may be smoke-tested on subsets, not just the full 303) while
# still catching "flags ~0" or "flags ~all" catastrophic failure outright.
EXPECTED_UCSD_TEST = 213
BAND_LOW, BAND_HIGH = 170, 250

FEATURE_COLS = ["rc_max_prob", "rc_p99_prob", "rc_p999_prob", "rc_sum_mass",
                "wt_volume_mm3", "et_volume_mm3", "tc_volume_mm3",
                "rc_n_components_0.3", "rc_largest_component_mm3_0.3",
                "rc_n_components_0.5", "rc_largest_component_mm3_0.5",
                "rc_n_components_0.7", "rc_largest_component_mm3_0.7",
                "rc_n_components_0.9", "rc_largest_component_mm3_0.9"]
COMPONENT_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]


def load_ucsd_gate():
    """Baked replacement for treatment_status.is_ucsd() -- identical logic and data to
    the real function, without its 34GB raw-directory dependency. This IS the primary
    path this container attempts first; whether it transfers to test-phase IDs is
    exactly what Pass 1's count check is for."""
    data = json.load(open(UCSD_IDS_PATH))
    patients = set(data["ucsd_patients"])
    id_min, id_max = data["ucsd_id_min"], data["ucsd_id_max"]

    def is_ucsd(case_id):
        m = re.search(r"BraTS-MET-(\d+)-(\d+)", case_id)
        if not m:
            return False
        pid = m.group(1)
        if pid in patients:
            return True
        return id_min <= pid <= id_max

    return is_ucsd


def extract_gate_features(wt_p, tc_p, et_p, rc_mean, brain, voxvol):
    """Identical feature definition to scripts/phase5_rc_case_gate.py's _extract_one,
    computed here from the 5-fold ENSEMBLE MEAN (deployment-time distribution) rather
    than per-fold OOF probabilities (the distribution the classifier was trained on).
    This is a real, acknowledged train/deploy shift -- the same class of shift explored
    at length elsewhere in this project for the RC seed itself. Accepted here because:
    (a) this is a FALLBACK path only, active exclusively when the primary mechanism has
    already been detected as failing, so an imperfectly calibrated fallback beats the
    catastrophe it replaces; (b) rank-based (not threshold-based) selection is specifically
    chosen to be robust to exactly this kind of calibration drift."""
    rc_in_brain = rc_mean[brain]
    feats = dict(
        rc_max_prob=float(rc_mean.max()),
        rc_p99_prob=float(np.percentile(rc_in_brain, 99)) if rc_in_brain.size else 0.0,
        rc_p999_prob=float(np.percentile(rc_in_brain, 99.9)) if rc_in_brain.size else 0.0,
        rc_sum_mass=float(rc_mean[rc_mean >= 0.1].sum()),
        wt_volume_mm3=float((wt_p >= 0.5).sum() * voxvol),
        et_volume_mm3=float((et_p >= 0.5).sum() * voxvol),
        tc_volume_mm3=float((tc_p >= 0.5).sum() * voxvol),
    )
    for th in COMPONENT_THRESHOLDS:
        mask = rc_mean >= th
        if mask.any():
            labeled, n = cc_label(mask)
            sizes = np.bincount(labeled.ravel())[1:] * voxvol
            feats[f"rc_n_components_{th}"] = int(n)
            feats[f"rc_largest_component_mm3_{th}"] = float(sizes.max())
        else:
            feats[f"rc_n_components_{th}"] = 0
            feats[f"rc_largest_component_mm3_{th}"] = 0.0
    return feats


def discover_cases(input_dir):
    return sorted(d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d)))


def find_modality(case_dir, suffix):
    for f in os.listdir(case_dir):
        if f.endswith(suffix):
            return os.path.join(case_dir, f)
    return None


def validate_case(case_id, case_dir):
    """Returns (paths_dict_or_None, error_or_None, t1c_path_or_None). The third value is
    returned EVEN ON FAILURE whenever t1c itself is present and readable, specifically so
    a case that fails validation for some other reason (missing T2W, etc.) can still get
    a well-formed all-background fallback label written in its correct native geometry,
    per Part 5's hardening: a missing OUTPUT FILE risks invalidating the whole submission
    (GOTCHAS.md: package_submission.py does not assert output file count, and a real
    partial-write crash happened once in this project); a wrong-but-present, honestly
    near-zero-scoring prediction does not."""
    paths = {}
    missing = []
    for mod, suffix in MODALITY_SUFFIXES.items():
        p = find_modality(case_dir, suffix)
        if p is None:
            missing.append(mod)
        else:
            paths[mod] = p

    t1c_path = paths.get("t1c")
    if missing:
        return None, f"missing modalities: {missing}", t1c_path

    try:
        imgs = {mod: nib.load(p) for mod, p in paths.items()}
    except Exception as e:
        return None, f"unreadable NIfTI: {e}", t1c_path

    ref = imgs["t1c"]
    for mod, img in imgs.items():
        if img.shape != ref.shape:
            return None, f"{mod} shape {img.shape} != t1c shape {ref.shape}", t1c_path
        if np.abs(np.asarray(img.affine) - np.asarray(ref.affine)).max() > 1e-2:
            return None, f"{mod} affine != t1c affine", t1c_path
    return paths, None, t1c_path


def write_fallback_label(case_id, t1c_path, output_dir, reason):
    """Part 5 hardening: on ANY failure where t1c's own geometry is at least readable,
    write a well-formed all-background label rather than no file at all. Logged loudly
    (case ID + reason) so the organizers' own stdout log shows exactly where and why this
    fired -- never a silent fallback."""
    print(f"[FALLBACK-EMPTY] {case_id}: writing all-background label. Reason: {reason}", flush=True)
    try:
        src_img = nib.load(t1c_path)
        empty = np.zeros(src_img.shape, dtype=np.uint8)
        out_img = nib.Nifti1Image(empty, src_img.affine)
        nib.save(out_img, os.path.join(output_dir, f"{case_id}.nii.gz"))
        return True
    except Exception as e:
        print(f"[FALLBACK-EMPTY-FAILED] {case_id}: could not even write an empty label: {e}", flush=True)
        return False


def build_datalist(valid_cases, tmp_path):
    entries = [{"image": [p["t1n"], p["t1c"], p["t2w"], p["t2f"]]} for p in valid_cases.values()]
    with open(tmp_path, "w") as f:
        json.dump({"testing": entries}, f)


def run_fold_inference(fold, datalist_path, out_dir, input_dir):
    bundle_root = os.path.join(ALGO_ROOT, "Model", f"segresnet_{fold}")
    config_file = os.path.join(bundle_root, "configs", "hyper_parameters.yaml")
    ckpt = os.path.join(bundle_root, "model", "model.pt")
    cmd = [
        "python3", os.path.join(bundle_root, "scripts", "segmenter.py"),
        f"--config_file={config_file}",
        f"--pretrained_ckpt_name={ckpt}",
        "--infer#enabled=True",
        "--infer#save_mask=True",
        f"--infer#output_path={out_dir}",
        "--infer#data_list_key=testing",
        "--save_mask_mode=prob",
        f"--data_list_file_path={datalist_path}",
        f"--data_file_base_dir={input_dir}",
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "MLFLOW_ALLOW_FILE_STORE": "true"}
    # Part 4/5 hardening, found by direct testing on this multi-GPU (4x H200) dev host:
    # segmenter.py's own __main__ does `torch.multiprocessing.spawn(..., nprocs=torch.cuda
    # .device_count())` whenever more than one GPU is visible -- a silent DDP-shard-across-
    # GPUs path with no relation to the 1-GPU A10G target. The real eval hardware exposes
    # exactly one GPU (Synapse wiki Compute Constraints: "NVIDIA A10G GPU", singular) so
    # this never triggers there -- defaulting it here is a zero-risk safety net, not a
    # behavior change for any already-verified run: every byte-diff/timing measurement in
    # this project already set CUDA_VISIBLE_DEVICES=0 externally, which setdefault() below
    # leaves untouched.
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    print(f"[fold {fold}] running inference on {out_dir} ...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(result.stdout[-4000:], flush=True)
        print(result.stderr[-4000:], flush=True)
        raise RuntimeError(f"fold {fold} inference failed, see output above")
    print(f"[fold {fold}] done.", flush=True)


def find_fold_output(out_dir):
    result = {}
    for dirpath, _, filenames in os.walk(out_dir):
        files = [f for f in filenames if f.endswith(".nii.gz") or f.endswith(".nii")]
        if not files:
            continue
        case_id = os.path.basename(dirpath)
        if len(files) != 1:
            continue
        result[case_id] = os.path.join(dirpath, files[0])
    return result


def filter_components(mask, spacing, min_volume_mm3):
    if min_volume_mm3 <= 0:
        return mask
    labeled, volumes = labeled_components_mm3(mask, spacing)
    keep_ids = {i for i, v in volumes.items() if v >= min_volume_mm3}
    if not keep_ids:
        return np.zeros_like(mask)
    return np.isin(labeled, list(keep_ids))


def region_mask(labelmap, labels):
    m = np.zeros(labelmap.shape, dtype=bool)
    for l in labels:
        m |= labelmap == l
    return m


def refloor_post_combination(labelmap, spacing):
    filtered = {}
    for region, labels in REGION_LABELS.items():
        mask = region_mask(labelmap, labels)
        filtered[region] = filter_components(mask, spacing, MIN_VOLUME_MM3) if mask.any() else mask
    wt_f, tc_f, et_f, rc_f = filtered["wt"], filtered["tc"], filtered["et"], filtered["rc"]
    fixed = np.zeros(labelmap.shape, dtype=np.uint8)
    fixed[wt_f & ~tc_f] = 2
    fixed[tc_f & ~et_f] = 1
    fixed[et_f] = 3
    fixed[rc_f] = 4
    fixed[labelmap == 0] = 0
    return fixed


def build_prediction(wt_p, tc_p, et_p, rc_p, spacing, brain_mask, case_id, ucsd_assignment):
    wt = wt_p >= WT_THRESHOLD
    tc = tc_p >= TC_THRESHOLD
    et = et_p >= ET_THRESHOLD
    if not ucsd_assignment[case_id]:
        rc = np.zeros_like(rc_p, dtype=bool)
    else:
        rc = hysteresis_threshold(rc_p, flood_threshold=RC_FLOOD, seed_threshold=RC_SEED)
    wt = wt_no_et_filter(wt, et, spacing, min_et_voxels=WT_NO_ET_MIN_ET_VOXELS, rc_mask=rc)
    rc = rim_test_filter(rc, et, spacing)
    wt = filter_components(wt, spacing, MIN_VOLUME_MM3)
    tc = filter_components(tc, spacing, MIN_VOLUME_MM3)
    et = filter_components(et, spacing, MIN_VOLUME_MM3)
    rc = filter_components(rc, spacing, MIN_VOLUME_MM3)
    labelmap = combine_region_masks(wt, tc, et, rc, wt_p=wt_p, rc_p=rc_p,
                                     rc_conflict=RC_CONFLICT, brain_mask=brain_mask)
    return refloor_post_combination(labelmap, spacing)


def decide_ucsd_assignment(case_ids, is_ucsd_fn, gate_features):
    """PASS 1's decision. `gate_features`: dict case_id -> small scalar-feature dict
    (rc_max_prob, rc_p99_prob, volumes, component counts -- a handful of floats, not
    a full-resolution volume), computed once per case during the chunked inference
    pass below and cheap to hold for the whole cohort. Only actually consulted if the
    id_range check fails and the image-fallback path is needed."""
    id_flags = {cid: is_ucsd_fn(cid) for cid in case_ids}
    n_id = sum(id_flags.values())
    print(f"[PASS 1] is_ucsd() flags {n_id}/{len(case_ids)} cases "
          f"(expected ~{EXPECTED_UCSD_TEST}, band [{BAND_LOW},{BAND_HIGH}])", flush=True)

    if BAND_LOW <= n_id <= BAND_HIGH:
        print(f"proxy_path=id_range, n={n_id}", flush=True)
        return id_flags, "id_range", n_id, None

    print(f"[PASS 1] is_ucsd() count {n_id} OUTSIDE the expected band -- ID logic did "
          f"NOT transfer. Falling back to the image-only classifier.", flush=True)
    gate = joblib.load(IMAGE_GATE_PATH)
    clf, cols = gate["model"], gate["feature_cols"]

    scores = {}
    for cid in case_ids:
        if cid not in gate_features:
            scores[cid] = -np.inf  # unusable cases can't be scored; never selected
            continue
        feats = gate_features[cid]
        x = np.array([[feats[c] for c in cols]])
        scores[cid] = float(clf.predict_proba(x)[0, 1])

    n_select = min(EXPECTED_UCSD_TEST, len(case_ids))
    ranked = sorted(case_ids, key=lambda c: scores[c], reverse=True)
    selected = set(ranked[:n_select])
    assignment = {cid: (cid in selected) for cid in case_ids}
    print(f"proxy_path=image_fallback, n_id={n_id}, n_selected={len(selected)} "
          f"(rank-based, top {n_select} by classifier score, cv_auc={gate['cv_auc']:.4f})",
          flush=True)
    return assignment, "image_fallback", n_id, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/input")
    parser.add_argument("--output", default="/output")
    args = parser.parse_args()

    is_ucsd_fn = load_ucsd_gate()
    failures = {}
    fallback_written = {}
    t1c_for_case = {}  # case_id -> t1c path, populated even for cases that fail validation

    def log_progress(msg):
        print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)

    case_ids = discover_cases(args.input)
    log_progress(f"Discovered {len(case_ids)} case folder(s) under {args.input}")

    valid_cases = {}
    for i, cid in enumerate(case_ids):
        case_dir = os.path.join(args.input, cid)
        paths, err, t1c_path = validate_case(cid, case_dir)
        if t1c_path:
            t1c_for_case[cid] = t1c_path
        if err:
            print(f"[SKIP] {cid}: {err}", flush=True)
            failures[cid] = err
            continue
        valid_cases[cid] = paths
        if (i + 1) % 20 == 0 or (i + 1) == len(case_ids):
            log_progress(f"  pre-validated {i+1}/{len(case_ids)}")
    log_progress(f"{len(valid_cases)}/{len(case_ids)} cases passed pre-validation.")

    # Part 5 hardening: any case that failed pre-validation but has a readable t1c still
    # gets a well-formed all-background label now, rather than at the very end -- so a
    # crash later in the run does not also lose these already-known outputs.
    if failures:
        os.makedirs(args.output, exist_ok=True)
        for cid, err in list(failures.items()):
            if cid in t1c_for_case:
                if write_fallback_label(cid, t1c_for_case[cid], args.output, err):
                    fallback_written[cid] = err

    if not valid_cases:
        log_progress("No valid cases to run inference on. Exiting.")
        sys.exit(1 if failures else 0)

    # Cases are processed in fixed-size CHUNKS rather than all ~303 at once. Each
    # fold's segmenter.py subprocess writes a raw probability volume PER CASE to disk
    # before anything reads it back -- with all cases handed to segmenter.py in one
    # call (the original design), all 5 folds' outputs for the ENTIRE cohort exist on
    # disk simultaneously at the moment the 5th fold finishes, before any cleanup can
    # happen. Chunking bounds that peak to CHUNK_SIZE cases' worth of raw files rather
    # than the whole cohort's, at the cost of relaunching segmenter.py (Python/torch/
    # MONAI process startup, not just checkpoint load) once per fold PER CHUNK instead
    # of once per fold total -- realistically tens of seconds per relaunch, not
    # negligible at 5 folds x ~16 chunks for the full 303-case test set, but still small
    # against the ~6.7h projected runtime. 20 matches the smoke-test scale already
    # verified end-to-end (SUBMIT.md Step 3), so no single chunk here processes more
    # cases at once than has already been tested; raise it (e.g. to 50) if measured
    # relaunch overhead on the real hardware argues for fewer, larger chunks.
    CHUNK_SIZE = 20

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = os.path.join(tmp, "mean_cache")
        os.makedirs(cache_dir, exist_ok=True)

        case_ids_list = list(valid_cases.keys())
        chunks = [case_ids_list[i:i + CHUNK_SIZE] for i in range(0, len(case_ids_list), CHUNK_SIZE)]
        log_progress(f"Processing {len(case_ids_list)} case(s) in {len(chunks)} "
                     f"chunk(s) of up to {CHUNK_SIZE}")

        # Per-case scalar features for the image-fallback gate (see extract_gate_features)
        # -- computed for every case as part of the same streaming pass that builds the
        # mean-probability cache below, since the mean is already in hand at that point.
        # Cheap to keep for the whole cohort (a dict of ~15 floats per case), unlike the
        # full-resolution volumes it's derived from.
        gate_features = {}
        n_processed = 0

        # sw_batch_size=1 and overlap=0.625 are segmenter.py's OWN hardcoded defaults
        # (confirmed by reading algorithm_templates/segresnet/scripts/segmenter.py, not
        # overridden here in either direction) -- overlap is NOT 0.5 or 0.75; changing it
        # would alter output and break byte-equivalence with 9774164, so it is confirmed
        # and left untouched. Each fold is a SEPARATE subprocess (segmenter.py invoked
        # fresh per fold, never imported in-process), so GPU memory for one fold's model
        # is fully released by the OS when that subprocess exits.
        for chunk_idx, chunk_ids in enumerate(chunks):
            chunk_tmp = os.path.join(tmp, f"chunk_{chunk_idx}")
            os.makedirs(chunk_tmp, exist_ok=True)
            datalist_path = os.path.join(chunk_tmp, "datalist.json")
            build_datalist({cid: valid_cases[cid] for cid in chunk_ids}, datalist_path)

            fold_outputs = {}
            chunk_fold_failed = False
            for fold in FOLDS:
                fold_out_dir = os.path.join(chunk_tmp, f"fold_{fold}_prob")
                try:
                    log_progress(f"chunk {chunk_idx+1}/{len(chunks)}, fold {fold}: "
                                 f"starting inference on {len(chunk_ids)} cases")
                    run_fold_inference(fold, datalist_path, fold_out_dir, args.input)
                    fold_outputs[fold] = find_fold_output(fold_out_dir)
                    log_progress(f"chunk {chunk_idx+1}/{len(chunks)}, fold {fold}: done, "
                                 f"{len(fold_outputs[fold])} case outputs found")
                except Exception as e:
                    # A whole-fold failure within a chunk is not recoverable per-case
                    # (every case in this chunk needs all 5 folds to ensemble correctly)
                    # -- but per Part 5, still write all-background fallbacks for this
                    # chunk's cases and move on to the NEXT chunk, rather than losing
                    # every already-completed chunk's cases over one chunk's failure.
                    log_progress(f"[FATAL] chunk {chunk_idx+1} fold {fold} inference "
                                 f"failed entirely: {e}")
                    traceback.print_exc()
                    os.makedirs(args.output, exist_ok=True)
                    for cid in chunk_ids:
                        if cid not in fallback_written and cid in t1c_for_case:
                            if write_fallback_label(cid, t1c_for_case[cid], args.output,
                                                     f"chunk fold {fold} inference failed entirely: {e}"):
                                fallback_written[cid] = str(e)
                                failures[cid] = str(e)
                    chunk_fold_failed = True
                    break
            if chunk_fold_failed:
                shutil.rmtree(chunk_tmp, ignore_errors=True)
                continue

            for cid in chunk_ids:
                try:
                    paths = valid_cases[cid]
                    src_img = nib.load(paths["t1c"])
                    probs_per_fold = []
                    for fold in FOLDS:
                        if cid not in fold_outputs[fold]:
                            raise RuntimeError(f"fold {fold} produced no output for this case")
                        img = nib.load(fold_outputs[fold][cid])
                        arr = np.asarray(img.dataobj)
                        if arr.shape[-1] == 4:
                            arr = np.moveaxis(arr, -1, 0)
                        if arr.shape[1:] != src_img.shape:
                            raise RuntimeError(
                                f"fold {fold} output shape {arr.shape[1:]} != source {src_img.shape}")
                        probs_per_fold.append(arr.astype(np.float32))
                    mean_prob = np.mean(probs_per_fold, axis=0)
                    spacing = src_img.header.get_zooms()[:3]
                    brain_mask = np.asarray(src_img.dataobj) != 0

                    # Cache only the COMBINED mean (one 4-channel array) rather than
                    # leaving all 5 raw per-fold volumes on disk -- this is the actual
                    # disk-usage fix, on top of the chunking above.
                    np.savez(os.path.join(cache_dir, f"{cid}.npz"),
                             mean_prob=mean_prob, brain_mask=brain_mask,
                             spacing=np.array(spacing), affine=np.asarray(src_img.affine))

                    voxvol = float(np.prod(spacing))
                    gate_features[cid] = extract_gate_features(
                        mean_prob[0], mean_prob[1], mean_prob[2], mean_prob[3], brain_mask, voxvol)
                except Exception as e:
                    print(f"[SKIP-INFER] {cid}: {e}", flush=True)
                    traceback.print_exc()
                    failures[cid] = str(e)
                    if cid in t1c_for_case:
                        os.makedirs(args.output, exist_ok=True)
                        if write_fallback_label(cid, t1c_for_case[cid], args.output, str(e)):
                            fallback_written[cid] = str(e)
                    continue
                n_processed += 1
                if n_processed % 20 == 0 or n_processed == len(case_ids_list):
                    log_progress(f"  assembled + cached: {n_processed}/{len(case_ids_list)}")

            # Free this chunk's raw per-fold probability files now that each case's
            # mean has been cached -- at most one chunk's worth of raw 5-fold files
            # exists on disk at a time, not the whole cohort's.
            shutil.rmtree(chunk_tmp, ignore_errors=True)

        # PASS 1's decision. The id_range path needs no probability data at all (pure
        # case-ID string check); the image-fallback path (taken only if that check
        # fails) uses the small scalar features already computed above -- no full
        # volumes are touched again for this decision.
        ucsd_assignment, proxy_path, n_id, fallback_scores = decide_ucsd_assignment(
            list(valid_cases.keys()), is_ucsd_fn, gate_features)

        # Per the wiki's own strict "/output as a flat structure, do NOT create
        # sub-folders / extra files -- doing so will invalidate your final submission"
        # language: the decision log is NOT written into /output. Logged to stdout only.
        print(f"=== PROXY DECISION: path={proxy_path} n_id_flagged={n_id} "
              f"n_cases={len(valid_cases)} ===", flush=True)

        # PASS 2: build and write each case's final prediction one case at a time,
        # reading back only the small cached mean (not the raw per-fold files, which
        # are already gone) -- and deleting each case's cache entry immediately after
        # use, so the cache's own footprint shrinks through PASS 2 rather than sitting
        # at full size until the container exits.
        os.makedirs(args.output, exist_ok=True)
        for i, cid in enumerate(valid_cases):
            cache_path = os.path.join(cache_dir, f"{cid}.npz")
            if not os.path.exists(cache_path):
                continue  # already handled (fallback written) above
            try:
                cached = np.load(cache_path)
                mean_prob = cached["mean_prob"]
                brain_mask = cached["brain_mask"]
                spacing = tuple(cached["spacing"])
                affine = cached["affine"]
                wt_p, tc_p, et_p, rc_p = mean_prob[0], mean_prob[1], mean_prob[2], mean_prob[3]

                labelmap = build_prediction(wt_p, tc_p, et_p, rc_p, spacing, brain_mask,
                                             cid, ucsd_assignment)
                # Trivially-empty (all-background) is a valid output, written like any other.

                out_img = nib.Nifti1Image(labelmap.astype(np.uint8), affine)
                out_path = os.path.join(args.output, f"{cid}.nii.gz")
                nib.save(out_img, out_path)

                # mean_prob.shape[1:] was already asserted equal to the source image's
                # own shape when the cache was built (chunk pass above), so this check
                # is equivalent to comparing against the true source geometry.
                assert out_img.shape == mean_prob.shape[1:], f"{cid}: shape mismatch after save"
                assert np.abs(np.asarray(out_img.affine) - affine).max() < 1e-3, \
                    f"{cid}: affine mismatch after save"
            except Exception as e:
                print(f"[SKIP-POST] {cid}: {e}", flush=True)
                traceback.print_exc()
                failures[cid] = str(e)
                if cid in t1c_for_case:
                    if write_fallback_label(cid, t1c_for_case[cid], args.output, str(e)):
                        fallback_written[cid] = str(e)
                continue
            finally:
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
            if (i + 1) % 20 == 0 or (i + 1) == len(valid_cases):
                log_progress(f"  post-processed and written: {i+1}/{len(valid_cases)}")

    written = len([f for f in os.listdir(args.output) if f.endswith(".nii.gz")])
    log_progress(f"=== DONE: {written}/{len(case_ids)} output files written to {args.output} "
                 f"({len(fallback_written)} were all-background fallbacks) ===")
    if failures:
        print(f"=== {len(failures)} case(s) hit an error during the run: ===", flush=True)
        for cid, err in failures.items():
            fb = "fallback written" if cid in fallback_written else "NO OUTPUT WRITTEN"
            print(f"  {cid} [{fb}]: {err}", flush=True)
        if len(fallback_written) < len(failures):
            # At least one case has NO output file at all (t1c itself unreadable/missing)
            # -- this is the one situation Part 5 cannot fully harden against, since there
            # is no reference geometry to build even an empty label from. Non-zero exit
            # makes this visible; every other failure still produced a valid file.
            sys.exit(2)


if __name__ == "__main__":
    main()
