"""
brats_lesionwise.py -- lesion-wise BraTS-METS scoring, ported from the actively-maintained
official implementation (github.com/rachitsaluja/BraTS-2024-Metrics, metrics_MET.py --
last updated 2026-05-28, still the canonical BraTS lesion-wise scorer as of this writing;
its sibling repo BraTS-2023-Metrics was updated even more recently, 2026-06-25, confirming
this lineage is still live for the current challenge cycle). NOT a from-scratch metric:
the matching algorithm (dilate-by-1-then-combine-touching-components, any-overlap TP/FP/FN,
per-lesion Dice/HD95/NSD formulas) is a faithful line-for-line port of that script's
`save_tmp_files` + `get_LesionWiseScores` logic, rewritten to run entirely in memory
(no per-call disk round-trips) so `sweep_postproc.py` can score hundreds of configurations
without paying that I/O cost, and with ground-truth connected components cached and reused
across configurations (only the prediction side changes per sweep point).

Two things this port adds on top of upstream, both taken directly from the official
BraTS-METS 2026 challenge-page text (pasted into this session 2026-07-28), not guessed:
  1. The small/large split threshold is 27 mm^3 (upstream hardcodes a 2-unit threshold
     used generically across all BraTS challenges; the METS 2026 rules specify 27 mm^3
     specifically, and say it gates the SEGMENTATION metrics (DSC/NSD/HD95) only -- "We
     will apply the segmentation evaluation metrics only for lesions larger than 27 mm3."
     Detection (F1) is reported for all/large/small lesion subsets separately.
  2. "For the cases in which the algorithm fails to produce a result metric for a specific
     test case, there will be no penalties, i.e. the metric won't be set to its worst
     possible value" -- matches upstream's own convention (both-empty -> score 1.0, the
     best possible value, never dropped to 0), which this port preserves exactly for
     every metric (dice/nsd/hd95/precision/recall/f1).

Un-verified assumption, flagged not hidden: which NSD tolerance (0.5mm or 1.0mm, both of
which upstream computes) the real leaderboard reports as "lesionwise_nsd_mean" is not
stated anywhere we could find. Both are computed and returned; the CLI defaults to
reporting 1.0mm as the headline number (GOTCHAS M15's existing convention elsewhere in
this project), 0.5mm alongside it.

FP/large/small bucketing (reverse-engineered from the arithmetic of a real leaderboard
email pasted into this session, then cross-checked against upstream's own code shape --
NOT independently confirmed against source, flagged accordingly): "large_instance" and
"all_instance" report the IDENTICAL fp count (upstream's own Num_FP is never size-filtered
either -- only Num_TP/Num_FN get a small-lesion subtraction). "small_instance" is a
separately-restricted computation: only lesions (GT side for tp/fn, predicted side for fp)
below 27 mm^3 count. This reproduces the exact real-data pattern where
all_instance_fp == large_instance_fp != small_instance_fp.

--- 2026-07-29 source-verification pass (RADIOLOGIST_INPUT.md r4 Phase 2.2), against the
actual vendored metrics_MET.py fetched this session (github.com/rachitsaluja/
BraTS-2024-Metrics, function get_LesionWiseScores / get_LesionWiseResults) ---

(a) small_instance FP bucketing: STILL UNCONFIRMED, status unchanged. The vendored source
    contains NO small/large split concept at all -- it computes exactly one Num_FP per
    region, and its only size-based mechanism is a flat `lesion_volume_thresh = 2` (mm^3,
    not 27) used solely to drop near-zero-volume GT lesions out of Num_TP/Num_FN entirely
    (not to define a separate small-instance metric). The 27mm^3 split's EXISTENCE and
    threshold value are correctly sourced from the challenge-page text (see above), not
    from this file, and remain trustworthy on that basis -- but this specific port's
    small_instance FP-restriction logic has no source confirmation, positive or negative,
    from the file actually available. No new evidence either way this pass.

(b) empty/empty (and empty-GT/non-empty-pred) convention: NOT SETTLEABLE from this source
    at all -- the fetched file contains ONLY per-case Num_TP/Num_FP/Num_FN computation, no
    cohort-averaging/aggregation code whatsoever (that logic, whatever it is, lives in the
    challenge's own leaderboard driver, which was not available to read). Worth flagging
    precisely: the challenge-page quote used to justify the "no penalty" convention
    ("fails to produce a result metric... won't be set to its worst possible value") most
    naturally describes a genuinely UNDEFINED case (both empty, precision/recall 0/0) --
    for the asymmetric case this project actually leans on (empty GT, non-empty
    prediction), get_LesionWiseScores produces a perfectly well-defined result (Num_FP =
    number of predicted components, Num_TP=Num_FN=0, giving precision=0/recall=1/F1=0 by
    this project's own _prf convention), i.e. NOT a failure to produce a result. So the
    challenge text may not even license excl_empty's treatment of this specific case.
    `excl_empty`'s justification remains exactly what it always was -- it reproduces the
    real leaderboard's empirically observed pattern (SESSION_D5_SWEEP_SUMMARY.md Sec 1) --
    an inference from a fit, not a source-confirmed rule. This is unchanged by this pass,
    but is flagged here as the single largest unresolved-methodology risk feeding Phase
    4.1b and Phase 5, since both lean on this convention heavily.

(c) multi-prediction matching (does a second predicted component overlapping an
    ALREADY-matched GT lesion count as a second FP, get merged, or get dropped for free):
    CONFIRMED from source. `get_LesionWiseScores` appends every dilation-overlapping
    predicted-component id to a flat `tp` list per GT lesion (one append per (gt lesion,
    overlapping pred id) pair); `Num_FP = len(fp)` where `fp` is the set of predicted ids
    NOT in that `tp` list. A second predicted component overlapping an already-matched GT
    lesion has its id added to `tp` exactly like the first, so it is excluded from `fp`
    (costs nothing) and does not create an extra TP either, since Num_TP counts matched GT
    lesions (`len(gt_tp)`), not predicted components. Redundant candidates are free,
    confirmed directly, not inferred -- this validates Phase 4.1's break-even rule exactly
    as stated (p_red carries zero cost; no tightening required).
"""
import math
import os
import sys

import cc3d
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation, generate_binary_structure

sys.path.insert(0, os.path.dirname(__file__))
from _brats_surface_distance import compute_robust_hausdorff, compute_surface_distances
from medpy.metric.binary import __surface_distances as medpy_surface_distances

DILATION_STRUCT = generate_binary_structure(3, 2)  # upstream's dilation_struct, dil_factor=1
PRED_VOXEL_PREFILTER = 2  # upstream save_tmp_files: drop combined predicted lesions <= 2 raw voxels
SMALL_LARGE_THRESH_MM3 = 27.0  # official BraTS-METS 2026 rule, not upstream's generic default

REGIONS = {
    "wt": (1, 2, 3),
    "tc": (1, 3),
    "et": (3,),
    "rc": (4,),
}


def region_mask(labelmap, labels):
    return np.isin(labelmap, labels)


def dice(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if not a.any() and not b.any():
        return 1.0
    denom = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / denom) if denom else 1.0


def normalized_surface_dice(a, b, threshold_mm, spacing):
    """Verbatim port of upstream's normalized_surface_dice (nnU-Net lineage), via medpy's
    internal __surface_distances. Edge cases (one/both empty) handled by the caller,
    matching upstream's own call-site guards -- this function assumes both non-empty."""
    a_to_b = medpy_surface_distances(a, b, spacing, 1)
    b_to_a = medpy_surface_distances(b, a, spacing, 1)
    return _nsd_from_distances(a_to_b, b_to_a, threshold_mm)


def _nsd_from_distances(a_to_b, b_to_a, threshold_mm):
    tp_a = np.sum(a_to_b <= threshold_mm) / len(a_to_b)
    tp_b = np.sum(b_to_a <= threshold_mm) / len(b_to_a)
    fp = np.sum(a_to_b > threshold_mm) / len(a_to_b)
    fn = np.sum(b_to_a > threshold_mm) / len(b_to_a)
    return (tp_a + tp_b) / (tp_a + tp_b + fp + fn + 1e-8)


def _bbox_slices(mask_a, mask_b, spacing, pad_mm):
    """Bounding box of mask_a|mask_b, padded by pad_mm (converted to voxels per axis).
    Safe for a THRESHOLD-based surface distance query (NSD): any surface point whose
    true nearest-neighbor distance is <= pad_mm is guaranteed to have that neighbor
    inside the padded box, so the <=threshold boolean this feeds is exact as long as
    pad_mm >= threshold_mm. NOT safe for magnitude queries like HD95 (unbounded)."""
    union = mask_a | mask_b
    coords = np.argwhere(union)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    pad_vox = np.array([int(np.ceil(pad_mm / s)) for s in spacing])
    mins = np.maximum(mins - pad_vox, 0)
    maxs = np.minimum(maxs + pad_vox, mask_a.shape)
    return tuple(slice(int(mn), int(mx)) for mn, mx in zip(mins, maxs))


def _nsd_both(pred_mask, gt_mask, spacing):
    """Returns (nsd@0.5mm, nsd@1.0mm), computing the underlying medpy surface-distance
    arrays exactly ONCE and reusing them for both tolerances (upstream calls
    normalized_surface_dice separately per tolerance, recomputing the same distance
    transform twice -- an easy win when we need both, which we always do here)."""
    if not gt_mask.any() and not pred_mask.any():
        return 1.0, 1.0
    if not gt_mask.any() or not pred_mask.any():
        return 0.0, 0.0
    # medpy's surface-distance call runs a full-volume EDT per invocation -- cropping to
    # a padded bounding box first (see _bbox_slices) is a ~50x speedup on typical lesion
    # sizes and is exact for this threshold-based use, not an approximation. Pad to the
    # larger of the two tolerances (1.0mm) so one crop serves both.
    box = _bbox_slices(pred_mask, gt_mask, spacing, pad_mm=1.0 + 3.0)
    a, b = pred_mask[box], gt_mask[box]
    a_to_b = medpy_surface_distances(a, b, spacing, 1)
    b_to_a = medpy_surface_distances(b, a, spacing, 1)
    return (float(_nsd_from_distances(a_to_b, b_to_a, 0.5)),
            float(_nsd_from_distances(a_to_b, b_to_a, 1.0)))


def _combine_by_dilation(raw_cc, dilated_cc):
    """Vectorized equivalent of upstream's get_GTseg_combinedByDilation /
    get_Predseg_combinedByDilation: every original-CC voxel is relabeled with the id of
    the DILATED blob it falls inside (raw mask is always a subset of the dilated mask,
    and dilated blobs are disjoint, so this is an exact one-line replacement for
    upstream's per-component Python loop, not an approximation of it)."""
    out = np.zeros_like(dilated_cc)
    nz = raw_cc > 0
    out[nz] = dilated_cc[nz]
    return out


def build_lesions(mask, spacing, voxel_prefilter=0, seed_mask=None, min_split_size_mm3=0.0):
    """CC labeling + dilation-based component-merging, matching upstream's
    save_tmp_files exactly (cc3d connectivity=26, dilation via generate_binary_structure
    (3,2) x1 iteration). `voxel_prefilter` (raw voxel count, NOT mm^3) drops small
    combined lesions post-merge -- upstream applies this to predictions only (2 voxels);
    call with voxel_prefilter=0 for ground truth (upstream never filters GT here).

    `seed_mask`, if given (D3's watershed fix): replaces the naive connected-components
    step with watershed_split.split_by_seeds(mask, seed_mask, ...) -- a confluent WT blob
    containing 2+ distinct ET seed components gets split into separate instances before
    the dilation-combining step runs, instead of staying one component that can only ever
    match one GT lesion. Only meaningful for PREDICTIONS (see watershed_split.py's
    docstring for why GT is never split this way).

    Returns (combined_label_map, {lesion_id: volume_mm3}). The combined_label_map is
    what everything downstream (matching, FP detection) operates on."""
    mask = np.asarray(mask, dtype=bool)
    voxel_vol = float(np.prod(spacing))
    full_shape = mask.shape
    if not mask.any():
        return np.zeros(full_shape, dtype=np.int32), {}

    # Crop to the mask's own bounding box (padded past the dilation's 1-voxel reach)
    # before running cc3d/dilation -- a predicted region mask is typically a small
    # fraction of the full FOV, and both operations are O(full volume) otherwise.
    coords = np.argwhere(mask)
    mins = np.maximum(coords.min(axis=0) - 2, 0)
    maxs = np.minimum(coords.max(axis=0) + 3, full_shape)
    box = tuple(slice(int(a), int(b)) for a, b in zip(mins, maxs))
    mask_c = mask[box]

    if seed_mask is not None:
        # Watershed split IS the instancing decision here -- skip the dilation-combining
        # step entirely rather than run it afterward. The two split pieces are, by
        # construction, still touching (they came from one physically connected blob), so
        # a generic 1-voxel dilation-merge would immediately re-fuse them and silently
        # undo the split. Trade-off: this also forgoes the dilation step's OTHER role
        # (bridging a genuine ~1-voxel gap between separate raw components) for this
        # region -- accepted as a minor, documented cost, not hidden.
        from watershed_split import split_by_seeds
        combined_c = split_by_seeds(mask_c, np.asarray(seed_mask, dtype=bool)[box], spacing,
                                     min_split_size_mm3=min_split_size_mm3)
    else:
        raw_cc = cc3d.connected_components(mask_c.astype(np.uint8), connectivity=26)
        dilated_mask = binary_dilation(mask_c, structure=DILATION_STRUCT, iterations=1)
        dilated_cc = cc3d.connected_components(dilated_mask.astype(np.uint8), connectivity=26)
        combined_c = _combine_by_dilation(raw_cc, dilated_cc)

    ids, counts = np.unique(combined_c, return_counts=True)
    keep = {int(i) for i, c in zip(ids, counts) if i != 0 and c > voxel_prefilter}
    if not keep:
        return np.zeros(full_shape, dtype=np.int32), {}
    if len(keep) != len(ids) - (1 if 0 in ids else 0):
        combined_c = np.where(np.isin(combined_c, list(keep)), combined_c, 0)
    volumes = {int(i): float(c) * voxel_vol for i, c in zip(ids, counts) if int(i) in keep}

    combined = np.zeros(full_shape, dtype=combined_c.dtype)
    combined[box] = combined_c
    return combined, volumes


def build_gt_cache(gt_labelmap, spacing):
    """Call ONCE per case, reused across every swept prediction configuration --
    this is the expensive-to-recompute, config-invariant half of the algorithm."""
    cache = {}
    for region, labels in REGIONS.items():
        mask = region_mask(gt_labelmap, labels)
        combined, volumes = build_lesions(mask, spacing, voxel_prefilter=0)
        cache[region] = {"combined": combined, "volumes": volumes, "mask": mask}
    return cache


def _score_region(gt_combined, gt_volumes, pred_combined, pred_volumes, spacing, hd_value):
    """Faithful port of get_LesionWiseScores' matching loop, in memory. Matching itself
    (dilate + overlap lookup) is restricted to each lesion's own padded bounding box
    (via scipy's find_objects, one O(N) pass) rather than dilating/searching the full
    volume per lesion -- this is the dominant cost otherwise (a full-volume
    binary_dilation call per lesion) and cropping to a box padded beyond the dilation's
    reach (1 iteration of an 18-connectivity structure -> 2 voxels of padding is ample)
    changes nothing about the result, since the dilation can't reach outside that box."""
    matched_pred_ids = set()
    per_lesion = []  # dicts: gt_id, gt_vol, matched, dice, hd95, nsd05, nsd10

    gt_objects = ndimage.find_objects(gt_combined) if gt_volumes else []

    for gt_id, gt_vol in gt_volumes.items():
        obj = gt_objects[gt_id - 1]
        box = tuple(slice(max(s.start - 2, 0), min(s.stop + 2, dim))
                    for s, dim in zip(obj, gt_combined.shape))
        gt_tmp_local = gt_combined[box] == gt_id
        gt_dilated_local = binary_dilation(gt_tmp_local, structure=DILATION_STRUCT, iterations=1)
        overlap_ids = set(np.unique(pred_combined[box][gt_dilated_local]).tolist()) - {0}
        matched = len(overlap_ids) > 0

        gt_tmp = gt_combined == gt_id
        if matched:
            matched_pred_ids |= overlap_ids
            pred_blob = np.isin(pred_combined, list(overlap_ids))
        else:
            pred_blob = np.zeros_like(gt_tmp)

        d = dice(pred_blob, gt_tmp)
        if not gt_tmp.any() and not pred_blob.any():
            hd = 0.0
        else:
            sd = compute_surface_distances(gt_tmp, pred_blob, spacing)
            hd = compute_robust_hausdorff(sd, 95)
            if math.isinf(hd):
                hd = hd_value
        nsd05, nsd10 = _nsd_both(pred_blob, gt_tmp, spacing)

        per_lesion.append(dict(gt_id=gt_id, gt_vol=gt_vol, matched=matched,
                                dice=d, hd95=hd, nsd05=nsd05, nsd10=nsd10))

    fp_ids = set(pred_volumes.keys()) - matched_pred_ids
    return per_lesion, fp_ids


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def score_case(pred_labelmap, gt_labelmap, spacing, gt_cache=None, size_thresh_mm3=SMALL_LARGE_THRESH_MM3,
               seed_masks=None, min_split_size_mm3=0.0):
    """Score one case across all 4 regions. Pass a pre-built `gt_cache` (from
    build_gt_cache) to skip the GT-side connected-component work -- the whole point of
    separating it out for a sweep that scores many prediction configurations per case.

    `seed_masks`: optional {region: seed_mask} (e.g. {"wt": et_mask} for D3's watershed
    fix) -- applied to the PREDICTION side only, never GT (see watershed_split.py)."""
    if gt_cache is None:
        gt_cache = build_gt_cache(gt_labelmap, spacing)
    seed_masks = seed_masks or {}

    hd_value = math.sqrt(sum(s ** 2 for s in pred_labelmap.shape))  # upstream's own formula, voxel units

    out = {}
    for region, labels in REGIONS.items():
        pred_mask = region_mask(pred_labelmap, labels)
        pred_combined, pred_volumes = build_lesions(
            pred_mask, spacing, voxel_prefilter=PRED_VOXEL_PREFILTER,
            seed_mask=seed_masks.get(region), min_split_size_mm3=min_split_size_mm3)
        gt_combined, gt_volumes = gt_cache[region]["combined"], gt_cache[region]["volumes"]

        per_lesion, fp_ids = _score_region(gt_combined, gt_volumes, pred_combined, pred_volumes, spacing, hd_value)

        large = [l for l in per_lesion if l["gt_vol"] >= size_thresh_mm3]
        small = [l for l in per_lesion if l["gt_vol"] < size_thresh_mm3]

        tp_all = sum(1 for l in per_lesion if l["matched"])
        fn_all = sum(1 for l in per_lesion if not l["matched"])
        fp_all = len(fp_ids)

        tp_large = sum(1 for l in large if l["matched"])
        fn_large = sum(1 for l in large if not l["matched"])

        tp_small = sum(1 for l in small if l["matched"])
        fn_small = sum(1 for l in small if not l["matched"])
        fp_small = sum(1 for pid in fp_ids if pred_volumes[pid] < size_thresh_mm3)

        p_all, r_all, f1_all = _prf(tp_all, fp_all, fn_all)
        p_large, r_large, f1_large = _prf(tp_large, fp_all, fn_large)
        p_small, r_small, f1_small = _prf(tp_small, fp_small, fn_small)

        # segmentation metrics: only over lesions >= 27mm^3, per the official rule.
        # Denominator includes ALL false positives (any size) -- upstream's own
        # lesion_wise_dice = sum(dice over thresh rows) / (len(thresh rows) + len(fp)).
        denom = len(large) + fp_all
        if denom > 0:
            lw_dice = sum(l["dice"] for l in large) / denom
            lw_hd95 = (sum(l["hd95"] for l in large) + fp_all * hd_value) / denom
            lw_nsd05 = sum(l["nsd05"] for l in large) / denom
            lw_nsd10 = sum(l["nsd10"] for l in large) / denom
        else:
            lw_dice, lw_hd95, lw_nsd05, lw_nsd10 = 1.0, 0.0, 1.0, 1.0

        out[region] = {
            "all_instance": {"tp": tp_all, "fp": fp_all, "fn": fn_all,
                              "precision": p_all, "recall": r_all, "f1": f1_all,
                              "trivial": (tp_all + fp_all + fn_all) == 0},
            "large_instance": {"tp": tp_large, "fp": fp_all, "fn": fn_large,
                               "precision": p_large, "recall": r_large, "f1": f1_large,
                               "trivial": (tp_large + fp_all + fn_large) == 0},
            "small_instance": {"tp": tp_small, "fp": fp_small, "fn": fn_small,
                               "precision": p_small, "recall": r_small, "f1": f1_small,
                               "trivial": (tp_small + fp_small + fn_small) == 0},
            "lesionwise_dsc": lw_dice,
            "lesionwise_hd95": lw_hd95,
            "lesionwise_nsd@0.5mm": lw_nsd05,
            "lesionwise_nsd@1.0mm": lw_nsd10,
            "lesionwise_trivial": denom == 0,  # no large GT lesions and no FPs at all
        }
    return out


def aggregate_cohort(per_case_results):
    """Mean/std across cases for every field, in the leaderboard's flattened
    all_instance_f1_et / lesionwise_dsc_mean_et / etc naming convention. Also returns a
    LOCAL composite score -- NOT the official ranking (BraTS ranks submissions against
    each other via rank-sum across teams per the challenge rules; that can't be
    reproduced from a single run with no other teams to rank against). This composite is
    only for comparing our own sweep configurations against each other.

    Every field is reported under BOTH averaging conventions the "empty/empty" question
    (NEXT_STEPS.md D2) is actually asking about -- we could not confirm from source which
    one the real evaluator uses, so both are computed rather than assumed:
      - "<field>_<region>": standard mean, both-empty cases score 1.0 and count in the
        average (this project's/upstream's own convention elsewhere -- score_case.py's
        dice(), rachitsaluja's own full_dice check).
      - "<field>_<region>_excl_empty": same mean but with both-empty cases DROPPED from
        the average entirely (denominator shrinks) rather than credited.
      - "n_trivial_<bucket>_<region>" / "n_cases": how many cases were both-empty for
        that bucket, so the two conventions' divergence is visible, not just their gap."""
    regions = list(REGIONS.keys())
    out = {}
    for region in regions:
        for bucket in ["all_instance", "large_instance", "small_instance"]:
            trivial = [c[region][bucket]["trivial"] for c in per_case_results]
            out[f"n_trivial_{bucket}_{region}"] = int(sum(trivial))
            for field in ["tp", "fp", "fn", "f1"]:
                vals = [c[region][bucket][field] for c in per_case_results]
                out[f"{bucket}_{field}_{region}"] = float(np.mean(vals))
                non_trivial_vals = [v for v, t in zip(vals, trivial) if not t]
                out[f"{bucket}_{field}_{region}_excl_empty"] = float(np.mean(non_trivial_vals)) if non_trivial_vals else float("nan")
        lw_trivial = [c[region]["lesionwise_trivial"] for c in per_case_results]
        out[f"n_trivial_lesionwise_{region}"] = int(sum(lw_trivial))
        for metric, key in [("lesionwise_dsc", "lesionwise_dsc"), ("lesionwise_hd95", "lesionwise_hd95"),
                             ("lesionwise_nsd", "lesionwise_nsd@1.0mm"), ("lesionwise_nsd_0.5mm", "lesionwise_nsd@0.5mm")]:
            vals = [c[region][key] for c in per_case_results]
            out[f"{metric}_mean_{region}"] = float(np.mean(vals))
            out[f"{metric}_std_{region}"] = float(np.std(vals))
            non_trivial_vals = [v for v, t in zip(vals, lw_trivial) if not t]
            out[f"{metric}_mean_{region}_excl_empty"] = float(np.mean(non_trivial_vals)) if non_trivial_vals else float("nan")
            out[f"{metric}_std_{region}_excl_empty"] = float(np.std(non_trivial_vals)) if non_trivial_vals else float("nan")

    # Composite uses the excl_empty convention -- validated 2026-07-28 against a real
    # leaderboard email: scoring empty/empty as 1.0 and including it INVERTED the RC
    # comparison (RC scored highest of all 4 regions instead of far lowest, contradicting
    # the real baseline's RC=0.124 vs ET/TC/WT~0.65-0.71). Excluding empty/empty cases
    # from the average reproduced the real pattern closely (RC far below the rest,
    # small~1/3-1/2 of large, WT lowest lesionwise_dsc) -- see NEXT_STEPS.md D2's
    # std_rc=0.031 clue, now confirmed empirically rather than just inferred from it.
    # nanmean, not mean: a region can legitimately have ZERO non-trivial cases in a small
    # or heavily-zeroed sample (e.g. RC under a hard-zero rule with few real-RC cases in
    # the batch), making its own excl_empty mean NaN by construction (see above). A plain
    # mean would let that one NaN silently poison the entire composite; nanmean instead
    # drops that region from THIS composite calculation and is reported so it's visible,
    # not hidden.
    seg_vals = [out[f"lesionwise_dsc_mean_{r}_excl_empty"] for r in regions]
    det_vals = [out[f"all_instance_f1_{r}_excl_empty"] for r in regions]
    if any(np.isnan(v) for v in seg_vals) or any(np.isnan(v) for v in det_vals):
        out["composite_nan_regions_dropped"] = (
            [r for r, v in zip(regions, seg_vals) if np.isnan(v)]
            + [r for r, v in zip(regions, det_vals) if np.isnan(v)]
        )
    seg_composite = float(np.nanmean(seg_vals))
    det_composite = float(np.nanmean(det_vals))
    out["composite_segmentation_local"] = seg_composite
    out["composite_detection_local"] = det_composite
    out["composite_overall_local"] = float((seg_composite + det_composite) / 2)
    # Kept alongside for reference/comparison, NOT used for ranking -- this is the
    # convention shown to disagree with the real evaluator's behavior.
    out["composite_segmentation_local_incl_empty"] = float(np.mean([out[f"lesionwise_dsc_mean_{r}"] for r in regions]))
    out["composite_detection_local_incl_empty"] = float(np.mean([out[f"all_instance_f1_{r}"] for r in regions]))
    out["n_cases"] = len(per_case_results)
    return out
