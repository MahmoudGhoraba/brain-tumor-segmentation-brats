"""
Scoring for BraTS-METs region masks: subject-wise DSC/NSD (GOTCHAS M8) plus a
lesion-wise precision/recall/F1 (GOTCHAS M9/M17) for the detection leaderboard
question. `panoptica` is the official implementation for the lesion-wise
metric; if unavailable, `lesionwise_prf` below implements a documented,
simpler any-overlap matching rule as a fallback -- NOT a claim of parity with
panoptica's actual algorithm, just a same-shaped number to reconcile against.

Regions scored: whole_tumor (label in {1,2,3}), tumor_core ({1,3}),
enhancing_tumor ({3}). resection_cavity ({4}) is NOT part of the official
2026 segmentation score (GOTCHAS M11 / C8) but is computed too since we've
been tracking it throughout the smoke test.
"""

import numpy as np
from scipy import ndimage

REGIONS = {
    "whole_tumor": (1, 2, 3),
    "tumor_core": (1, 3),
    "enhancing_tumor": (3,),
    "resection_cavity": (4,),
}


def region_mask(labelmap, labels):
    return np.isin(labelmap, labels)


def dice(pred_mask, gt_mask):
    p, g = pred_mask.astype(bool), gt_mask.astype(bool)
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0  # both empty: trivially perfect agreement
    return float(2.0 * (p & g).sum() / denom)


def nsd(pred_mask, gt_mask, spacing, tolerance_mm):
    """Normalized surface Dice at a given tolerance (mm), via MONAI."""
    from monai.metrics import compute_surface_dice
    import torch
    p, g = pred_mask.astype(bool), gt_mask.astype(bool)
    if not p.any() and not g.any():
        return 1.0
    if not p.any() or not g.any():
        return 0.0
    pt = torch.from_numpy(p)[None, None].float()
    gt = torch.from_numpy(g)[None, None].float()
    spacing = [float(x) for x in spacing]
    val = compute_surface_dice(pt, gt, class_thresholds=[tolerance_mm], spacing=spacing, include_background=True)
    return float(val.item())


def subject_wise_scores(pred_labelmap, gt_labelmap, spacing, nsd_tolerances_mm=(1.0, 2.0)):
    out = {}
    for region, labels in REGIONS.items():
        p = region_mask(pred_labelmap, labels)
        g = region_mask(gt_labelmap, labels)
        out[region] = {"dsc": dice(p, g)}
        for tol in nsd_tolerances_mm:
            out[region][f"nsd@{tol}mm"] = nsd(p, g, spacing, tol)
    return out


def labeled_components_mm3(mask, spacing):
    """Label connected components. Returns (labeled_array, {component_id: volume_mm3})."""
    voxel_vol = float(np.prod(spacing))
    labeled, n = ndimage.label(mask)
    if n == 0:
        return labeled, {}
    sizes = ndimage.sum(np.ones_like(labeled), labeled, index=np.arange(1, n + 1))
    volumes = {i: float(sizes[i - 1]) * voxel_vol for i in range(1, n + 1)}
    return labeled, volumes


def lesionwise_prf(pred_mask, gt_mask, spacing, min_volume_mm3=0.0, apply_filter_to="gt_only"):
    """
    Fallback lesion-wise precision/recall/F1 (panoptica install did not
    complete within the time box -- see caller/report for that status).

    Matching rule (explicit, since this is NOT panoptica's actual algorithm):
    a GT component is a TP if ANY predicted voxel overlaps it; a predicted
    component is an FP if it overlaps NO GT component. This is a simple
    any-overlap rule, not an IoU-threshold match -- documented, not hidden.
    Implemented via label-array lookups (np.unique on the overlap region),
    not full-volume boolean ANDs, to stay fast on full-resolution cases.

    apply_filter_to: "gt_only" filters components < min_volume_mm3 out of GT
        only (predictions below threshold become FPs, per GOTCHAS M13's
        "GT-only" interpretation); "both" filters both GT and predictions
        (small predictions are simply ignored, not penalised).
    """
    gt_labeled, gt_vols = labeled_components_mm3(gt_mask, spacing)
    pred_labeled, pred_vols = labeled_components_mm3(pred_mask, spacing)

    gt_ids = [i for i, v in gt_vols.items() if v >= min_volume_mm3]
    pred_ids = [i for i, v in pred_vols.items() if v >= min_volume_mm3] if apply_filter_to == "both" else list(pred_vols.keys())

    # which GT components does each predicted voxel overlap, and vice versa
    overlap_mask = (gt_labeled > 0) & (pred_labeled > 0)
    gt_hit_by_pred = set(np.unique(gt_labeled[overlap_mask]).tolist()) - {0}
    pred_hit_by_gt = set(np.unique(pred_labeled[overlap_mask]).tolist()) - {0}

    tp = len([i for i in gt_ids if i in gt_hit_by_pred])
    fn = len(gt_ids) - tp
    tp_pred_side = len([i for i in pred_ids if i in pred_hit_by_gt])
    fp = len(pred_ids) - tp_pred_side

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if len(gt_ids) == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1,
                n_gt_components=len(gt_ids), n_pred_components=len(pred_ids))
