"""
postproc_filters.py -- false-positive filters for the D5 sweep, applied to already-
thresholded region masks before build_lesions/scoring (or before final labelmap combination
for the real submission pipeline). Each filter is a swept PARAMETER, not a hardcoded rule
(NEXT_STEPS.md's own framing) -- the sweep decides whether and how hard to apply each.

Priority per the 2026-07-28 message reprioritizing D4: filter 2 (WT-no-ET) is the one
worth building properly, since WT carries the highest FP rate (1.09/case). Filters 1
(enhancement contrast) and 4 (PCA shape) are included as cheap dimensions since C1 killed
the rationale for spending implementation effort making aggressive thresholds survivable.
The RC rim test is included too, though C2 measured only 4.5% of RC FPs match its
signature -- built for completeness/cheapness, not because it's expected to move much.
"""
import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, generate_binary_structure
from skimage.filters import apply_hysteresis_threshold

from brats_lesionwise import build_lesions, PRED_VOXEL_PREFILTER

_STRUCT = generate_binary_structure(3, 2)


def hysteresis_threshold(prob_channel, flood_threshold, seed_threshold):
    """Decouples the two effects a single RC threshold conflates (measured 2026-07-28:
    RC detection F1 keeps climbing to 0.99 while RC Dice peaks at 0.9 and declines past
    it -- a high cutoff keeps killing ungated false alarms AND starts eroding genuine
    moderate-confidence cavity voxels, and a single threshold can't separate those). A
    component only exists if some part of it clears `seed_threshold` (kills the ungated
    false-alarm population); once it exists, its full extent is recovered down to
    `flood_threshold` (keeps genuine-but-moderate-confidence voxels instead of eroding
    them). Thin wrapper around skimage's apply_hysteresis_threshold (Canny's algorithm),
    not reimplemented -- `low`/`high` there map to `flood_threshold`/`seed_threshold`
    here, named for what they mean in this context rather than skimage's edge-detection
    vocabulary."""
    return apply_hysteresis_threshold(prob_channel, low=flood_threshold, high=seed_threshold)


def rim_test_filter(rc_mask, et_mask, spacing, dilation_iters=1, containment_threshold=0.5):
    """C2's rim test (flowchart Step 6A): an RC component mostly enclosed by a filled ET
    envelope is a necrotic core, not a cavity -- drop it. Measured to catch only ~4.5% of
    real RC false positives (C2, 2026-07-28); included because it's cheap and correct on
    its own terms, not because it's expected to be the main RC fix."""
    if not rc_mask.any() or not et_mask.any():
        return rc_mask
    et_dilated = binary_dilation(et_mask, structure=_STRUCT, iterations=dilation_iters)
    envelope = binary_fill_holes(et_dilated)

    combined, volumes = build_lesions(rc_mask, spacing, voxel_prefilter=0)
    keep = np.ones_like(rc_mask)
    for lesion_id in volumes:
        comp_mask = combined == lesion_id
        containment = (comp_mask & envelope).sum() / comp_mask.sum()
        if containment > containment_threshold:
            keep[comp_mask] = False
    return rc_mask & keep.astype(bool)


def wt_no_et_filter(wt_mask, et_mask, spacing, min_et_voxels=1, rc_mask=None):
    """D2/NEXT_STEPS filter 2: a WT connected component with no (or negligible) enclosed
    ET mass is likely gliosis/old infarct/treatment change, not a met -- brain mets
    essentially always enhance. Exception: skip this filter for components that overlap a
    predicted resection cavity, where non-enhancing signal is expected (NEXT_STEPS'
    explicit carve-out). `min_et_voxels` is the swept threshold: how much enclosed ET mass
    a WT component needs to survive."""
    if not wt_mask.any():
        return wt_mask
    combined, volumes = build_lesions(wt_mask, spacing, voxel_prefilter=PRED_VOXEL_PREFILTER)
    keep = np.zeros_like(wt_mask)
    for lesion_id in volumes:
        comp_mask = combined == lesion_id
        et_voxels = int((comp_mask & et_mask).sum())
        near_rc = rc_mask is not None and (comp_mask & rc_mask).any()
        if et_voxels >= min_et_voxels or near_rc:
            keep[comp_mask] = True
    # Components below build_lesions' own voxel prefilter (tiny specks) were already
    # dropped from `combined`/`volumes` -- reproduce that on the raw mask too, since this
    # filter's job is to REPLACE wt_mask, not silently reintroduce specks build_lesions
    # would have filtered anyway.
    return wt_mask & keep.astype(bool)


def enhancement_contrast_filter(et_mask, t1c, t1n, spacing, shell_dilation_iters=3, min_contrast=0.0):
    """Filter 1 (Q3/Q4): drop ET components not meaningfully brighter on t1c than a
    dilated shell of surrounding tissue, relative to how much brighter that same shell
    region is on t1n -- t1-bright mimics (subacute haemorrhage, proteinaceous cyst, fat,
    mineralisation) fail this. Preprocessing z-scores each channel independently within
    the brain mask (NOT a shared intensity scale) -- uses the component-vs-shell CONTRAST
    DIFFERENCE between channels, never a raw voxel-wise t1c-t1n subtraction."""
    if not et_mask.any():
        return et_mask
    combined, volumes = build_lesions(et_mask, spacing, voxel_prefilter=0)
    keep = np.zeros_like(et_mask)
    for lesion_id in volumes:
        comp_mask = combined == lesion_id
        shell = binary_dilation(comp_mask, structure=_STRUCT, iterations=shell_dilation_iters) & ~comp_mask
        if not shell.any():
            keep[comp_mask] = True
            continue
        t1c_contrast = t1c[comp_mask].mean() - t1c[shell].mean()
        t1n_contrast = t1n[comp_mask].mean() - t1n[shell].mean()
        if (t1c_contrast - t1n_contrast) >= min_contrast:
            keep[comp_mask] = True
    return et_mask & keep.astype(bool)


def shape_reject_filter(mask, spacing, max_size_mm3=500.0, elongation_threshold=3.0):
    """Filter 4 (Q5): PCA on each component's voxel coordinates -- vessels/dura are
    tubular/sheet-like (lambda1 >> lambda2), mets are blobby. Gated to components below
    max_size_mm3 so a genuinely large lesion can never be removed."""
    if not mask.any():
        return mask
    combined, volumes = build_lesions(mask, spacing, voxel_prefilter=0)
    keep = np.ones_like(mask)
    for lesion_id, vol in volumes.items():
        if vol >= max_size_mm3:
            continue
        comp_mask = combined == lesion_id
        coords = np.argwhere(comp_mask) * np.array(spacing)
        if coords.shape[0] < 4:
            continue
        coords = coords - coords.mean(axis=0)
        cov = np.cov(coords.T)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eigvals = np.clip(eigvals, 1e-8, None)
        elongation = eigvals[0] / eigvals[1]
        if elongation > elongation_threshold:
            keep[comp_mask] = False
    return mask & keep.astype(bool)
