"""
The missing link: Auto3DSeg's sigmoid multi-region SegResNet bundle never
converts its 4 independent region-probability channels into a single BraTS
label map. Traced empirically on the smoke test (GOTCHAS C11): the bundle's
own save path is

    AsDiscreted(keys="seg", threshold=0.5)   # no argmax, sigmoid branch

which thresholds each of the 4 channels independently and saves all 4 as an
(X, Y, Z, 4) volume. Nothing downstream -- not validate.py, infer.py, train.py,
algo.py, nor MONAI's own auto3dseg package -- ever combines them. That means
no code path in this project can produce a valid single-channel {0,1,2,3,4}
submission mask without this module.

Four consumers: the smoke test, the Phase D threshold sweep, the Phase E
submission, the MLCube container. Import this, do not reinline it.
"""

import numpy as np

# Order this module assumes for the 4 input channels. Must match the
# resolved config's `class_names` exactly -- see assert_channel_order.
CHANNEL_ORDER = ("whole_tumor", "tumor_core", "enhancing_tumor", "resection_cavity")


def assert_channel_order(class_names):
    """
    Fail loudly if the caller's config disagrees with the channel order this
    module assumes (WT=0, TC=1, ET=2, RC=3). Call this against the resolved
    hyper_parameters.yaml (or equivalent) before trusting any output of
    regions_to_labelmap -- a silent channel transposition would produce a
    plausible-but-wrong label map forever, with nothing to catch it.
    """
    if list(class_names) != list(CHANNEL_ORDER):
        raise ValueError(
            f"regions_to_labelmap assumes channel order {CHANNEL_ORDER}, "
            f"but the config's class_names is {list(class_names)}. Refusing "
            "to proceed -- a channel order mismatch would silently "
            "transpose regions instead of raising."
        )


def regions_to_labelmap(
    probs,
    thresholds=(0.5, 0.5, 0.5, 0.5),
    rc_conflict="higher_prob",
    brain_mask=None,
    spacing=None,
    min_volume_mm3=None,
):
    """
    Convert 4-channel region probabilities into a single discrete BraTS-style
    label map.

    Args:
        probs: float array, shape (4, X, Y, Z), channel order
            0=whole_tumor, 1=tumor_core, 2=enhancing_tumor, 3=resection_cavity.
            Probabilities, not already-thresholded masks -- the Phase D sweep
            needs to try many `thresholds` against the same underlying
            probabilities without re-running GPU inference.
        thresholds: per-channel threshold in (wt, tc, et, rc) order, applied
            independently before hierarchy enforcement.
        rc_conflict: rule for voxels where RC and a tumor region both fire.
            RC is anatomically exclusive of tumor labels but predicted by an
            independent channel, so conflicts are possible.
            "higher_prob" (default, the only rule implemented): at a
            conflicting voxel, compare the RC channel's raw probability
            against the WT channel's raw probability (WT is the outer
            envelope of all tumor regions, so it's the natural single
            competing signal) and give the voxel to whichever is higher. Ties
            go to tumor (strict `>` for RC to win). This is a parameter, not
            a hardcoded rule, so Phase D can sweep alternatives.
        brain_mask: optional bool array, shape (X, Y, Z), True inside the
            brain (e.g. `t1c > 0`). If given, applied as the LAST step: any
            voxel outside the mask is forced to background (0), for every
            region. BraTS ground truth is always inside the brain, so this
            can only remove false positives -- exactly what lesion-wise Dice
            rewards. `None` (default) skips masking entirely inside this
            function; whether it's applied by default is a decision for the
            caller, not this function, so Phase D can sweep it on/off the
            same way it sweeps `thresholds` and `rc_conflict`. Caught a real
            bug on the smoke test's 1-epoch checkpoint: predicted whole_tumor
            volume was 2.6x the actual brain volume on one case (regions
            predicted in air, outside the head) -- see brain_volume_ratios.
        spacing: NOT used by this function. Accepted only so the signature
            stays stable for callers that also need it for min-volume
            filtering elsewhere. Passing a non-None value raises, same as
            min_volume_mm3 -- see below.
        min_volume_mm3: must stay None. Min-volume / small-component
            filtering is a SEPARATE, later step and does not belong in this
            function: it has to operate on connected components of the
            REGIONS (ET/TC/WT/RC) before the hierarchy is flattened, not on
            the combined output labels, because BraTS-METs' lesion-wise
            metric scores regions. Filtering post-combination would silently
            corrupt the hierarchy (e.g. deleting a small TC component without
            touching the ET voxels nested inside it, or vice versa). Raising
            here instead of silently ignoring it.

    Returns:
        uint8 array, shape (X, Y, Z), values in {0, 1, 2, 3, 4}.

    Order of operations (deliberate -- not the only valid choice, documented
    so Phase D can reason about what it's sweeping):
      a. threshold each channel independently at its own threshold.
      b. ENFORCE hierarchy before converting: tc |= et, then wt |= tc.
         Measured on the smoke test's 1-epoch checkpoint, the raw
         independently-thresholded channels violated containment (TC voxel
         count exceeded WT voxel count on multiple cases). Converting without
         this step produces label 3 (ET) sitting outside the WT region,
         which is structurally invalid no matter how good the model is.
      c. resolve RC-vs-tumor conflicts per `rc_conflict`.
      d. convert: 3 where et; 1 where tc & ~et; 2 where wt & ~tc; 4 where rc.
      e. if `brain_mask` given: zero out everything outside it.
    """
    if spacing is not None or min_volume_mm3 is not None:
        raise ValueError(
            "regions_to_labelmap does not do min-volume filtering. "
            "spacing/min_volume_mm3 must be applied in a separate step, "
            "operating on region connected components, after this function "
            "returns -- not before or inside it. See the docstring."
        )

    probs = np.asarray(probs)
    if probs.ndim != 4 or probs.shape[0] != 4:
        raise ValueError(f"expected probs shape (4, X, Y, Z), got {probs.shape}")

    thresholds = np.asarray(thresholds, dtype=probs.dtype)
    if thresholds.shape != (4,):
        raise ValueError(f"expected 4 thresholds (wt, tc, et, rc), got {thresholds.shape}")

    wt_p, tc_p, et_p, rc_p = probs[0], probs[1], probs[2], probs[3]

    # a. threshold each channel independently
    wt = wt_p >= thresholds[0]
    tc = tc_p >= thresholds[1]
    et = et_p >= thresholds[2]
    rc = rc_p >= thresholds[3]

    return combine_region_masks(wt, tc, et, rc, wt_p=wt_p, rc_p=rc_p, rc_conflict=rc_conflict, brain_mask=brain_mask)


def combine_region_masks(wt, tc, et, rc, wt_p, rc_p, rc_conflict="higher_prob", brain_mask=None):
    """
    Steps (b)-(e) of regions_to_labelmap, factored out so a caller that needs
    to filter small connected components OUT of wt/tc/et/rc first (Phase D's
    min-volume sweep -- see regions_to_labelmap's docstring on why that must
    happen BEFORE hierarchy enforcement, not after) can reuse the exact same,
    already-tested hierarchy/conflict/combine logic instead of duplicating it.

    wt, tc, et, rc: bool arrays, already thresholded (and, for the min-volume
        sweep case, already component-filtered). Same shape.
    wt_p, rc_p: the WT and RC probability channels (float), needed only for
        the "higher_prob" RC-vs-tumor conflict rule.
    """
    # b. enforce hierarchy before converting -- this is the step that
    # prevents label 3 from ever landing outside WT.
    tc = tc | et
    wt = wt | tc

    # c. RC-vs-tumor conflict resolution
    if rc_conflict == "higher_prob":
        conflict = rc & wt
        rc_wins = conflict & (rc_p > wt_p)
        tumor_wins = conflict & ~rc_wins
        wt = wt & ~rc_wins
        tc = tc & ~rc_wins
        et = et & ~rc_wins
        rc = rc & ~tumor_wins
    else:
        raise ValueError(f"unknown rc_conflict rule: {rc_conflict!r}")

    # d. convert to a single discrete label map
    labelmap = np.zeros(wt.shape, dtype=np.uint8)
    labelmap[wt & ~tc] = 2
    labelmap[tc & ~et] = 1
    labelmap[et] = 3
    labelmap[rc] = 4  # independent region; no overlap left after (c)

    # e. brain masking -- last step, can only remove predictions, never add
    if brain_mask is not None:
        brain_mask = np.asarray(brain_mask, dtype=bool)
        if brain_mask.shape != labelmap.shape:
            raise ValueError(f"brain_mask shape {brain_mask.shape} != labelmap shape {labelmap.shape}")
        labelmap[~brain_mask] = 0

    return labelmap


def brain_volume_ratios(labelmap, brain_mask):
    """
    Structural sanity check (GOTCHAS M6): predicted volume for any region
    must not exceed brain volume. Report as a ratio per region; a ratio > 1.0
    is a catastrophic-failure signal that would otherwise require someone to
    notice an implausible voxel count by eye.

    Args:
        labelmap: uint8 array (X, Y, Z), values in {0,1,2,3,4}.
        brain_mask: bool array (X, Y, Z), True inside the brain.

    Returns:
        dict of region name -> voxel_count / brain_voxel_count.
    """
    brain_mask = np.asarray(brain_mask, dtype=bool)
    brain_voxels = int(brain_mask.sum())
    if brain_voxels == 0:
        raise ValueError("brain_mask is empty -- cannot compute a ratio against zero brain voxels")

    regions = {
        "whole_tumor": np.isin(labelmap, [1, 2, 3]),
        "tumor_core": np.isin(labelmap, [1, 3]),
        "enhancing_tumor": labelmap == 3,
        "resection_cavity": labelmap == 4,
    }
    return {name: float(mask.sum()) / brain_voxels for name, mask in regions.items()}
