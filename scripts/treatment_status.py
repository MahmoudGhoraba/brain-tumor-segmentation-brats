"""
treatment_status.py -- a "could this case possibly contain a resection cavity" proxy,
derived from institution membership rather than guessed from the timepoint suffix.

Why not the timepoint suffix ("-000" = baseline = pre-treatment): checked against the
BraTS-METS 2025 Lighthouse Challenge paper's institutional breakdown (arXiv 2504.12527,
Table 2, same 1296/179/303 split this project uses) and it does NOT hold. Institutions
Duke/NCI/Missouri/WashU/Yale contribute ZERO post-treatment cases in that table regardless
of how many follow-up timepoints a patient has; only UCSF and UCSD contribute
post-treatment cases. So "-003" from Duke is still pre-treatment; "-000" from UCSD could
already be post-op if that happens to be the first scan in that patient's post-surgical
series. Institution, not suffix, is what gates RC-capability.

Why UCSD specifically, and why this is a MEASUREMENT not a guess: TrainingData_batch1.zip
is not uniformly flat -- 646 of 1296 cases sit one level deeper under a
"UCSD - Training/" folder (GOTCHAS D8, scripts/merge_corrected_labels.py::find_raw_case_dirs).
Patient ID numbers are institution-blocked with zero overlap (checked directly: UCSD
training patients span 01057-01356, everyone else spans 00001-00822, intersection empty).
Checked directly against ground truth: of the 650 non-UCSD training cases, ZERO have any
RC voxel in their label (0/650); of the 646 UCSD cases, 151/646 (23.4%) do. That is a
clean, data-confirmed signal, not an assumption -- non-UCSD training cases cannot have RC
because none of them ever do, matching the paper's "0 post-treatment cases" for those
institutions exactly.

Validation-phase cases have no directory-structure institution tag (the raw
ValidationData_batch1 folder is flat), so this module extends the SAME UCSD patient-ID
set found in training: a validation case is "UCSD" if its patient ID exactly matches a
training UCSD patient (same longitudinal patient, a later timepoint) OR falls inside the
UCSD training ID range (01057-01356) for patients not seen in training.

Corrected 2026-07-29 (RADIOLOGIST_INPUT.md r4 Phase 5.2 / FULL_RECORD.md Sec 8+11): this
paragraph previously claimed the proxy recovers "89 of the paper's stated 91 UCSD
validation cases," with a 2-case conservative gap. Re-run directly against the real
179-case validation datalist (data/manifests/datalist.json's "testing" split): `is_ucsd`
returns 91 True / 88 False -- an EXACT match to the paper's 91 UCSD cases, no gap at all.
The "89 of 91" figure was stale (most likely describing an earlier, since-fixed version of
this module) and is corrected here rather than left to mislead the next reader.

Net effect: `is_possibly_post_treatment(case_id)` is a reliable NO for non-UCSD cases
(confirmed against real labels) and a "maybe" for UCSD cases -- exactly the
pre-treatment/ambiguous split NEXT_STEPS.md D2 asks for ("zero RC on pre-treatment cases
... confirm metadata reliably encodes treatment status; if it does not, skip rather than
guess"). UCSD cases are NOT zeroed by this module; they still go through the normal
threshold/rim-test pipeline.
"""
import os
import re

RAW_TRAIN_UCSD_DIR = "/workspace/data/raw/TrainingData_batch1/UCSD - Training"
CASE_RE = re.compile(r"BraTS-MET-(\d+)-(\d+)")

_ucsd_patients = None
_ucsd_id_min = None
_ucsd_id_max = None


def _load():
    global _ucsd_patients, _ucsd_id_min, _ucsd_id_max
    if _ucsd_patients is not None:
        return
    patients = set()
    for d in os.listdir(RAW_TRAIN_UCSD_DIR):
        m = CASE_RE.search(d)
        if m:
            patients.add(m.group(1))
    _ucsd_patients = patients
    _ucsd_id_min = min(patients)
    _ucsd_id_max = max(patients)


def patient_id_of(case_id):
    m = CASE_RE.search(case_id)
    if not m:
        raise ValueError(f"case_id does not match BraTS-MET-<patient>-<timepoint>: {case_id}")
    return m.group(1)


def is_ucsd(case_id):
    """True if this case's patient is a known (or ID-range-plausible) UCSD patient --
    the institution capable of post-treatment/RC-bearing scans among the cohort this
    proxy can't otherwise resolve. See module docstring for the ~2/91 known gap."""
    _load()
    pid = patient_id_of(case_id)
    if pid in _ucsd_patients:
        return True
    return _ucsd_id_min <= pid <= _ucsd_id_max


def is_reliably_pretreatment(case_id):
    """True only when we have a DATA-CONFIRMED guarantee this case cannot contain a
    resection cavity (non-UCSD: 0/650 training cases with RC, matching the paper's
    documented 0-post-treatment-cases institutions exactly). False means "unknown/maybe",
    not "is post-treatment" -- UCSD cases include both pre- and post-treatment scans and
    still need the normal per-case pipeline (threshold, rim test), not a blanket verdict."""
    return not is_ucsd(case_id)


def zero_rc_if_pretreatment(labelmap, case_id, rc_label=4):
    """Zero out the resection-cavity label wherever this case is confirmed pre-treatment.
    Leaves UCSD (ambiguous) cases untouched."""
    if is_reliably_pretreatment(case_id):
        labelmap = labelmap.copy()
        labelmap[labelmap == rc_label] = 0
    return labelmap
