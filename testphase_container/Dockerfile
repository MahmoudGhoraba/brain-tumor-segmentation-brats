# Test-phase container for BraTS 2026 Task 1 (Brain Metastases Segmentation).
#
# Target hardware (Synapse wiki, syn74274097, "Submission Instructions" -> Containerized
# Algorithm -> Compute Constraints, Tasks 1-3 row, fetched 2026-07-30):
#   NVIDIA A10G GPU (24GB VRAM, sm_86), 16 vCPUs, 200GB storage, Max CUDA version 13.0.
#   Docker runtime limits: --memory=48G, --shm-size=16G. Inference time limit: 12 hours
#   total (across all 303 test cases).
#
# --- Part 1 decision: architecture coverage, verified not assumed ---
# This project's dev/verification environment (H200, sm_90) runs torch==2.13.0+cu130,
# a STANDARD PyTorch pip wheel (confirmed via `pip show torch` -- installed in a venv,
# not NGC-bundled). Its own torch.cuda.get_arch_list() reports:
#   ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
# sm_86 (the A10G evaluation target) AND sm_90 (this dev/verification GPU) are BOTH
# covered by the exact build already used to produce the byte-verified reproduction of
# submission 9774164. No base-image or torch-version change is forced, and none is made:
# switching to an older CUDA build (e.g. cu121) "for safety" would introduce an
# UNVERIFIED numerical environment in exchange for solving a coverage problem that does
# not exist here. torch==2.13.0+cu130 also sits exactly at the evaluation hardware's
# stated "Max CUDA version: 13.0" ceiling, not over it.
#
# Base image: nvidia/cuda cudnn-runtime (NOT the full NGC PyTorch image) -- smaller,
# and torch is installed explicitly pinned rather than inherited from a bundled version
# whose exact torch/arch coverage would need separate verification. VERIFY at build time
# on the H200 machine (this exact tag was not build-tested in this session, since no
# container runtime is available here -- see CONTAINER.md):
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

# --- DESIGN CHOICE, stated up front: this container reuses each fold's own
# scripts/segmenter.py `--infer` CLI path verbatim (the SAME mechanism scripts/
# ensemble.py already used to build submission 9774164, byte-verified against the real
# scored submission -- see configs/FROZEN_TESTPHASE.yaml's header). A hand-rolled,
# lighter-weight reimplementation of the preprocessing/inverse-transform pipeline was
# considered and rejected: that transform pair (resample to 1mm iso, invert back to
# each case's native geometry) is explicitly flagged elsewhere in this project as "the
# highest-risk code," given native-space spacing spans 0.200-1.400 mm^3/voxel. ---

WORKDIR /opt/algorithm

# --- Zero network access at EVALUATION time (wiki, verbatim: "Zero network access:
# Submissions are run without internet. All dependencies must be installed within your
# Dockerfile, not via runtime scripts.") -- network IS available at BUILD time, which is
# when everything below is pinned and installed once, permanently. ---
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# torch pinned explicitly and separately from requirements.txt (Part 1's decision) --
# same build verified in this session to cover sm_86 + sm_90 and to reproduce 9774164
# byte-for-byte.
RUN pip install --no-cache-dir --break-system-packages \
    torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130

COPY testphase_container/requirements.txt /opt/algorithm/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /opt/algorithm/requirements.txt

# --- Part 2.4: defensive offline env vars. Audited entrypoint.py and every module it
# imports (regions_to_labelmap, postproc_filters, treatment_status, score_case) plus
# Model/segresnet_*/scripts/segmenter.py for download=True, torch.hub, hf_hub, pooch,
# and bare URLs -- none found in any code path actually exercised (grep, 2026-07-30; the
# only URL hit anywhere was a license header comment). These are set defensively anyway,
# since MONAI's own transform/bundle modules are imported in full and could in principle
# reach for a cache directory or remote resource in a code path not directly exercised. ---
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV TORCH_HOME=/opt/torch
ENV MONAI_DATA_DIRECTORY=/opt/monai
ENV MPLCONFIGDIR=/tmp
ENV PYTHONUNBUFFERED=1
# GOTCHAS.md C15: buffered print() output can sit unflushed until process exit --
# unacceptable inside a 12-hour-budget run with no way to attach and check progress.

# --- Entrypoint + the exact post-processing modules already validated against
# submission 9774164 (byte-identical, see configs/FROZEN_TESTPHASE.yaml). No training
# code paths included -- these four scripts contain inference-adjacent utilities only. ---
COPY testphase_container/entrypoint.py /opt/algorithm/entrypoint.py
COPY scripts/regions_to_labelmap.py /opt/algorithm/scripts/regions_to_labelmap.py
COPY scripts/postproc_filters.py /opt/algorithm/scripts/postproc_filters.py
COPY scripts/treatment_status.py /opt/algorithm/scripts/treatment_status.py
COPY scripts/score_case.py /opt/algorithm/scripts/score_case.py
# brats_lesionwise.py + its vendored _brats_surface_distance/ module: found by the
# fresh-venv smoke test (Part 3) to be a real, missing transitive dependency of
# postproc_filters.py's wt_no_et_filter/rim_test_filter (both call build_lesions()).
# Not in the first pass of this Dockerfile -- added after the gap was actually caught,
# not assumed complete from a manual read of the top-level imports alone.
#
# NOTE (Part 3, second gap caught by actually RUNNING the pipeline, not just importing
# it): requirements.txt now also pins fire==0.7.1 + termcolor==3.3.0. Without `fire`,
# segmenter.py's own __main__ block (Auto3DSeg template, unmodified) silently discards
# every CLI arg entrypoint.py passes it and falls back to a hardcoded, nonexistent
# relative config path -- a failure that looks like a path bug, not a missing package.
# See requirements.txt's comment for the exact reproduction.
COPY scripts/brats_lesionwise.py /opt/algorithm/scripts/brats_lesionwise.py
COPY scripts/_brats_surface_distance/ /opt/algorithm/scripts/_brats_surface_distance/
COPY configs/FROZEN_TESTPHASE.yaml /opt/algorithm/configs/FROZEN_TESTPHASE.yaml

# --- treatment_status.py's is_ucsd() reads the UCSD training-patient directory listing
# at import time (RAW_TRAIN_UCSD_DIR). That raw directory is 34GB and NOT bundled here
# -- baking the derived ID list instead. See entrypoint.py's patch of this at startup,
# and Part 1 of the site-proxy hardening task for the two-pass detect-and-fallback logic
# built around exactly this risk. ---
COPY configs/ucsd_patient_ids.json /opt/algorithm/configs/ucsd_patient_ids.json

# --- The image-only RC case-gate fallback classifier (zero patient-ID dependency by
# construction, AUC 0.869 on training-corpus patient-grouped CV), used only when Pass 1
# detects the primary site-proxy's ID-range logic did not transfer to the real
# test-phase case IDs. See entrypoint.py's decide_ucsd_assignment() and
# proxy_fallback_validation.log for the acceptance measurement. ---
COPY configs/rc_image_only_gate.joblib /opt/algorithm/configs/rc_image_only_gate.joblib

# --- Five full fold bundles: each fold's resolved config + inference script (Auto3DSeg's
# own scripts/segmenter.py, unmodified) + checkpoint. Reused verbatim, not reimplemented,
# per the design note above. ~333MB/fold checkpoint + ~1MB config + ~2.2MB scripts,
# five folds =~ 1.7GB total, PLUS configs/rc_image_only_gate.joblib (260KB) +
# ucsd_patient_ids.json (3.5KB) + FROZEN_TESTPHASE.yaml (8KB) above -- combined
# non-base-image payload ~1.7GB (measured on disk, see CONTAINER.md), comfortably inside
# the 200GB storage limit. ---
COPY Model/segresnet_0/configs/ /opt/algorithm/Model/segresnet_0/configs/
COPY Model/segresnet_0/scripts/ /opt/algorithm/Model/segresnet_0/scripts/
COPY Model/segresnet_0/model/model.pt /opt/algorithm/Model/segresnet_0/model/model.pt
COPY Model/segresnet_1/configs/ /opt/algorithm/Model/segresnet_1/configs/
COPY Model/segresnet_1/scripts/ /opt/algorithm/Model/segresnet_1/scripts/
COPY Model/segresnet_1/model/model.pt /opt/algorithm/Model/segresnet_1/model/model.pt
COPY Model/segresnet_2/configs/ /opt/algorithm/Model/segresnet_2/configs/
COPY Model/segresnet_2/scripts/ /opt/algorithm/Model/segresnet_2/scripts/
COPY Model/segresnet_2/model/model.pt /opt/algorithm/Model/segresnet_2/model/model.pt
COPY Model/segresnet_3/configs/ /opt/algorithm/Model/segresnet_3/configs/
COPY Model/segresnet_3/scripts/ /opt/algorithm/Model/segresnet_3/scripts/
COPY Model/segresnet_3/model/model.pt /opt/algorithm/Model/segresnet_3/model/model.pt
COPY Model/segresnet_4/configs/ /opt/algorithm/Model/segresnet_4/configs/
COPY Model/segresnet_4/scripts/ /opt/algorithm/Model/segresnet_4/scripts/
COPY Model/segresnet_4/model/model.pt /opt/algorithm/Model/segresnet_4/model/model.pt

# /input is supplied read-only by the evaluation harness; /output must be written flat.
# Neither directory is created here -- the harness owns both bind mounts.

ENTRYPOINT ["python3", "/opt/algorithm/entrypoint.py"]
CMD ["--input", "/input", "--output", "/output"]

# =============================================================================
# VERIFICATION STATUS, as of this build's authoring (2026-07-30). Read
# BUILD_AND_TEST.md and CONTAINER.md for full detail; this is a summary only.
#
# VERIFIED IN A FRESH VENV (this project's H200 dev host, NOT this Docker image itself --
# no OCI container runtime is available in the sandbox that wrote this Dockerfile; a
# manually-staged /opt/algorithm mirror of this exact file layout was used instead):
#   - Full import chain (entrypoint.py + every fold's segmenter.py) succeeds cleanly.
#   - entrypoint.py runs end-to-end on 20 real cases, all 5 folds, real GPU inference.
#   - Post-processing logic is byte-identical to submission 9774164 on all 179 real
#     validation cases (using cached probabilities, no live inference in that check).
#   - End-to-end (live inference + post-processing) is NOT byte-identical to 9774164
#     (11/20 exact, 9/20 differ by 1-20 voxels / 0.0039% of foreground voxels) --
#     root-caused to torch.backends.cudnn.benchmark=True under AMP, a genuine run-to-run
#     non-determinism confirmed by running the same fold twice on identical hardware.
#   - Metric-invariance (the replacement acceptance bar): 0/20 cases show any
#     instance-level tp/fp/fn/f1 mismatch between two independent runs, scored against
#     real ground truth, despite voxel-level drift in 11/20 of those cases.
#   - Peak GPU VRAM: 15,334 MiB (single GPU, 20 real cases, one fold) -- 9,242 MiB of
#     A10G headroom, cross-validated by two independent measurement methods.
#   - torch.cuda.get_arch_list() includes both sm_86 (A10G) and sm_90 (H200 dev host).
#
# MEASURED ON H200 (same hardware as above, so "measured on H200" and "verified in a
# fresh venv" are the same sessions here -- distinguished from PROJECTED below):
#   - Stage-separated timing (preprocess/infer/postproc) via synchronized, in-process
#     timing of segmenter.py's own methods.
#   - Real per-case-per-fold GPU inference throughput: 11.16 s/case-fold aggregate,
#     ~0.6 s/case-fold of which is GPU-bound (sliding-window inference itself).
#
# PROJECTED, NOT MEASURED (no A10G, no 16-vCPU host, and no real docker build/run
# available anywhere in the authoring sandbox):
#   - Runtime on the real A10G target: ~6.66 hours projected vs. a 12-hour budget
#     (8x GPU-scaling factor applied only to the GPU-bound stage; CPU/IO-bound stage
#     left unscaled from its H200-measured value). Sensitivity to the CPU/IO scaling
#     factor -- the binding, unmeasured risk -- is tabulated in BUILD_AND_TEST.md Part 4b.
#   - Whether this Dockerfile builds at all (base image availability, dependency
#     resolution) -- untested; no Docker in the authoring sandbox.
#   - Whether containerization itself (vs. the manually-staged venv mirror) changes any
#     number above.
#   - The site-proxy's is_ucsd() ID-range logic's transfer to genuine test-phase patient
#     IDs (mitigated, not eliminated, by the image-only fallback classifier).
# =============================================================================
