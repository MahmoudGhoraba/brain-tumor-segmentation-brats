#!/usr/bin/env bash
# Literal, ordered commands for the H200 (or any real Docker+GPU host) to build, smoke-test,
# push, and submit this container. Fill in PROJECT_ID below before running Step 4+.
#
# Nothing in this file executes docker build/run in this sandbox for you -- there is no
# Docker daemon here. Run this ON THE MACHINE WITH DOCKER, from the repo root.
set -euo pipefail

IMAGE_NAME="bratsmet2026-task1"
TAG="v3"  # v1/v2 are the two Docker submissions already stuck INVALID on Synapse
          # (9774264, 9774732) -- v3 is the memory/disk fix (chunked, streaming
          # entrypoint.py), a genuinely different image, so it gets its own tag rather
          # than overwriting v1/v2's digests.
PROJECT_ID="syn75814805"  # "Challenge Submission Project-syn74274097-3597025" -- confirmed
                          # via synapseclient login as MahmoudGhoraba, 2026-08-04, the only
                          # project on this account and named directly after this challenge.

# --- Step 1: build. Context MUST be repo root, not testphase_container/ (SUBMIT.md bug,
# fixed 2026-08-04) -- the Dockerfile's COPY paths only resolve from here. ---
cd /workspace
docker build -f testphase_container/Dockerfile -t "${IMAGE_NAME}:${TAG}" .

# --- Step 2: architecture check -- expect 2.13.0+cu130 and 'sm_86' in the printed list. ---
docker run --rm --gpus=all "${IMAGE_NAME}:${TAG}" python3 -c \
  "import torch; print(torch.__version__); print(torch.cuda.get_arch_list())"

# --- Step 3: smoke test against a handful of real cases, exactly per the wiki's own
# --network none / --gpus=all form. Reuses the same 2 sample cases already verified
# end-to-end outside Docker on 2026-08-04 (see chat record) -- this time through the
# actual built image. Swap in more ValidationData_batch1 cases for a fuller check. ---
SMOKE_IN=/tmp/container_smoke_input
SMOKE_OUT=/tmp/container_smoke_output
rm -rf "$SMOKE_IN" "$SMOKE_OUT"
mkdir -p "$SMOKE_IN" "$SMOKE_OUT"
for c in BraTS-MET-01073-001 BraTS-MET-01071-002; do
  cp -r "/workspace/Dataset/BraTS_sample_dataset/$c" "$SMOKE_IN/"
done

docker run --rm --network none --gpus=all \
  --volume "$SMOKE_IN:/input:ro" \
  --volume "$SMOKE_OUT:/output:rw" \
  --memory=48g --shm-size=16g \
  "${IMAGE_NAME}:${TAG}"

echo "=== smoke test output files: ==="
ls -la "$SMOKE_OUT"

# --- Step 3b: metric-invariance check, THROUGH THE ACTUAL CONTAINER this time (this is
# the one thing that could not be verified outside Docker -- everything else about this
# fix was already confirmed in a staged /opt/algorithm mirror with real GPU inference
# against real ground truth, see chat record 2026-08-07; that confirmed the entrypoint.py
# logic itself, not the containerized image). Run the real image TWICE on the same
# GT-labeled fixture and confirm instance-level TP/FP/FN/F1 are identical between runs. ---
MI_IN=/tmp/metric_check_input
MI_GT=/tmp/metric_check_gt
MI_OUT_A=/tmp/metric_check_output_runA
MI_OUT_B=/tmp/metric_check_output_runB
rm -rf "$MI_IN" "$MI_GT" "$MI_OUT_A" "$MI_OUT_B"
mkdir -p "$MI_IN" "$MI_GT" "$MI_OUT_A" "$MI_OUT_B"
for c in BraTS-MET-01073-001 BraTS-MET-01071-002 BraTS-MET-01071-001 BraTS-MET-01073-002; do
  mkdir -p "$MI_IN/$c"
  cp /workspace/Dataset/BraTS_sample_dataset/$c/${c}-t1n.nii.gz \
     /workspace/Dataset/BraTS_sample_dataset/$c/${c}-t1c.nii.gz \
     /workspace/Dataset/BraTS_sample_dataset/$c/${c}-t2w.nii.gz \
     /workspace/Dataset/BraTS_sample_dataset/$c/${c}-t2f.nii.gz \
     "$MI_IN/$c/"
  cp /workspace/Dataset/BraTS_sample_dataset/$c/${c}-seg.nii.gz "$MI_GT/"
done

docker run --rm --network none --gpus=all \
  --volume "$MI_IN:/input:ro" --volume "$MI_OUT_A:/output:rw" \
  --memory=48g --shm-size=16g "${IMAGE_NAME}:${TAG}"
docker run --rm --network none --gpus=all \
  --volume "$MI_IN:/input:ro" --volume "$MI_OUT_B:/output:rw" \
  --memory=48g --shm-size=16g "${IMAGE_NAME}:${TAG}"

python3 /workspace/scripts/score_metric_invariance.py \
  --run-a "$MI_OUT_A" --run-b "$MI_OUT_B" --gt-dir "$MI_GT"
echo "^^^ Must print METRIC-INVARIANCE: PASSED before you continue to push. If it prints"
echo "    FAILED, stop here -- do not push or submit."

# --- Step 4: push. Requires `docker login docker.synapse.org` first, with a Synapse
# Personal Access Token as the password (not your account password) --
# generate one at https://www.synapse.org/#!PersonalAccessTokens ---
if [ "$PROJECT_ID" = "REPLACE_ME" ]; then
  echo "Set PROJECT_ID at the top of this script to your Synapse project's syn... ID before pushing." >&2
  exit 1
fi

TOKEN_FILE="/workspace/secrets/synapse_token.txt"
if [ -s "$TOKEN_FILE" ]; then
  # Synapse username confirmed via synapseclient login, 2026-08-04: MahmoudGhoraba
  docker login docker.synapse.org --username MahmoudGhoraba --password-stdin < "$TOKEN_FILE"
else
  docker login docker.synapse.org
fi
docker tag "${IMAGE_NAME}:${TAG}" "docker.synapse.org/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"
docker push "docker.synapse.org/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"

echo "=== Pushed. Remaining steps are manual on the Synapse site: ==="
echo "1. https://challenges.synapse.org/Challenges/DetailsPage/Task1?id=syn74274097#Submission"
echo "   -> Docker-submission widget -> select docker.synapse.org/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"
echo "   -> submit against evaluation queue 9619627 (Task 1 Docker queue)."
echo "2. Confirm your short paper is submitted on OpenReview with the correct Synapse team"
echo "   name -- MANDATORY. Docker submissions with no linked short paper are not evaluated."
