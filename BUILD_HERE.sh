#!/usr/bin/env bash
# Self-contained build+test+push script. Run this FROM WHEREVER YOU EXTRACTED THE
# ARCHIVE -- unlike the old run_submission.sh, this one does NOT assume /workspace;
# it uses its own location as the build context root.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE_NAME="bratsmet2026-task1"
TAG="v3"
PROJECT_ID="syn75814805"

echo "=== Step 1: build (context = $(pwd)) ==="
docker build -f testphase_container/Dockerfile -t "${IMAGE_NAME}:${TAG}" .

echo "=== Step 2: architecture check -- expect 2.13.0+cu130 and 'sm_86' in the list ==="
docker run --rm --gpus=all "${IMAGE_NAME}:${TAG}" python3 -c \
  "import torch; print(torch.__version__); print(torch.cuda.get_arch_list())"

echo "=== Step 3: smoke test (4 real sample cases) ==="
SMOKE_IN="$(pwd)/tmp_smoke_input"
SMOKE_OUT="$(pwd)/tmp_smoke_output"
rm -rf "$SMOKE_IN" "$SMOKE_OUT"
mkdir -p "$SMOKE_IN" "$SMOKE_OUT"
for c in BraTS-MET-01073-001 BraTS-MET-01071-002 BraTS-MET-01071-001 BraTS-MET-01073-002; do
  cp -r "Dataset/BraTS_sample_dataset/$c" "$SMOKE_IN/"
done
docker run --rm --network none --gpus=all \
  --volume "$SMOKE_IN:/input:ro" --volume "$SMOKE_OUT:/output:rw" \
  --memory=48g --shm-size=16g "${IMAGE_NAME}:${TAG}"
echo "smoke test output files:"; ls -la "$SMOKE_OUT"

echo "=== Step 3b: metric-invariance check (run twice, compare against real GT) ==="
MI_OUT_A="$(pwd)/tmp_mi_output_A"
MI_OUT_B="$(pwd)/tmp_mi_output_B"
MI_GT="$(pwd)/tmp_mi_gt"
rm -rf "$MI_OUT_A" "$MI_OUT_B" "$MI_GT"
mkdir -p "$MI_OUT_A" "$MI_OUT_B" "$MI_GT"
for c in BraTS-MET-01073-001 BraTS-MET-01071-002 BraTS-MET-01071-001 BraTS-MET-01073-002; do
  cp "Dataset/BraTS_sample_dataset/$c/${c}-seg.nii.gz" "$MI_GT/"
done
docker run --rm --network none --gpus=all \
  --volume "$SMOKE_IN:/input:ro" --volume "$MI_OUT_A:/output:rw" \
  --memory=48g --shm-size=16g "${IMAGE_NAME}:${TAG}"
docker run --rm --network none --gpus=all \
  --volume "$SMOKE_IN:/input:ro" --volume "$MI_OUT_B:/output:rw" \
  --memory=48g --shm-size=16g "${IMAGE_NAME}:${TAG}"
python3 scripts/score_metric_invariance.py --run-a "$MI_OUT_A" --run-b "$MI_OUT_B" --gt-dir "$MI_GT"
echo "^^^ Must say METRIC-INVARIANCE: PASSED. If it says FAILED, STOP -- do not push."

echo "=== Step 4: push ==="
read -p "Metric-invariance passed and you want to push+submit now? [y/N] " CONFIRM
if [ "$CONFIRM" != "y" ]; then echo "Stopping before push."; exit 0; fi

docker login docker.synapse.org
docker tag "${IMAGE_NAME}:${TAG}" "docker.synapse.org/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"
docker push "docker.synapse.org/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"

echo "=== Pushed. Tell Claude it's pushed -- it will register the Synapse submission via API. ==="
