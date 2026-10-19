#!/usr/bin/env bash
# One-time environment setup on a Jetson AGX Orin (JetPack 6.2 / L4T R36.4, CUDA 12.6, TensorRT 10.3).
# Creates ./venv (with system site-packages for the TensorRT Python bindings), clones the
# third-party model repos at pinned commits, and downloads all weights (see the licences in README.md).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 -m venv --system-site-packages venv
venv/bin/pip install -U pip
# Jetson CUDA 12.6 PyTorch build; pinned so later installs cannot replace it with a generic wheel.
venv/bin/pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple torch==2.8.0 torchvision==0.23.0
venv/bin/pip install -c constraints.txt -r requirements.txt
# MoGe-2 geometry utilities (pinned as in MoGe's pyproject; --no-deps keeps the Jetson numpy/torch)
venv/bin/pip install --no-deps -c constraints.txt \
  "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37"

mkdir -p third_party && cd third_party
clone() {  # clone <repo> <commit>
  local dir; dir=$(basename "$1")
  [ -d "$dir" ] || git clone -q "https://github.com/$1" "$dir"
  git -C "$dir" checkout -q "$2"
}
while read -r repo commit; do clone "$repo" "$commit"; done < "$ROOT/third_party.lock"
cd "$ROOT"
# Metric3D v2 (config loading only) and UniDepth V2; --no-deps keeps the Jetson numpy/torch
venv/bin/pip install --no-deps -c constraints.txt mmengine==0.10.7 yapf==0.43.0 platformdirs rich termcolor tomli
venv/bin/pip install --no-deps -c constraints.txt -e third_party/UniDepth

mkdir -p weights && cd weights
# Fast-FoundationStereo research checkpoints (NVlabs Google Drive folder from the official README)
[ -d ffs ] || ../venv/bin/gdown -q --folder https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap -O ffs
# commercially licensed C-Fast-FoundationStereo (NVIDIA Open Model Agreement), same architecture as 23-36-37
../venv/bin/hf download nvidia/c-fast-foundationstereo model_best_bp2_serialize.pth cfg.yaml --local-dir ffs/c-ffs
for s in Small:vits Base:vitb Large:vitl; do
  ../venv/bin/hf download "depth-anything/Depth-Anything-V2-Metric-Hypersim-${s%%:*}" \
    "depth_anything_v2_metric_hypersim_${s##*:}.pth" --local-dir dav2
done
../venv/bin/hf download apple/DepthPro depth_pro.pt --local-dir depthpro
../venv/bin/hf download depth-anything/DA3METRIC-LARGE --local-dir da3/DA3METRIC-LARGE
for v in vits vitb vitl; do
  ../venv/bin/hf download "Ruicheng/moge-2-$v-normal" model.pt --local-dir "moge2/moge-2-$v-normal"
done
# Metric3D v2 and UniDepth V2 weights are downloaded on first use (methods/metric3d, methods/unidepth).
# NVIDIA ESS / Light ESS from NGC: distributed under the NVIDIA TAO model EULA, which downloading accepts
# (https://developer.download.nvidia.com/licenses/tao_toolkit_21-08_models_eula.pdf)
mkdir -p ess
if [ ! -d ess/dnn_stereo_disparity_v4.1.0_onnx ]; then
  curl -sL -o ess/ess.tar.gz "https://api.ngc.nvidia.com/v2/models/nvidia/isaac/dnn_stereo_disparity/versions/4.1.0_onnx/files/dnn_stereo_disparity_v4.1.0_onnx.tar.gz"
  tar -xzf ess/ess.tar.gz -C ess && rm ess/ess.tar.gz
fi
