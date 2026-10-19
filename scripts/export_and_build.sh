#!/usr/bin/env bash
# Export every benchmarked network to ONNX and build TensorRT FP16 engines on the Jetson.
# Usage: scripts/export_and_build.sh   (idempotent: skips files that already exist)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/venv/bin/python"
TRTEXEC=/usr/src/tensorrt/bin/trtexec
CACHE="$ROOT/engines/timing.cache"
mkdir -p "$ROOT/engines"/{ffs,dav2,depthpro}

build() {  # build <onnx> -> <engine>
  local onnx=$1 engine=${1%.onnx}.engine
  [ -f "$engine" ] && return
  echo "[build] $(date +%T) $engine"
  "$TRTEXEC" --onnx="$onnx" --saveEngine="$engine" --fp16 --skipInference \
    --timingCacheFile="$CACHE" > "${engine%.engine}.build.log" 2>&1 \
    || { echo "FAILED $engine"; tail -20 "${engine%.engine}.build.log"; }
}

# Depth Anything V2 metric (Hypersim): S/B/L at the native 518 short side, S/L at twice that
# 1988x2884 is full resolution (main evaluation); export uses memory-efficient attention
for spec in vits:518x770 vitb:518x770 vitl:518x770 vits:1008x1512 vitl:1008x1512 \
            vits:1988x2884 vitb:1988x2884 vitl:1988x2884; do
  enc=${spec%%:*}; res=${spec#*:}; h=${res%x*}; w=${res#*x}
  onnx="$ROOT/engines/dav2/dav2_${enc}_${res}.onnx"
  [ -f "$onnx" ] || "$PY" "$ROOT/methods/depth_anything_v2/export_onnx.py" --encoder "$enc" --height "$h" --width "$w" --out "$onnx" > /dev/null 2>&1
  build "$onnx"
done

# Depth Pro (fixed 1536x1536)
onnx="$ROOT/engines/depthpro/depthpro_1536.onnx"
[ -f "$onnx" ] || "$PY" "$ROOT/methods/depth_pro/export_onnx.py" --out "$onnx" > /dev/null 2>&1
build "$onnx"
# Fast-FoundationStereo: two checkpoints at 3:2 input sizes.
# max_disp must cover the GT range: max d/W over all scenes except Shopvac is 0.261 (Couch),
# so use ceil32(0.27 * W) at the network input width.
# On a 64 GB Orin, 960x1440 (max_disp 416) fails to build (no TensorRT tactic for the cost-volume
# pooling) and 1280x1920 cannot even be exported (host OOM); 768x1152 is the largest that fits.
FFS="$ROOT/third_party/Fast-FoundationStereo"
[ -f "$ROOT/weights/ffs/c-ffs/model_best_bp2_serialize.pth" ] && "$PY" "$ROOT/scripts/prepare_cffs.py"
for spec in "23-36-37:8:768x1152" "20-30-48:4:768x1152" \
            "23-36-37:8:384x576" "20-30-48:4:384x576" "23-36-37:8:512x768" "20-30-48:4:512x768" \
            "c-ffs:8:768x1152" "c-ffs:8:512x768" "c-ffs:8:384x576"; do  # c-ffs: commercial checkpoint
  IFS=: read -r name iters res <<< "$spec"
  h=${res%x*}; w=${res#*x}; md=$(( ( (w * 27 / 100) + 31 ) / 32 * 32 ))
  onnx="$ROOT/engines/ffs/ffs_${name}_it${iters}_${res}.onnx"
  if [ ! -f "$onnx" ]; then
    (cd "$FFS" && "$PY" scripts/make_single_onnx.py --model_dir "$ROOT/weights/ffs/$name/model_best_bp2_serialize.pth" \
      --save_path "$ROOT/engines/ffs" --height "$h" --width "$w" --valid_iters "$iters" --max_disp "$md" \
      --onnx_name "$(basename "${onnx%.onnx}")" > /dev/null 2>&1) || { echo "EXPORT FAILED $onnx"; continue; }
  fi
  build "$onnx"
done

# Depth Anything 3 Metric-Large and MoGe-2 S/B/L at native, twice-native and full resolution.
# Both export attention as softmax(q*scale k^T) v and MoGe-2 stops at the class token (its metric-scale
# head runs outside the engine): TensorRT 10.3 mis-fuses the alternatives (see the run.py docstrings).
mkdir -p "$ROOT/engines"/{da3,moge2}
for spec in 504:350x504 1008:700x1008 0:1988x2884; do  # longest side (0 = full) : engine size
  onnx="$ROOT/engines/da3/da3metric_large_${spec#*:}.onnx"
  [ -f "$onnx" ] || "$PY" "$ROOT/methods/da3/run.py" --res "${spec%%:*}" --export "$onnx" > /dev/null 2>&1
  build "$onnx"
done
for spec in 3600:700x1008 14400:1400x2030 0:1988x2884; do  # tokens (0 = full) : engine size
  for v in vits vitb vitl; do
    onnx="$ROOT/engines/moge2/moge2_${v}_${spec#*:}.onnx"
    [ -f "$onnx" ] || "$PY" "$ROOT/methods/moge2/run.py" --variant "$v" --num_tokens "${spec%%:*}" --export "$onnx" > /dev/null 2>&1
    build "$onnx"
  done
done

# Metric3D v2 ViT-S/L at the native, 2x and full input (methods/metric3d/export_onnx.py)
mkdir -p "$ROOT/engines/metric3d"
for v in vit_small vit_large; do for spec in native:616x1064 2x:1232x2128 full:1988x2884; do
  onnx="$ROOT/engines/metric3d/metric3d_${v}_${spec#*:}.onnx"
  [ -f "$onnx" ] || "$PY" "$ROOT/methods/metric3d/export_onnx.py" --variant $v --size ${spec%%:*} > /dev/null 2>&1
  build "$onnx"
done; done

# UniDepth V2 S/B/L with the camera as rays, at the sizes its pixel budget picks for Middlebury
# (methods/unidepth/run.py reports them): native 644x938, 2x 1302x1890, full 1988x2884
mkdir -p "$ROOT/engines/unidepth"
for v in vits14 vitb14 vitl14; do for res in 644x938 1302x1890 1988x2884; do
  onnx="$ROOT/engines/unidepth/unidepthv2_${v}_${res}.onnx"
  [ -f "$onnx" ] || "$PY" "$ROOT/methods/unidepth/export_onnx.py" --variant $v --shape ${res%x*} ${res#*x} > /dev/null 2>&1
  build "$onnx"
done; done

# NVIDIA ESS / Light ESS (NGC nvidia/isaac/dnn_stereo_disparity v4.1.0_onnx, NVIDIA TAO model EULA):
# fixed-size ONNX with custom layers from the bundled TensorRT plugin library
ESS="$ROOT/weights/ess/dnn_stereo_disparity_v4.1.0_onnx"
mkdir -p "$ROOT/engines/ess"
for m in ess light_ess; do
  engine="$ROOT/engines/ess/$m.engine"
  [ -f "$engine" ] || "$TRTEXEC" --onnx="$ESS/$m.onnx" --saveEngine="$engine" --fp16 --skipInference \
    --staticPlugins="$ESS/plugins/aarch64/ess_plugins.so" > "${engine%.engine}.build.log" 2>&1 || echo "FAILED $engine"
done

echo ALLDONE
