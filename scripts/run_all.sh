#!/usr/bin/env bash
# The paper's pipeline: accuracy + speed of every method at every input size, scoring on the 10
# Middlebury scenes with disparity < 256 px, resolution scaling, depth discontinuities, tables and
# figures. Needs the engines from export_and_build.sh and libSGM from methods/libsgm/build_libsgm.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY=venv/bin/python

# ---- networks: accuracy -------------------------------------------------------------------------
# --save_native also keeps each output at the network's own resolution (edge analysis)
for e in engines/ffs/*.engine;  do $PY scripts/eval_networks.py --kind ffs  --engine "$e" --save_preds --save_native; done
for e in engines/dav2/*.engine; do $PY scripts/eval_networks.py --kind dav2 --engine "$e" --save_preds --save_native; done
# the FP16 engine of the Small model is numerically unreliable at full resolution: FP32 accuracy
$PY scripts/eval_networks.py --kind dav2 --engine engines/dav2/dav2_vits_1988x2884.engine --torch_fp32 --save_preds
# Depth Pro: FP32 accuracy, with the true focal length and with its own estimate
$PY scripts/eval_networks.py --kind depthpro --engine engines/depthpro/depthpro_1536.engine --torch_fp32 --save_preds
$PY scripts/eval_networks.py --kind depthpro --engine engines/depthpro/depthpro_1536.engine --torch_fp32 --estimate_focal --save_preds
# Depth Anything 3 Metric-L and MoGe-2: FP32 accuracy (official pre/post-processing), FP16 engine check
for r in 504 1008 0; do $PY methods/da3/run.py --res $r --save_preds --save_native; done  # longest side, 0 = full
for n in 3600:700x1008 14400:1400x2030 0:1988x2884; do for v in vits vitb vitl; do  # tokens, 0 = full
  $PY methods/moge2/run.py --variant $v --num_tokens ${n%%:*} --save_preds
  $PY methods/moge2/run.py --variant $v --num_tokens ${n%%:*} --gt_fov --save_preds --save_native
  $PY methods/moge2/run.py --variant $v --num_tokens ${n%%:*} --check_engine engines/moge2/moge2_${v}_${n#*:}.engine
done; done
# Metric3D v2 (ViT-S/L, true focal length) and UniDepth V2 (S/B/L, true intrinsics): FP32 accuracy
for sz in native 2x full; do
  for v in vit_small vit_large; do $PY methods/metric3d/run.py --variant $v --size $sz --save_preds --save_native; done
  for v in vits14 vitb14 vitl14; do $PY methods/unidepth/run.py --variant $v --size $sz --save_preds --save_native; done
done
$PY scripts/check_new_engines.py   # FP16 engines of Metric3D v2 and UniDepth V2 vs FP32
# NVIDIA ESS / Light ESS stereo (TensorRT engines with the ESS plugin library)
for m in ess light_ess; do $PY methods/ess/run.py --model $m --save_preds --save_native; done

# ---- networks: speed and power (nothing else on the GPU) ---------------------------------------
$PY scripts/bench_speed.py --engines engines/ffs/ffs_{23-36-37,20-30-48}_*.engine engines/{dav2,depthpro}/*.engine   # -> networks.json
$PY scripts/bench_speed.py --engines engines/dav2/dav2_vit{s,b,l}_1988x2884.engine --out results/speed/dav2_full.json
$PY scripts/bench_speed.py --engines engines/da3/da3metric_large_350x504.engine \
  engines/moge2/moge2_vit{s,b,l}_700x1008.engine --out results/speed/mono_v3.json
$PY scripts/bench_speed.py --engines engines/da3/da3metric_large_1988x2884.engine \
  engines/moge2/moge2_vit{s,b,l}_{1400x2030,1988x2884}.engine --out results/speed/mono_hires.json
$PY scripts/bench_speed.py --engines engines/da3/da3metric_large_700x1008.engine --out results/speed/mono_hires2.json
$PY scripts/bench_speed.py --engines engines/ffs/ffs_{23-36-37,20-30-48}_*_{384x576,512x768}.engine --out results/speed/ffs_scaling.json
$PY scripts/bench_speed.py --engines engines/ffs/ffs_c-ffs_*.engine --out results/speed/cffs.json
$PY scripts/bench_speed.py --engines engines/metric3d/*.engine engines/unidepth/*.engine --out results/speed/new_methods.json
$PY scripts/bench_speed.py --engines engines/ess/*.engine --out results/speed/ess_engines.json \
  --plugins weights/ess/dnn_stereo_disparity_v4.1.0_onnx/plugins/aarch64/ess_plugins.so

# ---- libSGM: P1/P2 tuning per resolution, accuracy, speed ---------------------------------------
$PY scripts/score_v2.py > /dev/null  # writes results/v2/scene_sets.json (the 10 scenes) used by tune.py
for s in 1 2 4; do  # full, half, quarter resolution
  $PY methods/libsgm/tune.py  --scale $s --paths 4
  $PY methods/libsgm/run.py   --scale $s --paths 4 --tuned --save_preds
  $PY methods/libsgm/bench.py --scale $s --paths 4 --tuned
done

# ---- scoring, tables, figures --------------------------------------------------------------------
$PY scripts/score_v2.py
$PY scripts/make_figures_v2.py
$PY scripts/make_dataset_table.py
$PY scripts/make_depth_figure.py
$PY scripts/make_depth_figure.py --set lowres
$PY scripts/make_scaling_figure.py   # resolution scaling figure + scale-drift table
$PY scripts/make_recommendations.py  # best stereo / monocular configuration per frame-rate budget
$PY scripts/edge_analysis.py         # depth discontinuities: flying points + point-cloud figure
