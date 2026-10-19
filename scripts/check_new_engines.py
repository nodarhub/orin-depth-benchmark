"""FP16 TensorRT engines of Metric3D v2 and UniDepth V2 against their FP32 PyTorch graphs on one scene,
with identical inputs: median and 99th-percentile relative depth difference.

  python scripts/check_new_engines.py --scene Adirondack
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from common.middlebury import load_scene  # noqa: E402
from common.trt_runner import TRTRunner  # noqa: E402


def report(name, ref, out):
    ref, out = ref.astype(np.float64).ravel(), out.astype(np.float64).ravel()
    ok = np.isfinite(out) & (np.abs(ref) > 1e-6)
    rel = np.abs(out[ok] - ref[ok]) / np.abs(ref[ok])
    print(f"{name}: median rel diff {100 * np.median(rel):.3f}%  p99 {100 * np.percentile(rel, 99):.3f}%  "
          f"finite {100 * np.isfinite(out).mean():.1f}%", flush=True)


def check_metric3d(s):
    sys.path.insert(0, os.path.join(ROOT, "methods/metric3d"))
    import export_onnx as M
    for engine in sorted(glob.glob(os.path.join(ROOT, "engines/metric3d/*.engine"))):
        tag = os.path.basename(engine)[:-7]
        variant = "vit_small" if "vit_small" in tag else "vit_large"
        h, w = map(int, tag.split("_")[-1].split("x"))
        rgb = cv2.resize(cv2.cvtColor(s["left"], cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1))[None].cuda()
        model = M.Export(M.update_vit_sampling(M.load_model(variant))).cuda().eval()
        with torch.no_grad():
            ref = model(x).float().cpu().numpy()
        del model
        torch.cuda.empty_cache()
        out = list(TRTRunner(engine)(x).values())[0]
        report(tag, ref, out)


def check_unidepth(s):
    import types
    sys.modules.setdefault("wandb", types.ModuleType("wandb"))
    from unidepth.models.unidepthv2.export import UniDepthV2ONNXcam
    for engine in sorted(glob.glob(os.path.join(ROOT, "engines/unidepth/*.engine"))):
        tag = os.path.basename(engine)[:-7]
        variant = tag.split("_")[1]
        h, w = map(int, tag.split("_")[-1].split("x"))
        H, W = s["left"].shape[:2]
        rgb = cv2.resize(cv2.cvtColor(s["left"], cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_AREA)
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        x = torch.from_numpy(((rgb / 255.0 - mean) / std).astype(np.float32).transpose(2, 0, 1))[None].cuda()
        K = s["calib"]["cam0"].astype(np.float64).copy()
        K[0] *= w / W
        K[1] *= h / H
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        r = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)])
        r /= np.linalg.norm(r, axis=0, keepdims=True)
        rays = torch.from_numpy(r.astype(np.float32))[None].cuda()
        model = UniDepthV2ONNXcam.from_pretrained(f"lpiccinelli/unidepth-v2-{variant}",
                                                  cache_dir=os.path.join(ROOT, "weights/unidepth")).cuda().eval()
        with torch.no_grad():
            ref = model(x, rays)[0][:, 2].float().cpu().numpy()
        del model
        torch.cuda.empty_cache()
        out = TRTRunner(engine)(x, rays)["pts_3d"][:, 2]
        report(tag, ref, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="Adirondack")
    p.add_argument("--only", choices=["metric3d", "unidepth"], default=None)
    a = p.parse_args()
    s = load_scene(os.path.join(ROOT, "data/middlebury2014"), a.scene)
    if a.only in (None, "metric3d"):
        check_metric3d(s)
    if a.only in (None, "unidepth"):
        check_unidepth(s)


if __name__ == "__main__":
    main()
