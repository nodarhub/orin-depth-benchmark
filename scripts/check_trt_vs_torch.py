"""Sanity check: TensorRT FP16 engine output vs. FP32 PyTorch reference on one Middlebury image.

Reports median relative difference of the predicted depth/disparity. Large ViTs can
overflow in FP16, so a result is only trusted if the difference is small (<1-2%).
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from common.middlebury import load_scene  # noqa: E402
from common.trt_runner import TRTRunner  # noqa: E402
from scripts.eval_networks import imagenet_norm  # noqa: E402


def torch_model(kind, engine):
    name = os.path.basename(engine)
    if kind == "dav2":
        sys.path.insert(0, os.path.join(ROOT, "third_party/Depth-Anything-V2/metric_depth"))
        from depth_anything_v2.dpt import DepthAnythingV2
        from methods.depth_anything_v2.export_onnx import CONFIGS, _sdpa_forward
        from depth_anything_v2.dinov2_layers import attention as _att
        _att.Attention.forward = _sdpa_forward  # same math; avoids N x N attention memory at full res
        enc = name.split("_")[1]
        m = DepthAnythingV2(**CONFIGS[enc], max_depth=20.0)
        m.load_state_dict(torch.load(os.path.join(ROOT, f"weights/dav2/depth_anything_v2_metric_hypersim_{enc}.pth"),
                                     map_location="cpu"))
        return m.eval().cuda()
    if kind == "depthpro":
        sys.path.insert(0, os.path.join(ROOT, "third_party/ml-depth-pro/src"))
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT, create_model_and_transforms
        cfg = DEFAULT_MONODEPTH_CONFIG_DICT
        cfg.checkpoint_uri = os.path.join(ROOT, "weights/depthpro/depth_pro.pt")
        m, _ = create_model_and_transforms(cfg, device=torch.device("cuda"))
        return m.eval()
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True)
    p.add_argument("--kind", choices=["dav2", "depthpro"], required=True)
    p.add_argument("--scene", default="Adirondack")
    a = p.parse_args()
    s = load_scene(os.path.join(ROOT, "data/middlebury2014"), a.scene)
    r = TRTRunner(a.engine)
    if a.kind == "dav2":
        _, _, h, w = r.input_shape()
        x = imagenet_norm(s["left"], h, w)
    else:
        rgb = cv2.cvtColor(cv2.resize(s["left"], (1536, 1536), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = ((rgb.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
    trt_out = r(x)
    trt_y = list(trt_out.values())[0].squeeze()
    del r
    torch.cuda.empty_cache()
    m = torch_model(a.kind, a.engine)
    with torch.no_grad():
        ref = m(torch.from_numpy(x).cuda())
    ref_y = (ref[0] if isinstance(ref, tuple) else ref).float().cpu().numpy().squeeze()
    rel = np.abs(trt_y - ref_y) / np.maximum(np.abs(ref_y), 1e-6)
    print(f"{os.path.basename(a.engine)}: median rel diff {np.median(rel)*100:.3f}%  p99 {np.percentile(rel, 99)*100:.3f}%  "
          f"finite {np.isfinite(trt_y).mean()*100:.1f}%")
    if a.kind == "depthpro":
        print("fov trt", trt_out["fov_deg"].ravel(), "torch", ref[1].float().cpu().numpy().ravel())


if __name__ == "__main__":
    main()
