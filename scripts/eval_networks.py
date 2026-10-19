"""Run TensorRT engines on Middlebury 2014 (full-resolution GT) and score them.

Every method is evaluated on the same full-resolution GT. Networks run at their
engine input size; outputs are bilinearly upsampled to full resolution
(disparities are also rescaled by W_full / W_in).

Usage:
  python scripts/eval_networks.py --engine engines/ffs/ffs_23-36-37_it8_640x960.engine --kind ffs
  python scripts/eval_networks.py --engine engines/dav2/dav2_vitl_518x770.engine --kind dav2
  python scripts/eval_networks.py --engine engines/depthpro/depthpro_1536.engine --kind depthpro [--estimate_focal]
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from common.metrics import all_metrics  # noqa: E402
from common.middlebury import SCENES, load_scene  # noqa: E402
from common.trt_runner import TRTRunner  # noqa: E402

MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


def imagenet_norm(bgr, h, w):
    rgb = cv2.cvtColor(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return ((rgb.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]


NATIVE = {}  # last output at the network's own resolution (disparity in full-resolution pixels, or depth)


def run_ffs(runner, s):
    _, _, h, w = runner.input_shape()
    H, W = s["left"].shape[:2]
    out = runner(imagenet_norm(s["left"], h, w), imagenet_norm(s["right"], h, w))
    disp = next(iter(out.values())).reshape(h, w)
    NATIVE["pred"] = disp * (W / w)
    disp = cv2.resize(disp, (W, H), interpolation=cv2.INTER_LINEAR) * (W / w)
    return disp, None


class TorchDAV2:
    """FP32 PyTorch Depth Anything V2 with the TRTRunner interface (accuracy reference when an FP16
    engine is numerically unreliable). Uses memory-efficient attention (identical math)."""

    def __init__(self, engine, shape):
        import torch
        sys.path.insert(0, os.path.join(ROOT, "third_party/Depth-Anything-V2/metric_depth"))
        from depth_anything_v2.dpt import DepthAnythingV2
        from depth_anything_v2.dinov2_layers import attention as att
        from methods.depth_anything_v2.export_onnx import CONFIGS, _sdpa_forward
        att.Attention.forward = _sdpa_forward
        enc = os.path.basename(engine).split("_")[1]
        m = DepthAnythingV2(**CONFIGS[enc], max_depth=20.0)
        m.load_state_dict(torch.load(os.path.join(ROOT, f"weights/dav2/depth_anything_v2_metric_hypersim_{enc}.pth"),
                                     map_location="cpu"))
        self.torch, self.m, self.shape = torch, m.eval().cuda(), shape

    def input_shape(self):
        return self.shape

    def __call__(self, x):
        with self.torch.no_grad():
            return {"depth": self.m(self.torch.from_numpy(x).cuda()).float().cpu().numpy()}


def run_dav2(runner, s):
    _, _, h, w = runner.input_shape()
    H, W = s["left"].shape[:2]
    depth = runner(imagenet_norm(s["left"], h, w))["depth"].reshape(h, w)
    NATIVE["pred"] = depth
    return None, cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)


class TorchDepthPro:
    """FP32 PyTorch Depth Pro with the same interface as TRTRunner (accuracy reference:
    the FP16 engine drifts ~2% from FP32, so accuracy is reported from FP32)."""

    def __init__(self):
        import torch
        sys.path.insert(0, os.path.join(ROOT, "third_party/ml-depth-pro/src"))
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT, create_model_and_transforms
        cfg = DEFAULT_MONODEPTH_CONFIG_DICT
        cfg.checkpoint_uri = os.path.join(ROOT, "weights/depthpro/depth_pro.pt")
        self.torch = torch
        self.m, _ = create_model_and_transforms(cfg, device=torch.device("cuda"))
        self.m.eval()

    def __call__(self, x):
        with self.torch.no_grad():
            c, fov = self.m(self.torch.from_numpy(x).cuda())
        return {"canonical_inverse_depth": c.float().cpu().numpy(), "fov_deg": fov.float().cpu().numpy()}


def run_depthpro(runner, s, estimate_focal):
    H, W = s["left"].shape[:2]
    rgb = cv2.cvtColor(cv2.resize(s["left"], (1536, 1536), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    out = runner(np.ascontiguousarray(x.transpose(2, 0, 1)[None]))
    canon = out["canonical_inverse_depth"].reshape(1536, 1536)
    fov = float(out["fov_deg"].ravel()[0])
    f_px = 0.5 * W / np.tan(0.5 * np.deg2rad(fov)) if estimate_focal else s["f"]
    inv = cv2.resize(canon * (W / f_px), (W, H), interpolation=cv2.INTER_LINEAR)
    depth = 1.0 / np.clip(inv, 1e-4, 1e4)
    return None, depth, {"fov_deg": fov, "f_px_used": float(f_px), "f_px_true": float(s["f"])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True)
    p.add_argument("--kind", choices=["ffs", "dav2", "depthpro"], required=True)
    p.add_argument("--data", default=os.path.join(ROOT, "data/middlebury2014"))
    p.add_argument("--out_dir", default=os.path.join(ROOT, "results/accuracy"))
    p.add_argument("--estimate_focal", action="store_true", help="Depth Pro: use its own FOV estimate")
    p.add_argument("--save_preds", action="store_true")
    p.add_argument("--save_native", action="store_true",
                   help="also save the output at the network's own resolution (results/preds_native/)")
    p.add_argument("--torch_fp32", action="store_true", help="Depth Pro / Depth Anything V2: FP32 PyTorch instead of the engine")
    p.add_argument("--scenes", nargs="*", default=SCENES)
    a = p.parse_args()
    if a.torch_fp32 and a.kind == "dav2":
        h, w = map(int, os.path.splitext(os.path.basename(a.engine))[0].split("_")[-1].split("x"))
        runner = TorchDAV2(a.engine, (1, 3, h, w))
    elif a.torch_fp32:
        runner = TorchDepthPro()
    else:
        runner = TRTRunner(a.engine)
    tag = (os.path.splitext(os.path.basename(a.engine))[0] + ("_fp32" if a.torch_fp32 else "")
           + ("_estf" if a.estimate_focal else ""))
    os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    for name in a.scenes:
        s = load_scene(a.data, name)
        t0 = time.perf_counter()
        extra = {}
        if a.kind == "ffs":
            disp, depth = run_ffs(runner, s)
        elif a.kind == "dav2":
            disp, depth = run_dav2(runner, s)
        else:
            disp, depth, extra = run_depthpro(runner, s, a.estimate_focal)
        e2e_ms = (time.perf_counter() - t0) * 1e3
        m = all_metrics(disp, depth, s, mono=a.kind != "ffs")
        m.update(scene=name, method=tag, e2e_python_ms=e2e_ms, **extra)
        rows.append(m)
        print(name, {k: round(v, 3) for k, v in m.items() if k in ("bad2", "absrel", "d1", "aligned_absrel", "aligned_bad2")})
        if a.save_preds:
            pd = os.path.join(ROOT, "results/preds", tag)
            os.makedirs(pd, exist_ok=True)
            arr = disp if disp is not None else depth
            np.save(os.path.join(pd, f"{name}.npy"), arr.astype(np.float32))
        if a.save_native and "pred" in NATIVE:
            pd = os.path.join(ROOT, "results/preds_native", tag)
            os.makedirs(pd, exist_ok=True)
            np.save(os.path.join(pd, f"{name}.npy"), NATIVE.pop("pred").astype(np.float32))
    with open(os.path.join(a.out_dir, f"{tag}.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
