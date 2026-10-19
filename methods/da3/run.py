"""Depth Anything 3 metric (DA3Metric-Large) on Middlebury 2014: predictions and ONNX export.

The network sees the image resized to its native processing resolution (longest side 504, both sides
rounded to multiples of 14: 350x504 for Middlebury's 3:2 images), as in the official
InputProcessor ("upper_bound_resize"). Its output becomes metric depth via
    depth_m = f_proc * output / 300,
where f_proc is the focal length in pixels of the processed image (official README); we use the
ground-truth focal length scaled to the processed size. Depth is upsampled bilinearly to full resolution.

Usage:
  python methods/da3/run.py --save_preds                    # FP32 PyTorch (accuracy reference)
  python methods/da3/run.py --export engines/da3/da3metric_large_350x504.onnx
  python methods/da3/run.py --engine engines/da3/da3metric_large_350x504.engine --save_preds
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
from common.metrics import all_metrics  # noqa: E402
from common.middlebury import SCENES, load_scene  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
PATCH = 14


def proc_size(H, W, res=504):
    """Official upper_bound_resize: longest side -> res, then each side to the nearest multiple of 14."""
    s = res / max(H, W)
    h, w = int(round(H * s)), int(round(W * s))

    def near(x):
        down = (x // PATCH) * PATCH
        return down + PATCH if abs(down + PATCH - x) <= abs(x - down) else down
    return near(h), near(w)


def preprocess(bgr, h, w):
    rgb = cv2.cvtColor(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def load_model(weights):
    import torch
    from safetensors.torch import load_file
    sys.path.insert(0, os.path.join(ROOT, "third_party/Depth-Anything-3/src"))
    from depth_anything_3.cfg import create_object
    cfg = json.load(open(os.path.join(weights, "config.json")))["config"]
    net = create_object(cfg)
    sd = load_file(os.path.join(weights, "model.safetensors"))
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert not missing, missing[:10]
    return net.eval().cuda(), torch


def _export_attention_forward(self, x, pos=None, attn_mask=None):
    """DA3 DINOv2 Attention.forward (no attention mask) with common.export_attention.classic_attention."""
    from common.export_attention import classic_attention
    B, N, C = x.shape
    q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k = self.q_norm(q), self.k_norm(k)
    if self.rope is not None and pos is not None:
        q, k = self.rope(q, pos), self.rope(k, pos)
    x = classic_attention(q, k, v, self.scale)
    return self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, N, C)))


class Wrapper:
    """(1,3,h,w) normalised RGB -> (1,h,w) network output (relative-to-focal depth)."""

    @staticmethod
    def make(net, torch):
        class W(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = net

            def forward(self, x):
                return self.net(x[:, None])["depth"][:, 0]
        return W().eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=os.path.join(ROOT, "weights/da3/DA3METRIC-LARGE"))
    p.add_argument("--data", default=os.path.join(ROOT, "data/middlebury2014"))
    p.add_argument("--res", type=int, default=504, help="longest side of the processed image (native: 504; 0 = each image at its own size)")
    p.add_argument("--engine", help="TensorRT engine instead of FP32 PyTorch")
    p.add_argument("--export", help="write ONNX at the processed size of the Middlebury images and exit")
    p.add_argument("--save_preds", action="store_true")
    p.add_argument("--save_native", action="store_true",
                   help="also save the depth at the processed resolution (results/preds_native/)")
    p.add_argument("--scenes", nargs="*", default=SCENES)
    a = p.parse_args()

    if a.engine:
        from common.trt_runner import TRTRunner
        runner = TRTRunner(a.engine)
        _, _, h, w = runner.input_shape()
        infer = lambda x: next(iter(runner(x).values())).reshape(h, w)  # noqa: E731
        tag = os.path.splitext(os.path.basename(a.engine))[0]
    else:
        net, torch = load_model(a.weights)
        model = Wrapper.make(net, torch).cuda()
        h, w = proc_size(1988, 2884, a.res or 2884)  # all Middlebury 2014 scenes are ~1988x2884 (3:2)
        if a.export:
            # classic, query-blocked attention for TensorRT (common/export_attention.py); same math
            from depth_anything_3.model.dinov2.layers import attention as da3_attention
            da3_attention.Attention.forward = _export_attention_forward
            os.makedirs(os.path.dirname(os.path.abspath(a.export)), exist_ok=True)
            with torch.no_grad():
                torch.onnx.export(model, torch.randn(1, 3, h, w, device="cuda"), a.export, input_names=["image"],
                                  output_names=["depth"], opset_version=17, dynamo=False)
            print("wrote", a.export, (h, w))
            return

        def infer(x):
            with torch.no_grad():
                return model(torch.from_numpy(x).cuda()).float().cpu().numpy().reshape(x.shape[2:])
        tag = f"da3metric_large_{h}x{w}_fp32"

    rows = []
    for name in a.scenes:
        s = load_scene(a.data, name)
        H, W = s["left"].shape[:2]
        # official pipeline: each image at its own processed size (Bicycle1 gives 336x504);
        # a fixed-size engine takes every image at its input size
        hp, wp = (h, w) if a.engine else proc_size(H, W, a.res or max(H, W))
        out = infer(preprocess(s["left"], hp, wp))
        f_proc = s["f"] * 0.5 * (wp / W + hp / H)  # mean of fx, fy in processed pixels
        depth = cv2.resize(out * f_proc / 300.0, (W, H), interpolation=cv2.INTER_LINEAR)
        m = all_metrics(None, depth, s, mono=True)
        m.update(scene=name, method=tag)
        rows.append(m)
        print(name, {k: round(v, 3) for k, v in m.items() if k in ("absrel", "aligned_absrel", "median_scale")})
        if a.save_preds:
            d = os.path.join(ROOT, "results/preds", tag)
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, f"{name}.npy"), depth.astype(np.float32))
        if a.save_native:
            d = os.path.join(ROOT, "results/preds_native", tag)
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, f"{name}.npy"), (out * f_proc / 300.0).astype(np.float32))
    os.makedirs(os.path.join(ROOT, "results/accuracy"), exist_ok=True)
    json.dump(rows, open(os.path.join(ROOT, "results/accuracy", f"{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
