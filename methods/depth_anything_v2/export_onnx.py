"""Export Depth Anything V2 (metric, Hypersim/indoor) to ONNX at a fixed resolution.

Input:  image (1,3,H,W) float32, ImageNet-normalised RGB, H and W multiples of 14
Output: depth (1,H,W) float32, metres
"""
import argparse
import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, os.path.join(ROOT, "third_party/Depth-Anything-V2/metric_depth"))
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402
from depth_anything_v2.dinov2_layers import attention as _dinov2_attention  # noqa: E402


def _sdpa_forward(self, x):
    """Same computation as DINOv2's Attention.forward (softmax(q k^T / sqrt(d)) v), via
    scaled_dot_product_attention so tracing at high resolution does not materialise the full
    N x N attention matrix (29k tokens at full Middlebury resolution)."""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    x = torch.nn.functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
    return self.proj(x.transpose(1, 2).reshape(B, N, C))

CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", choices=CONFIGS, default="vitl")
    p.add_argument("--height", type=int, default=518)
    p.add_argument("--width", type=int, default=770)
    p.add_argument("--max_depth", type=float, default=20.0)  # Hypersim checkpoints
    p.add_argument("--weights_dir", default=os.path.join(ROOT, "weights/dav2"))
    p.add_argument("--out", required=True)
    a = p.parse_args()
    assert a.height % 14 == 0 and a.width % 14 == 0
    _dinov2_attention.Attention.forward = _sdpa_forward
    model = DepthAnythingV2(**CONFIGS[a.encoder], max_depth=a.max_depth)
    sd = torch.load(os.path.join(a.weights_dir, f"depth_anything_v2_metric_hypersim_{a.encoder}.pth"),
                    map_location="cpu")
    model.load_state_dict(sd)
    model.eval().cuda()
    x = torch.randn(1, 3, a.height, a.width, device="cuda")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(model, x, a.out, input_names=["image"], output_names=["depth"],
                          opset_version=17, dynamo=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
