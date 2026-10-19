"""ONNX export of UniDepth V2 with the camera given as rays (the official UniDepthV2ONNXcam,
unidepth/models/unidepthv2/export.py) at a fixed processed size, for TensorRT FP16 timing. Attention
is exported in the classic form (common/export_attention.py). Accuracy is measured with FP32 PyTorch
(run.py).

  python methods/unidepth/export_onnx.py --variant vitl14 --shape 644 938
"""
import argparse
import os
import sys
import types

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
sys.modules.setdefault("wandb", types.ModuleType("wandb"))
from unidepth.models.unidepthv2.export import UniDepthV2ONNXcam  # noqa: E402
from common.export_attention import patch_sdpa  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["vits14", "vitb14", "vitl14"], default="vitl14")
    p.add_argument("--shape", type=int, nargs=2, required=True, help="processed height width (multiples of 14)")
    a = p.parse_args()
    h, w = a.shape
    model = UniDepthV2ONNXcam.from_pretrained(f"lpiccinelli/unidepth-v2-{a.variant}",
                                              cache_dir=os.path.join(ROOT, "weights/unidepth"))
    patch_sdpa()
    model = model.cuda().eval()
    out = os.path.join(ROOT, "engines/unidepth", f"unidepthv2_{a.variant}_{h}x{w}.onnx")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    x, rays = torch.rand(1, 3, h, w, device="cuda"), torch.rand(1, 3, h, w, device="cuda")
    with torch.no_grad():
        torch.onnx.export(model, (x, rays), out, input_names=["rgbs", "rays"],
                          output_names=["pts_3d", "confidence", "intrinsics"], opset_version=17, dynamo=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
