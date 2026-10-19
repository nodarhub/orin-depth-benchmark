"""ONNX export of Metric3D v2 (ViT) at a fixed input size, for TensorRT FP16 timing.
Follows the official onnx/metric3d_onnx_export.py (normalisation inside the graph, bilinear instead of
bicubic position-embedding interpolation for TensorRT) and exports attention in the classic form
(common/export_attention.py). Accuracy is measured with FP32 PyTorch (run.py).

  python methods/metric3d/export_onnx.py --variant vit_large --size native
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from run import ROOT, SIZES, VARIANTS, load_model  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "third_party/Metric3D/onnx"))
from metric3d_onnx_export import update_vit_sampling  # noqa: E402
from common.export_attention import patch_sdpa  # noqa: E402


class Export(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))

    def forward(self, image):
        pred, _, _ = self.model.inference({"input": (image - self.mean) / self.std})
        return pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(VARIANTS), default="vit_large")
    p.add_argument("--size", choices=list(SIZES), default="native")
    a = p.parse_args()
    h, w = SIZES[a.size]
    patch_sdpa()
    model = update_vit_sampling(load_model(a.variant))
    out = os.path.join(ROOT, "engines/metric3d", f"metric3d_{a.variant}_{h}x{w}.onnx")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(Export(model).cuda().eval(), (torch.rand(1, 3, h, w, device="cuda") * 255,), out,
                          input_names=["image"], output_names=["depth"], opset_version=17, dynamo=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
