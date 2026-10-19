"""Export Apple Depth Pro to ONNX (fixed 1536x1536 network input).

Input:  image (1,3,1536,1536) float32 RGB scaled to [-1, 1]
Output: canonical_inverse_depth (1,1,1536,1536), fov_deg (1,1)
Metric depth is recovered outside the network (see run.py), exactly as
depth_pro.DepthPro.infer does: inverse_depth = canonical * (W / f_px).
"""
import argparse
import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, os.path.join(ROOT, "third_party/ml-depth-pro/src"))
from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT, create_model_and_transforms  # noqa: E402


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=os.path.join(ROOT, "weights/depthpro/depth_pro.pt"))
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cfg = DEFAULT_MONODEPTH_CONFIG_DICT
    cfg.checkpoint_uri = a.weights
    model, _ = create_model_and_transforms(cfg, device=torch.device("cuda"), precision=torch.float32)
    model.eval()
    x = torch.randn(1, 3, 1536, 1536, device="cuda")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(Wrapper(model), x, a.out, input_names=["image"],
                          output_names=["canonical_inverse_depth", "fov_deg"],
                          opset_version=17, dynamo=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
