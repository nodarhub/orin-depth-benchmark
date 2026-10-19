"""UniDepth V2 (ViT-S / ViT-B / ViT-L, https://github.com/lpiccinelli-eth/UniDepth): FP32 accuracy on
Middlebury through the official `infer`, given the true intrinsics. UniDepth chooses its processing size
from a pixel budget (shape_constraints: 0.2-0.6 MP by default, multiples of 14); we run the default
budget (native) and budgets of about 4x (2x per side) and the full image.

  python methods/unidepth/run.py --variant vitl14 --size native --save_preds --save_native
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
sys.modules.setdefault("wandb", types.ModuleType("wandb"))  # imported by unidepth.utils.visualization only
from unidepth.models import UniDepthV2  # noqa: E402
from unidepth.models.unidepthv2.unidepthv2 import get_paddings, get_resize_factor  # noqa: E402

# torch.inverse on CUDA needs a newer libcusolver than JetPack 6.2 ships with the Jetson torch build.
# The only inversion on UniDepth's inference path is of the 3x3 camera matrix (utils/camera.py), so do
# it on the CPU. Patched after the import, when UniDepth's TorchScript functions are already compiled.
_inverse = torch.inverse


def _cpu_inverse(x, *args, **kwargs):
    return _inverse(x.cpu(), *args, **kwargs).to(x.device)


torch.inverse = _cpu_inverse

from common.metrics import all_metrics  # noqa: E402
from common.middlebury import SCENES, load_scene  # noqa: E402

# pixel budgets (min, max): native = the released default; full = the whole image
CAPTURE = {}
SIZES = {"native": None, "2x": (2.2e6, 2.45e6), "full": (5.5e6, 5.8e6)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["vits14", "vitb14", "vitl14"], default="vitl14")
    p.add_argument("--size", choices=list(SIZES), default="native")
    p.add_argument("--data", default=os.path.join(ROOT, "data/middlebury2014"))
    p.add_argument("--save_preds", action="store_true")
    p.add_argument("--save_native", action="store_true")
    p.add_argument("--scenes", nargs="*", default=SCENES)
    a = p.parse_args()
    model = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2-{a.variant}",
                                       cache_dir=os.path.join(ROOT, "weights/unidepth")).cuda().eval()
    model.interpolation_mode = "bilinear"
    _encode_decode = model.encode_decode

    def encode_decode(*args, **kwargs):  # keep the processed-size outputs for --save_native
        out = _encode_decode(*args, **kwargs)
        CAPTURE["points"] = out[1]["points"]
        return out
    model.encode_decode = encode_decode
    if SIZES[a.size] is not None:
        model.shape_constraints["pixels_min"], model.shape_constraints["pixels_max"] = SIZES[a.size]
    rows, tag = [], None
    for name in a.scenes:
        s = load_scene(a.data, name)
        H, W = s["left"].shape[:2]
        K = torch.tensor(s["calib"]["cam0"], dtype=torch.float32)
        rgb = torch.from_numpy(s["left"][..., ::-1].copy()).permute(2, 0, 1)  # BGR -> RGB, uint8 CHW
        with torch.no_grad():
            out = model.infer(rgb, K)
        depth = out["depth"][0, 0].float().cpu().numpy()
        # processed size: UniDepth pads to its aspect-ratio bounds and resizes to the pixel budget
        _, (ph, pw) = get_paddings((H, W), model.shape_constraints["ratio_bounds"])
        _, (nh, nw) = get_resize_factor((ph, pw), [model.shape_constraints["pixels_min"],
                                                    model.shape_constraints["pixels_max"]])
        tag = tag or f"unidepthv2_{a.variant}_{a.size}_fp32"
        m = all_metrics(None, depth, s, mono=True)
        m.update(scene=name, method=tag, proc_h=int(nh), proc_w=int(nw))
        rows.append(m)
        print(name, (nh, nw), {k: round(v, 3) for k, v in m.items() if k in ("absrel", "aligned_absrel", "median_scale")},
              flush=True)
        if a.save_preds:
            d = os.path.join(ROOT, "results/preds", tag)
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, f"{name}.npy"), depth.astype(np.float32))
        if a.save_native:  # depth at the processed size, captured inside infer (before its upsampling)
            z = CAPTURE["points"][0, -1].float().cpu().numpy()
            pads, (ph, pw) = get_paddings((H, W), model.shape_constraints["ratio_bounds"])
            fy, fx = z.shape[0] / ph, z.shape[1] / pw  # remove the (scaled) aspect-ratio padding
            l, r, t, b = (int(round(v)) for v in (pads[0] * fx, pads[1] * fx, pads[2] * fy, pads[3] * fy))
            z = z[t:z.shape[0] - b, l:z.shape[1] - r]
            d = os.path.join(ROOT, "results/preds_native", tag)
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, f"{name}.npy"), z.astype(np.float32))
    os.makedirs(os.path.join(ROOT, "results/accuracy"), exist_ok=True)
    json.dump(rows, open(os.path.join(ROOT, "results/accuracy", f"{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
