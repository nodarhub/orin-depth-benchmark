"""Metric3D v2 (ViT-S / ViT-L, https://github.com/YvanYin/Metric3D): FP32 accuracy on Middlebury with the
official pre- and post-processing (hubconf.py example): the image is resized to fit the network input
keeping its aspect ratio, padded with the ImageNet mean, the focal length is scaled with the image, and
the canonical-camera depth is converted to metric depth with focal / 1000. We give the true focal length.

  python methods/metric3d/run.py --variant vit_large --size native --save_preds --save_native
  sizes: native (616x1064, the official ViT input), 2x (1232x2128), full (1988x2884)
"""
import argparse
import json
import os
import sys
import types

import cv2
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
M3D = os.path.join(ROOT, "third_party/Metric3D")
sys.path.insert(0, ROOT)
sys.path.insert(0, M3D)
# mmcv is only used for an environment report in mono/utils/comm.py; it does not build on Jetson
_mm, _mu = types.ModuleType("mmcv"), types.ModuleType("mmcv.utils")
_mu.collect_env, _mu.get_git_hash = (lambda: {}), (lambda *a, **k: "")
_mm.utils = _mu
sys.modules.setdefault("mmcv", _mm)
sys.modules.setdefault("mmcv.utils", _mu)
from mmengine import Config  # noqa: E402
from mono.model.monodepth_model import get_configured_monodepth_model  # noqa: E402

from common.export_attention import classic_attention  # noqa: E402
from common.metrics import all_metrics  # noqa: E402
from mono.model.backbones import ViT_DINO_reg  # noqa: E402

_attention_forward = ViT_DINO_reg.Attention.forward


def _blocked_attention_forward(self, x, attn_bias=None):
    """Same attention, with queries processed in blocks: the full-resolution input (~29k tokens) would
    otherwise materialise a 29k x 29k matrix per head and exceed the Orin's 64 GB."""
    if attn_bias is not None or self.tuning_mode is not None:
        return _attention_forward(self, x, attn_bias)
    B, N, C = x.shape
    q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    x = classic_attention(q, k, v, self.scale).transpose(1, 2).reshape(B, N, C)
    return self.proj_drop(self.proj(x))


ViT_DINO_reg.Attention.forward = _blocked_attention_forward
from common.middlebury import SCENES, load_scene  # noqa: E402

VARIANTS = {"vit_small": ("vit.raft5.small.py", "metric_depth_vit_small_800k.pth"),
            "vit_large": ("vit.raft5.large.py", "metric_depth_vit_large_800k.pth")}
SIZES = {"native": (616, 1064), "2x": (1232, 2128), "full": (1988, 2884)}
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


def load_model(variant):
    cfg_file, ckpt = VARIANTS[variant]
    cfg = Config.fromfile(os.path.join(M3D, "mono/configs/HourglassDecoder", cfg_file))
    model = get_configured_monodepth_model(cfg)
    url = f"https://huggingface.co/JUGGHM/Metric3D/resolve/main/{ckpt}"
    sd = torch.hub.load_state_dict_from_url(url, model_dir=os.path.join(ROOT, "weights/metric3d"),
                                            map_location="cpu")["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    return model.cuda().eval()


def preprocess(bgr, size):
    """Resize keeping the aspect ratio, pad to `size` with the mean colour; returns tensor, scale, pad."""
    ih, iw = size
    h, w = bgr.shape[:2]
    scale = min(ih / h, iw / w)
    rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (int(round(w * scale)), int(round(h * scale))),
                     interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    nh, nw = rgb.shape[:2]
    pad = [(ih - nh) // 2, ih - nh - (ih - nh) // 2, (iw - nw) // 2, iw - nw - (iw - nw) // 2]
    rgb = cv2.copyMakeBorder(rgb, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_CONSTANT, value=MEAN.tolist())
    x = torch.from_numpy(((rgb.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1))[None]
    return x, scale, pad


def infer(model, s, size):
    """Metric depth at the network's own (unpadded) resolution and upsampled to full resolution."""
    H, W = s["left"].shape[:2]
    x, scale, pad = preprocess(s["left"], size)
    with torch.no_grad():
        pred, _, _ = model.inference({"input": x.cuda()})
    d = pred[0, 0].float().cpu().numpy()
    d = d[pad[0]:d.shape[0] - pad[1], pad[2]:d.shape[1] - pad[3]]
    native = d * (s["f"] * scale / 1000.0)  # canonical camera (f = 1000 px) -> real focal length
    return cv2.resize(native, (W, H), interpolation=cv2.INTER_LINEAR), native


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(VARIANTS), default="vit_large")
    p.add_argument("--size", choices=list(SIZES), default="native")
    p.add_argument("--data", default=os.path.join(ROOT, "data/middlebury2014"))
    p.add_argument("--save_preds", action="store_true")
    p.add_argument("--save_native", action="store_true")
    p.add_argument("--scenes", nargs="*", default=SCENES)
    a = p.parse_args()
    model = load_model(a.variant)
    h, w = SIZES[a.size]
    tag = f"metric3d_{a.variant}_{h}x{w}_fp32"
    rows = []
    for name in a.scenes:
        s = load_scene(a.data, name)
        depth, native = infer(model, s, (h, w))
        m = all_metrics(None, depth, s, mono=True)
        m.update(scene=name, method=tag)
        rows.append(m)
        print(name, {k: round(v, 3) for k, v in m.items() if k in ("absrel", "aligned_absrel", "median_scale")}, flush=True)
        for flag, sub, arr in ((a.save_preds, "preds", depth), (a.save_native, "preds_native", native)):
            if flag:
                d = os.path.join(ROOT, "results", sub, tag)
                os.makedirs(d, exist_ok=True)
                np.save(os.path.join(d, f"{name}.npy"), arr.astype(np.float32))
    os.makedirs(os.path.join(ROOT, "results/accuracy"), exist_ok=True)
    json.dump(rows, open(os.path.join(ROOT, "results/accuracy", f"{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
