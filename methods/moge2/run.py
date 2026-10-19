"""MoGe-2 (metric point maps; ViT-S/B/L "-normal" checkpoints) on Middlebury 2014: predictions,
ONNX export and an FP16-engine fidelity check.

Accuracy uses the official MoGeModel.infer() in FP32 at its default resolution level 9 (3600 base
tokens = 50x72 patches = 700x1008 px for Middlebury's 3:2 images), with the predicted metric scale.
The FOV is either estimated by the model (default) or set to the ground truth (--gt_fov). The
validity mask is not applied, so the prediction is dense like the other monocular methods.

Usage:
  python methods/moge2/run.py --variant vitl --save_preds [--gt_fov]
  python methods/moge2/run.py --variant vitl --export engines/moge2/moge2_vitl_700x1008.onnx
  python methods/moge2/run.py --variant vitl --check_engine engines/moge2/moge2_vitl_700x1008.engine
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party/MoGe"))
from common.metrics import all_metrics  # noqa: E402
from common.middlebury import SCENES, load_scene  # noqa: E402

NUM_TOKENS = 3600  # resolution_level 9 (default) = top of num_tokens_range [1200, 3600]


def token_grid(H, W, num_tokens=NUM_TOKENS):
    ar = W / H
    return round((num_tokens / ar) ** 0.5), round((num_tokens * ar) ** 0.5)


def to_rgb01(bgr, size=None):
    if size is not None:
        bgr = cv2.resize(bgr, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def load_model(variant):
    import torch
    from moge.model.v2 import MoGeModel
    # the Jetson PyTorch wheel's cuSOLVER linalg library fails to load (undefined symbol); the only
    # inverses MoGe needs are of 3x3 intrinsics, so compute them on the CPU
    for fn in ("inv", "inv_ex"):
        f = getattr(torch.linalg, fn)
        setattr(torch.linalg, fn, lambda A, *args, _f=f, **kw: torch.utils._pytree.tree_map(
            lambda t: t.to(A.device) if isinstance(t, torch.Tensor) else t, _f(A.cpu(), *args, **kw)))
    m = MoGeModel.from_pretrained(os.path.join(ROOT, f"weights/moge2/moge-2-{variant}-normal/model.pt"))
    return m.eval().cuda(), torch


def _classic_attention(self, x, attn_bias=None):
    """Same computation as MoGe's DINOv2 Attention.forward, with the classic, query-blocked attention
    that TensorRT 10.3 builds correctly (common/export_attention.py)."""
    from common.export_attention import classic_attention
    B, N, C = x.shape
    q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
    return self.proj(classic_attention(q, k, v, self.scale).permute(0, 2, 1, 3).reshape(B, N, C))


def export_module(model, torch, rows, cols):
    """Network only, at a fixed token grid: (1,3,14*rows,14*cols) RGB in [0,1] -> points, metric_scale."""
    from moge.model.modules.dinov2.layers import attention
    attention.Attention.forward = _classic_attention
    model.encoder.onnx_compatible_mode = True  # no antialias in the (identity) internal resize

    class W(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model
            self.cls = None
            self.m.encoder.register_forward_hook(lambda mod, inp, out: setattr(self, "cls", out[1]))

        def forward(self, x):
            # the engine stops at the class token: TensorRT 10.3 computes MoGe-2-S/B's tiny metric-scale
            # MLP (3 small Gemms on one 384/768-vector) wrongly, so it runs outside the engine
            # (scale = exp(scale_head(cls)), negligible cost)
            o = self.m.forward(x, num_tokens=rows * cols)
            return o["points"], self.cls
    return W().eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["vits", "vitb", "vitl"], default="vitl")
    p.add_argument("--num_tokens", type=int, default=NUM_TOKENS,
                   help="base ViT tokens (default 3600 = resolution level 9; 0 = each image at its own size, 14 px patches)")
    p.add_argument("--data", default=os.path.join(ROOT, "data/middlebury2014"))
    p.add_argument("--gt_fov", action="store_true", help="give the model the ground-truth horizontal FOV")
    p.add_argument("--export", help="write ONNX of the network at the Middlebury token grid and exit")
    p.add_argument("--check_engine", help="compare a TensorRT engine with FP32 PyTorch on a few scenes")
    p.add_argument("--save_preds", action="store_true")
    p.add_argument("--save_native", action="store_true",
                   help="also save the depth at the token-grid resolution (results/preds_native/)")
    p.add_argument("--scenes", nargs="*", default=SCENES)
    a = p.parse_args()
    model, torch = load_model(a.variant)
    rows, cols = token_grid(1988, 2884, a.num_tokens or round(1988 / 14) * round(2884 / 14))
    h, w = 14 * rows, 14 * cols

    if a.export:
        os.makedirs(os.path.dirname(os.path.abspath(a.export)), exist_ok=True)
        mod = export_module(model, torch, rows, cols)
        with torch.no_grad():
            torch.onnx.export(mod, torch.rand(1, 3, h, w, device="cuda"), a.export, input_names=["image"],
                              output_names=["points", "cls_token"], opset_version=17, dynamo=False)
        print("wrote", a.export, (h, w))
        return

    if a.check_engine:
        from common.trt_runner import TRTRunner
        runner = TRTRunner(a.check_engine)
        mod = export_module(model, torch, rows, cols)
        for name in a.scenes[:5]:
            x = to_rgb01(load_scene(a.data, name)["left"], (h, w))
            with torch.no_grad():
                pt, cls = mod(torch.from_numpy(x).cuda())
                pt, sc = pt.float().cpu().numpy(), model.scale_head(cls).exp().float().cpu().numpy().ravel()
                out = runner(x)
                se = model.scale_head(torch.from_numpy(out["cls_token"]).cuda()).exp().float().cpu().numpy().ravel()
            pe = out["points"].reshape(pt.shape)
            zt, ze = pt[..., 2] * sc[:, None, None], pe[..., 2] * se[:, None, None]
            rel = np.abs(ze - zt) / np.maximum(np.abs(zt), 1e-6)
            print(f"{name}: median rel diff {100 * np.median(rel):.3f}%  p99 {100 * np.percentile(rel, 99):.3f}%  "
                  f"scale {float(sc[0]):.4f} vs {float(se[0]):.4f}")
        return

    tag = f"moge2_{a.variant}_{h}x{w}_fp32" + ("" if a.gt_fov else "_estf")
    out_rows = []
    for name in a.scenes:
        s = load_scene(a.data, name)
        H, W = s["left"].shape[:2]
        fov_x = float(np.rad2deg(2 * np.arctan(0.5 * W / s["f"]))) if a.gt_fov else None
        n_tok = a.num_tokens or round(H / 14) * round(W / 14)
        with torch.no_grad():
            o = model.infer(torch.from_numpy(to_rgb01(s["left"])[0]).cuda(), num_tokens=n_tok,
                            apply_mask=False, fov_x=fov_x, use_fp16=False)
        depth = o["depth"].float().cpu().numpy()
        fx_pred = float(o["intrinsics"][0, 0]) * W
        m = all_metrics(None, depth, s, mono=True)
        m.update(scene=name, method=tag, f_px_used=fx_pred, f_px_true=float(s["f"]),
                 token_grid=list(token_grid(H, W, n_tok)))
        out_rows.append(m)
        print(name, {k: round(v, 3) for k, v in m.items() if k in ("absrel", "aligned_absrel", "median_scale")},
              f"f {fx_pred:.0f}/{s['f']:.0f}")
        if a.save_preds:
            d = os.path.join(ROOT, "results/preds", tag)
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, f"{name}.npy"), depth.astype(np.float32))
        if a.save_native:  # same inference on the image resized to the token grid: output at that size
            gr, gc = token_grid(H, W, n_tok)
            with torch.no_grad():
                on = model.infer(torch.from_numpy(to_rgb01(s["left"], (14 * gr, 14 * gc))[0]).cuda(),
                                 num_tokens=n_tok, apply_mask=False, fov_x=fov_x, use_fp16=False)
            d = os.path.join(ROOT, "results/preds_native", tag)
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, f"{name}.npy"), on["depth"].float().cpu().numpy().astype(np.float32))
    os.makedirs(os.path.join(ROOT, "results/accuracy"), exist_ok=True)
    json.dump(out_rows, open(os.path.join(ROOT, "results/accuracy", f"{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
