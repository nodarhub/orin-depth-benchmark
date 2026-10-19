"""Accuracy of Fixstars libSGM on Middlebury 2014 at full, half or quarter resolution.

  python methods/libsgm/run.py --scale 1 --save_preds     # full resolution (≈1990×2880)
  python methods/libsgm/run.py --scale 2 --save_preds     # half (≈995×1440)
  python methods/libsgm/run.py --scale 4 --save_preds     # quarter (≈497×720)

256 disparity levels at every scale (libSGM's maximum), so at full resolution disparities above
255 px are out of range; the disparity is upscaled to the GT resolution and multiplied by --scale.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from sgm import LibSGM  # noqa: E402
from tuned import tag as make_tag, tuned_params  # noqa: E402
from common.metrics import all_metrics  # noqa: E402
from common.middlebury import SCENES, load_scene  # noqa: E402

TAGS = {1: "libsgm_full", 2: "libsgm_half", 4: "libsgm_quarter"}


def prepare(s, scale):
    g = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in (s["left"], s["right"])]
    if scale > 1:
        H, W = g[0].shape
        g = [cv2.resize(x, (round(W / scale), round(H / scale)), interpolation=cv2.INTER_AREA) for x in g]
    return g


def to_full_res(d, shape):
    """Nearest-neighbour upsampling of a low-resolution disparity map (keeps holes as holes) and
    rescaling of the disparity values to full-resolution pixels."""
    H, W = shape
    h, w = d.shape
    if (h, w) == (H, W):
        return d
    disp = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST) * (W / w)
    disp[cv2.resize((d > 0).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) == 0] = 0
    return disp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=int, default=1, choices=[1, 2, 4])
    p.add_argument("--paths", type=int, default=8, choices=[4, 8])
    p.add_argument("--p1", type=int, default=10)
    p.add_argument("--p2", type=int, default=120)
    p.add_argument("--tuned", action="store_true", help="use the P1/P2 selected by tune.py for this scale")
    p.add_argument("--data", default=os.path.join(ROOT, "data/middlebury2014"))
    p.add_argument("--out_dir", default=os.path.join(ROOT, "results/accuracy"))
    p.add_argument("--save_preds", action="store_true")
    p.add_argument("--scenes", nargs="*", default=SCENES)
    a = p.parse_args()
    if a.tuned:
        a.p1, a.p2 = tuned_params(a.scale)
    tag = make_tag(a.scale, a.paths, a.p1, a.p2)
    sgm, size, rows = None, None, []
    for name in a.scenes:
        s = load_scene(a.data, name)
        H, W = s["gt_disp"].shape
        g = prepare(s, a.scale)
        h, w = g[0].shape
        if size != (w, h):  # every scene has its own size; keep one instance (full res needs ~12 GB)
            sgm = None
            sgm, size = LibSGM(w, h, 256, a.paths, P1=a.p1, P2=a.p2), (w, h)
        disp = to_full_res(sgm(*g), (H, W))
        m = all_metrics(disp, None, s)
        m.update(scene=name, method=tag, proc_h=h, proc_w=w)
        rows.append(m)
        print(name, f"{w}x{h}", {k: round(v, 3) for k, v in m.items() if k in ("density", "bad2", "absrel")}, flush=True)
        if a.save_preds:
            pd = os.path.join(ROOT, "results/preds", tag)
            os.makedirs(pd, exist_ok=True)
            np.save(os.path.join(pd, f"{name}.npy"), disp.astype(np.float32))
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f"{tag}.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
