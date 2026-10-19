"""Middlebury 2014 loading helpers."""
import os

import cv2
import numpy as np

SCENES = [
    "Adirondack", "Backpack", "Bicycle1", "Cable", "Classroom1", "Couch", "Flowers",
    "Jadeplant", "Mask", "Motorcycle", "Piano", "Pipes", "Playroom", "Playtable",
    "Recycle", "Shelves", "Shopvac", "Sticks", "Storage", "Sword1", "Sword2",
    "Umbrella", "Vintage",
]


def read_pfm(path):
    with open(path, "rb") as f:
        header = f.readline().decode().strip()
        channels = 3 if header == "PF" else 1
        w, h = map(int, f.readline().decode().split())
        scale = float(f.readline().decode().strip())
        data = np.fromfile(f, "<f4" if scale < 0 else ">f4")
    data = data.reshape(h, w, channels) if channels == 3 else data.reshape(h, w)
    return np.flipud(data).astype(np.float32)


def read_calib(path):
    calib = {}
    for line in open(path):
        k, v = line.strip().split("=", 1)
        if v.startswith("["):
            rows = [list(map(float, r.split())) for r in v.strip("[]").split(";")]
            calib[k] = np.array(rows)
        else:
            calib[k] = float(v)
    return calib


def nonocc_mask(gt0, gt1, tol=1.0):
    """GT pixels of the left view that are also visible in the right view: the match x - d lies
    inside the right image and the right-view GT disparity agrees within `tol` px (Middlebury's
    occlusion definition, as in MiddEval3's mask0nocc). Occluded and out-of-view pixels have GT
    (from structured light) but cannot be matched by any stereo method."""
    H, W = gt0.shape
    valid = gt0 > 0
    xr = np.round(np.arange(W)[None, :] - gt0).astype(int)
    inside = valid & (xr >= 0)
    rows = np.broadcast_to(np.arange(H)[:, None], (H, W))
    d1 = np.zeros_like(gt0)
    d1[inside] = gt1[rows[inside], xr[inside]]
    return inside & np.isfinite(d1) & (d1 > 0) & (np.abs(d1 - gt0) <= tol)


def load_scene(root, scene):
    d = os.path.join(root, f"{scene}-perfect")
    calib = read_calib(os.path.join(d, "calib.txt"))
    left = cv2.imread(os.path.join(d, "im0.png"), cv2.IMREAD_COLOR)
    right = cv2.imread(os.path.join(d, "im1.png"), cv2.IMREAD_COLOR)
    gt = read_pfm(os.path.join(d, "disp0.pfm"))
    gt[~np.isfinite(gt)] = 0  # inf marks unknown disparity
    gt1 = read_pfm(os.path.join(d, "disp1.pfm"))
    gt1[~np.isfinite(gt1)] = 0
    return dict(
        name=scene, left=left, right=right, gt_disp=gt, calib=calib, nonocc=nonocc_mask(gt, gt1),
        f=calib["cam0"][0, 0], baseline_mm=calib["baseline"], doffs=calib["doffs"],
    )


def disp_to_depth_m(disp, f, baseline_mm, doffs):
    """Middlebury convention: Z = baseline * f / (d + doffs)."""
    depth = np.zeros_like(disp, dtype=np.float32)
    valid = disp > 0
    depth[valid] = baseline_mm * f / (disp[valid] + doffs) / 1000.0
    return depth


def depth_to_disp(depth_m, f, baseline_mm, doffs):
    disp = np.zeros_like(depth_m, dtype=np.float32)
    valid = depth_m > 0
    disp[valid] = baseline_mm * f / (depth_m[valid] * 1000.0) - doffs
    return disp
