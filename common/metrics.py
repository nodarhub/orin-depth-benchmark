"""Accuracy metrics. All evaluated at full Middlebury resolution on GT-valid pixels."""
import numpy as np

from .middlebury import disp_to_depth_m


def disparity_metrics(pred, gt, thresholds=(1, 2, 4)):
    """Stereo metrics. Pixels with no prediction (pred<=0) count as errors (dense eval)
    and are also reported separately as density."""
    valid = gt > 0
    has_pred = (pred > 0) & valid
    err = np.abs(pred - gt)
    err[~has_pred] = np.inf
    out = {"density": float(has_pred.sum() / valid.sum())}
    e_all = err[valid]
    for t in thresholds:
        out[f"bad{t}"] = float((e_all > t).mean() * 100)
        out[f"bad{t}_nonocc_pred"] = float((err[has_pred] > t).mean() * 100) if has_pred.any() else np.nan
    out["epe_pred"] = float(err[has_pred].mean()) if has_pred.any() else np.nan
    return out


def depth_metrics(pred_depth, gt_depth, prefix=""):
    """Standard monocular depth metrics on pixels where both are positive."""
    m = (gt_depth > 0) & (pred_depth > 0) & np.isfinite(pred_depth)
    p, g = pred_depth[m].astype(np.float64), gt_depth[m].astype(np.float64)
    ratio = np.maximum(p / g, g / p)
    return {
        f"{prefix}absrel": float(np.mean(np.abs(p - g) / g) * 100),
        f"{prefix}rmse_m": float(np.sqrt(np.mean((p - g) ** 2))),
        f"{prefix}d1": float(np.mean(ratio < 1.25) * 100),
        f"{prefix}d_1.05": float(np.mean(ratio < 1.05) * 100),
        f"{prefix}d_1.01": float(np.mean(ratio < 1.01) * 100),
    }


def median_scale(pred_depth, gt_depth):
    m = (gt_depth > 0) & (pred_depth > 0) & np.isfinite(pred_depth)
    return float(np.median(gt_depth[m]) / np.median(pred_depth[m]))


def all_metrics(pred_disp, pred_depth, scene, mono=False):
    """Metrics on all GT pixels, plus the same metrics on non-occluded pixels (prefix nonocc_)."""
    out = _all_metrics(pred_disp, pred_depth, scene, mono)
    if "nonocc" in scene:
        sub = dict(scene, gt_disp=np.where(scene["nonocc"], scene["gt_disp"], 0))
        out.update({f"nonocc_{k}": v for k, v in _all_metrics(pred_disp, pred_depth, sub, mono,
                                                              scale_scene=scene).items()})
        out["nonocc_fraction"] = float(scene["nonocc"].sum() / max((scene["gt_disp"] > 0).sum(), 1))
    return out


def _all_metrics(pred_disp, pred_depth, scene, mono=False, scale_scene=None):
    """pred_disp/pred_depth at full res. For stereo give disparity; for mono give depth.
    The missing one is derived via the Middlebury calibration so every method is
    scored in both spaces."""
    gt = scene["gt_disp"]
    f, B, doffs = scene["f"], scene["baseline_mm"], scene["doffs"]
    gt_depth = disp_to_depth_m(gt, f, B, doffs)
    from .middlebury import depth_to_disp
    if pred_depth is None:
        pred_depth = disp_to_depth_m(pred_disp, f, B, doffs)
    if pred_disp is None:
        pred_disp = depth_to_disp(pred_depth, f, B, doffs)
    out = {}
    out.update(disparity_metrics(pred_disp, gt))
    out.update(depth_metrics(pred_depth, gt_depth))
    if mono:
        # oracle scale is always fitted on all GT pixels, then evaluated on the given mask
        ss = scale_scene or scene
        s = median_scale(pred_depth, disp_to_depth_m(ss["gt_disp"], f, B, doffs))
        out["median_scale"] = s
        aligned = pred_depth * s
        out.update({f"aligned_{k}": v for k, v in
                    disparity_metrics(depth_to_disp(aligned, f, B, doffs), gt).items()})
        out.update(depth_metrics(aligned, gt_depth, prefix="aligned_"))
    return out


def fill_holes(disp):
    """MiddEval3-style hole filling: each invalid pixel takes the smaller (background)
    of the nearest valid disparities to its left and right on the same scanline."""
    d = disp.astype(np.float32).copy()
    H, W = d.shape
    valid = d > 0
    idx = np.where(valid, np.arange(W)[None, :], -1)
    left_idx = np.maximum.accumulate(idx, axis=1)
    idx_r = np.where(valid, np.arange(W)[None, :], W)
    right_idx = np.minimum.accumulate(idx_r[:, ::-1], axis=1)[:, ::-1]
    rows = np.arange(H)[:, None]
    lv = np.where(left_idx >= 0, d[rows, np.clip(left_idx, 0, W - 1)], np.inf)
    rv = np.where(right_idx < W, d[rows, np.clip(right_idx, 0, W - 1)], np.inf)
    fill = np.minimum(lv, rv)
    fill[~np.isfinite(fill)] = 0
    d[~valid] = fill[~valid]
    return d
