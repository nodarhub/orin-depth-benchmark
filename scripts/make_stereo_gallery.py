"""Depth maps of the fast stereo configurations on all 10 scenes, next to the ground truth: Light ESS, ESS,
libSGM at quarter, half and full resolution, and Fast-FoundationStereo 20-30-48 at its smallest input.

One row per scene; depth with one colour scale per scene (1st-99th percentile of the ground truth),
grey where there is no ground truth or no prediction (libSGM holes are not filled). The number in each
panel is the AbsRel on the non-occluded pixels of that scene. Column titles give the GPU time per frame.

  paper/figures/stereo_gallery.{pdf,png}
"""
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "methods/libsgm"))
from common.middlebury import disp_to_depth_m, load_scene  # noqa: E402
from tuned import tuned_tag  # noqa: E402

COLUMNS = [("Light ESS\n288×480", "light_ess_288x480"),
           ("ESS\n576×960", "ess_576x960"),
           ("libSGM\n≈497×720 (quarter)", tuned_tag(4)),
           ("libSGM\n≈995×1440 (half)", tuned_tag(2)),
           ("libSGM\nfull res.", tuned_tag(1)),
           ("FFS 20-30-48\n384×576", "ffs_20-30-48_it4_384x576")]
GREY = np.array([0.82, 0.82, 0.82])
STEP = 6  # display subsampling of the full-resolution maps


def to_rgb(values, valid, cmap, norm):
    rgb = np.empty(values.shape + (3,))
    rgb[:] = GREY
    rgb[valid] = cmap(norm(values[valid]))[:, :3]
    return rgb[::STEP, ::STEP]


def main():
    data = os.path.join(ROOT, "data/middlebury2014")
    scenes = json.load(open(os.path.join(ROOT, "results/v2/scene_sets.json")))["subset_lt255"]
    summ = pd.read_csv(os.path.join(ROOT, "results/v2/summary.csv"))
    ms = dict(zip(summ.method, summ.gpu_ms))
    titles = [f"{t}\n{ms[tag]:.1f} ms" if ms.get(tag) == ms.get(tag) and tag in ms else t for t, tag in COLUMNS]
    cmap = plt.get_cmap("turbo_r")
    ncol = 2 + len(COLUMNS)
    fig, axes = plt.subplots(len(scenes), ncol, figsize=(7.2, 0.64 * len(scenes) + 0.9),
                             gridspec_kw=dict(wspace=0.03, hspace=0.05))
    for ri, name in enumerate(scenes):
        s = load_scene(data, name)
        gt = disp_to_depth_m(s["gt_disp"], s["f"], s["baseline_mm"], s["doffs"])
        nocc = s["nonocc"] & (gt > 0)
        lo, hi = np.percentile(gt[gt > 0], [1, 99])
        norm = Normalize(vmin=lo, vmax=hi)
        axes[ri, 0].imshow(cv2.cvtColor(s["left"], cv2.COLOR_BGR2RGB)[::STEP, ::STEP])
        axes[ri, 1].imshow(to_rgb(gt, gt > 0, cmap, norm))
        for ci, (_, tag) in enumerate(COLUMNS):
            disp = np.load(os.path.join(ROOT, "results/preds", tag, name + ".npy")).astype(np.float32)
            depth = disp_to_depth_m(disp, s["f"], s["baseline_mm"], s["doffs"])
            valid = depth > 0
            ax = axes[ri, ci + 2]
            ax.imshow(to_rgb(depth, valid, cmap, norm))
            m = nocc & valid
            absrel = np.mean(np.abs(depth[m] - gt[m]) / gt[m]) * 100
            ax.text(0.03, 0.05, f"{absrel:.2f}%", transform=ax.transAxes, fontsize=4.5, color="white",
                    bbox=dict(facecolor="black", alpha=0.55, pad=0.8, lw=0))
        axes[ri, 0].set_ylabel(name, fontsize=5.5)
    for ci, t in enumerate(["Left image", "Ground truth"] + titles):
        axes[0, ci].set_title(t, fontsize=5, linespacing=1.0)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.3)
    cax = fig.add_axes([0.3, 0.085, 0.4, 0.008])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["near", "far"])
    cb.ax.tick_params(labelsize=5)
    cb.set_label("depth (per-scene ground-truth range, 1st–99th percentile); grey: no ground truth or no prediction",
                 fontsize=5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, f"paper/figures/stereo_gallery.{ext}"), bbox_inches="tight", dpi=300)
    print("wrote paper/figures/stereo_gallery.{pdf,png}")


if __name__ == "__main__":
    main()
