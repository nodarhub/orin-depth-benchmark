"""Qualitative figure: depth maps and relative depth error of every method on two scenes.
--set main: libSGM at full resolution and Fast-FoundationStereo at its maximum input size. Monocular networks are shown with the per-image oracle scale (median ratio to the
non-occluded GT), so the remaining error is structural rather than a global scale error.

Rows per scene: depth (shared colour scale from the GT depth range) and relative depth error on
non-occluded GT pixels (log scale). Raw outputs only: stereo holes are shown in grey, not filled.
Sword1 is the scene on which every method's error is closest to its median over the <255 px subset;
Classroom1 adds large planar surfaces (floor, desks, walls).
"""
import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from common.metrics import median_scale  # noqa: E402
from common.middlebury import disp_to_depth_m, load_scene  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "methods/libsgm"))
from tuned import tuned_tag  # noqa: E402

# (column title, prediction tag, kind)   kind: disp = stereo disparity, depth = monocular depth
# "{ms}" in a title is replaced by the method's GPU time from results/v2/summary.csv
# --set main: every method at full resolution, or its maximum / fixed input size; monocular depth
# with the per-image oracle scale, so the remaining error is structural
COLUMNS = [("libSGM\nfull res.\n{ms}", tuned_tag(1), "disp"),
           ("FFS 23-36-37\n768×1152 (max)\n{ms}", "ffs_23-36-37_it8_768x1152", "disp"),
           ("Depth Pro\n1536×1536\n(oracle scale)", "depthpro_1536_fp32", "depth_oracle"),
           ("DA V2-L\nfull res.\n(oracle scale)", "dav2_vitl_1988x2884", "depth_oracle"),
           ("DA3 Metric-L\nfull res.\n(oracle scale)", "da3metric_large_1988x2884_fp32", "depth_oracle"),
           ("MoGe-2-L\nfull res.\n(oracle scale)", "moge2_vitl_1988x2884_fp32", "depth_oracle")]
# --set lowres: stereo and monocular near the monocular models' native input, as deployed
# (monocular depth raw, i.e. with its own metric scale; Depth Anything 3 also with oracle scale)
COLUMNS_LOWRES = [("libSGM\n≈497×720\n{ms}", tuned_tag(4), "disp"),
                  ("FFS 20-30-48\n512×768\n{ms}", "ffs_20-30-48_it4_512x768", "disp"),
                  ("FFS 23-36-37\n512×768\n{ms}", "ffs_23-36-37_it8_512x768", "disp"),
                  ("DA V2-S\n518×770\n{ms}", "dav2_vits_518x770", "depth"),
                  ("DA V2-L\n518×770\n{ms}", "dav2_vitl_518x770", "depth"),
                  ("MoGe-2-L\n700×1008\n{ms}", "moge2_vitl_700x1008_fp32", "depth"),
                  ("DA3 Metric-L\n350×504\n{ms}", "da3metric_large_350x504_fp32", "depth"),
                  ("DA3 Metric-L\n350×504\noracle scale", "da3metric_large_350x504_fp32", "depth_oracle")]
GREY = np.array([0.82, 0.82, 0.82])


def to_rgb(values, valid, cmap, norm, step):
    rgb = np.empty(values.shape + (3,))
    rgb[:] = GREY
    rgb[valid] = cmap(norm(values[valid]))[:, :3]
    return rgb[::step, ::step]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="*", default=["Sword1", "Classroom1"])
    p.add_argument("--step", type=int, default=6, help="display subsampling of the full-resolution maps")
    p.add_argument("--set", choices=["main", "lowres"], default="main",
                   help="lowres: libSGM, Fast-FoundationStereo and the monocular models near their native input")
    p.add_argument("--out", default=None, help="default paper/figures/depth_maps[_lowres]")
    a = p.parse_args()
    a.out = a.out or ("paper/figures/depth_maps" + ("_lowres" if a.set == "lowres" else ""))
    columns = list(COLUMNS_LOWRES if a.set == "lowres" else COLUMNS)
    import pandas as pd
    summ = pd.read_csv(os.path.join(ROOT, "results/v2/summary.csv"))
    ms = {}
    for _, r in summ.iterrows():
        ms[r.method] = r.gpu_ms
        ms[r.method.replace("_fp32", "")] = r.gpu_ms

    def fmt_ms(t):
        v = ms.get(t, ms.get(t.replace("_fp32", "")))
        return "" if v is None or v != v else (f"{v:.1f} ms" if v < 100 else f"{v:.0f} ms")
    columns = [(title.replace("{ms}", fmt_ms(tag)), tag, kind) for title, tag, kind in columns]
    data = os.path.join(ROOT, "data/middlebury2014")
    ncol = 2 + len(columns)
    nrow = 2 * len(a.scenes)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 0.56 * nrow + 0.75),
                             gridspec_kw=dict(wspace=0.03, hspace=0.04))
    depth_cmap, err_cmap = plt.get_cmap("turbo_r"), plt.get_cmap("inferno")
    err_norm = LogNorm(vmin=0.1, vmax=100)
    for si, name in enumerate(a.scenes):
        s = load_scene(data, name)
        f, B, doffs = s["f"], s["baseline_mm"], s["doffs"]
        gt = disp_to_depth_m(s["gt_disp"], f, B, doffs)
        nocc = s["nonocc"] & (gt > 0)
        lo, hi = np.percentile(gt[gt > 0], [1, 99])
        dnorm = Normalize(vmin=lo, vmax=hi)
        r_depth, r_err = 2 * si, 2 * si + 1
        img = cv2.cvtColor(s["left"], cv2.COLOR_BGR2RGB)[::a.step, ::a.step]
        axes[r_depth, 0].imshow(img)
        axes[r_depth, 1].imshow(to_rgb(gt, gt > 0, depth_cmap, dnorm, a.step))
        mask_rgb = np.where(nocc[..., None], 1.0, GREY)[::a.step, ::a.step]
        axes[r_err, 0].imshow(mask_rgb)
        axes[r_err, 1].axis("off")
        for ci, (title, tag, kind) in enumerate(columns):
            pred = np.load(os.path.join(ROOT, "results/preds", tag, name + ".npy")).astype(np.float32)
            depth = disp_to_depth_m(pred, f, B, doffs) if kind == "disp" else pred
            if kind == "depth_oracle":
                depth = depth * median_scale(depth, np.where(nocc, gt, 0))
            valid = depth > 0
            axes[r_depth, ci + 2].imshow(to_rgb(depth, valid, depth_cmap, dnorm, a.step))
            m = nocc & valid
            err = np.zeros_like(gt)
            err[m] = np.abs(depth[m] - gt[m]) / gt[m] * 100
            axes[r_err, ci + 2].imshow(to_rgb(err, m, err_cmap, err_norm, a.step))
            absrel = err[m].mean()
            axes[r_err, ci + 2].text(0.03, 0.04, f"{absrel:.2f}%" if absrel < 10 else f"{absrel:.1f}%",
                                     transform=axes[r_err, ci + 2].transAxes, fontsize=5.5, color="white",
                                     bbox=dict(facecolor="black", alpha=0.55, pad=1, lw=0))
            if si == 0:
                axes[0, ci + 2].set_title(title, fontsize=5, linespacing=1.0)
        if si == 0:
            axes[0, 0].set_title("Left image", fontsize=5)
            axes[0, 1].set_title("Ground truth", fontsize=5)
        axes[r_depth, 0].set_ylabel(f"{name}\ndepth", fontsize=6)
        axes[r_err, 0].set_ylabel(f"{name}\nrel. error", fontsize=6)
        axes[r_err, 0].text(0.03, 0.04, "scored pixels", transform=axes[r_err, 0].transAxes, fontsize=5,
                            bbox=dict(facecolor="white", alpha=0.7, pad=1, lw=0))
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.3)
    cax1 = fig.add_axes([0.13, 0.06, 0.32, 0.014])
    cb1 = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=depth_cmap), cax=cax1, orientation="horizontal")
    cb1.set_ticks([0, 1])
    cb1.set_ticklabels(["near", "far"])
    cb1.ax.tick_params(labelsize=5.5)
    cb1.set_label("depth (per-scene GT range, 1st–99th percentile)", fontsize=5.5)
    cax2 = fig.add_axes([0.56, 0.06, 0.32, 0.014])
    cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=err_norm, cmap=err_cmap), cax=cax2, orientation="horizontal")
    cb2.set_ticks([0.1, 1, 10, 100])
    cb2.set_ticklabels(["0.1%", "1%", "10%", "100%"])
    cb2.ax.tick_params(labelsize=5.5)
    cb2.set_label("relative depth error |Ẑ−Z|/Z (non-occluded pixels)", fontsize=5.5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, f"{a.out}.{ext}"), bbox_inches="tight", dpi=300)
    print(f"wrote {a.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
