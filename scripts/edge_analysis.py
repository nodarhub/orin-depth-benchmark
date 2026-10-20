"""Depth discontinuities: "flying" points that a method places between the foreground and the
background surface at object edges, a failure mode that the mean AbsRel barely registers.

Every method is analysed at its own output resolution (results/preds_native/ for the networks, written
with --save_native; libSGM's predictions are already at full resolution or nearest-upsampled), so no
intermediate depths come from our own upsampling. Monocular depth gets the per-image oracle scale
first, so a global scale error is not counted as an edge error.

Edge pixels: non-occluded GT pixels whose (2R+1)^2 GT neighbourhood spans a depth ratio of at least
JUMP. A predicted depth at an edge pixel is "flying" if it lies more than 5% away from both the nearest
(Zmin) and the farthest (Zmax) GT surface in that neighbourhood and between them.

  results/v2/edges.csv               per scene and method
  paper/tables/edges.tex             means over the 10 scenes
  paper/figures/edge_pointclouds.{pdf,png}   top-down point clouds of a band of rows across an edge
  paper/figures/edge_pointclouds_stereo.{pdf,png}   the same for the fast stereo configurations
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "methods/libsgm"))
from common.metrics import median_scale  # noqa: E402
from common.middlebury import disp_to_depth_m, load_scene  # noqa: E402
from tuned import tuned_tag  # noqa: E402

R, JUMP, TOL = 8, 1.2, 0.05
# (label, tag, kind, source)  kind: disp | depth;  source: preds (full-res grid) | native
METHODS = [("libSGM, full res.", tuned_tag(1), "disp", "preds"),
           ("libSGM, quarter res.", tuned_tag(4), "disp", "preds"),
           ("FFS 23-36-37, 768×1152", "ffs_23-36-37_it8_768x1152", "disp", "native"),
           ("FFS 20-30-48, 768×1152", "ffs_20-30-48_it4_768x1152", "disp", "native"),
           ("C-FFS, 768×1152", "ffs_c-ffs_it8_768x1152", "disp", "native"),
           ("Depth Anything V2-S, 518×770", "dav2_vits_518x770", "depth", "native"),
           ("Depth Anything V2-L, 518×770", "dav2_vitl_518x770", "depth", "native"),
           ("Depth Anything 3 Metric-L, 350×504", "da3metric_large_350x504_fp32", "depth", "native"),
           ("MoGe-2-L, 700×1008", "moge2_vitl_700x1008_fp32", "depth", "native"),
           ("ESS, 576×960", "ess_576x960", "disp", "native"),
           ("Light ESS, 288×480", "light_ess_288x480", "disp", "native"),
           ("Metric3D v2-L, 616×1064", "metric3d_vit_large_616x1064_fp32", "depth", "native"),
           ("UniDepth V2-L, 644×938", "unidepthv2_vitl14_native_fp32", "depth", "native")]
# figure: columns and the edge crop (scene, first row, last row, first col, last col) of the band shown
FIG_METHODS = [METHODS[i] for i in (0, 2, 8, 7, 6)]  # libSGM full, FFS 23-36-37, MoGe-2-L, DA3, DA V2-L
# second figure: the fast stereo configurations only
STEREO_FIG_METHODS = [METHODS[10], METHODS[9],
                      ("libSGM, quarter res.", tuned_tag(4), "disp", "preds"),
                      ("libSGM, half res.", tuned_tag(2), "disp", "preds"),
                      METHODS[0],
                      ("FFS 20-30-48, 384×576", "ffs_20-30-48_it4_384x576", "disp", "native")]
CROPS = [("Adirondack", 380, 420, 1750, 2800), ("Motorcycle", 1280, 1320, 1950, 2800)]


def load_method(s, tag, kind, source, gt_depth, nocc):
    """Full-resolution depth (nearest-neighbour from the method's own grid) and the native grid itself."""
    H, W = s["gt_disp"].shape
    sub = "preds_native" if source == "native" else "preds"
    p = np.load(os.path.join(ROOT, "results", sub, tag, s["name"] + ".npy")).astype(np.float32)
    h, w = p.shape
    full = cv2.resize(p, (W, H), interpolation=cv2.INTER_NEAREST) if (h, w) != (H, W) else p
    to_depth = (lambda d: disp_to_depth_m(d, s["f"], s["baseline_mm"], s["doffs"])) if kind == "disp" else (lambda d: d)
    z_full, z_nat = to_depth(full), to_depth(p)
    if kind == "depth":  # oracle scale: per-image median GT/prediction ratio on non-occluded pixels
        k = median_scale(z_full, np.where(nocc, gt_depth, 0))
        z_full, z_nat = z_full * k, z_nat * k
    return z_full, z_nat


def edge_mask(gt_depth, nocc):
    valid = nocc & (gt_depth > 0)
    k = np.ones((2 * R + 1, 2 * R + 1), np.uint8)
    zmin = cv2.erode(np.where(valid, gt_depth, np.inf).astype(np.float32), k)
    zmax = cv2.dilate(np.where(valid, gt_depth, 0).astype(np.float32), k)
    edge = valid & np.isfinite(zmin) & (zmax >= JUMP * zmin)
    return edge, zmin, zmax


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure_only", action="store_true", help="only redraw the point-cloud figure")
    a = ap.parse_args()
    data = os.path.join(ROOT, "data/middlebury2014")
    scenes = json.load(open(os.path.join(ROOT, "results/v2/scene_sets.json")))["subset_lt255"]
    if not a.figure_only:
        metrics(data, scenes)
    figure(data, FIG_METHODS, "edge_pointclouds")
    figure(data, STEREO_FIG_METHODS, "edge_pointclouds_stereo")


def metrics(data, scenes):
    rows = []
    for name in scenes:
        s = load_scene(data, name)
        gt = disp_to_depth_m(s["gt_disp"], s["f"], s["baseline_mm"], s["doffs"])
        nocc = s["nonocc"] & (gt > 0)
        edge, zmin, zmax = edge_mask(gt, nocc)
        for label, tag, kind, source in METHODS:
            z, _ = load_method(s, tag, kind, source, gt, nocc)
            pe = edge & (z > 0)
            fly = pe & (z > zmin * (1 + TOL)) & (z < zmax / (1 + TOL))
            rel = np.abs(z - gt) / np.maximum(gt, 1e-6)
            off = nocc & ~edge & (z > 0)
            rows.append(dict(scene=name, method=label, edge_px=int(edge.sum()),
                             edge_density=pe.sum() / edge.sum() * 100, flying=fly.sum() / max(pe.sum(), 1) * 100,
                             absrel_edge=rel[pe].mean() * 100, absrel_off=rel[off].mean() * 100,
                             edge_share=edge.sum() / nocc.sum() * 100))
        print(name, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results/v2/edges.csv"), index=False)
    m = df.groupby("method", sort=False).mean(numeric_only=True)
    print(m.round(2).to_string())
    with open(os.path.join(ROOT, "paper/tables/edges.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n")
        f.write("Method (output resolution) & Density at edges [\\%] & Flying points [\\%] & "
                "AbsRel at edges [\\%] & AbsRel elsewhere [\\%] \\\\\n\\midrule\n")
        for label, r in m.iterrows():
            lab = label.replace("×", "$\\times$")
            f.write(f"{lab} & {r.edge_density:.0f} & {r.flying:.1f} & {r.absrel_edge:.2f} & {r.absrel_off:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"edge pixels: {df.drop_duplicates('scene').edge_share.mean():.1f}% of the non-occluded GT")


def figure(data, methods, out):
    # ---- figure: top-down view (lateral position X vs depth Z) of a band of rows across an edge ----
    plt.rcParams.update({"font.size": 7, "figure.dpi": 200})
    cols = [("Ground truth", None)] + [(m[0], m) for m in methods]
    # first column: the rectified left image with the analysed band of rows boxed
    # columns: image | spacer (room for the depth axis) | point clouds
    fig, axes_all = plt.subplots(len(CROPS), len(cols) + 2, figsize=(7.2, 2.1 * len(CROPS) + 0.3),
                                 gridspec_kw=dict(width_ratios=[2.3, 0.32] + [1] * len(cols), wspace=0.1))
    for ri in range(len(CROPS)):
        axes_all[ri, 1].axis("off")
        for ci in range(3, len(cols) + 2):
            axes_all[ri, ci].sharey(axes_all[ri, 2])
    axes = axes_all[:, 2:]
    for ri, (name, r0, r1, c0, c1) in enumerate(CROPS):
        s = load_scene(data, name)
        H, W = s["gt_disp"].shape
        f, cx = s["f"], s["calib"]["cam0"][0, 2]
        gt = disp_to_depth_m(s["gt_disp"], s["f"], s["baseline_mm"], s["doffs"])
        nocc = s["nonocc"] & (gt > 0)
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        band = np.zeros((H, W), bool)
        band[r0:r1, c0:c1] = True
        g = band & nocc
        axi = axes_all[ri, 0]
        axi.imshow(cv2.cvtColor(s["left"], cv2.COLOR_BGR2RGB))
        axi.add_patch(plt.Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, ec="red", lw=1.2))
        axi.set_xticks([])
        axi.set_yticks([])
        axi.set_ylabel(name, fontsize=7)
        # magnified inset below the image: the band plus some rows of context, band outlined
        frac_h = 0.34  # inset height as a fraction of the image axes height
        rows_needed = (c1 - c0) * frac_h * H / W  # keeps the inset at the image's pixel aspect
        pad = max(0, int(round((rows_needed - (r1 - r0)) / 2)))
        y0, y1 = max(0, r0 - pad), min(H, r1 + pad)
        ins = axi.inset_axes([0.0, -frac_h - 0.04, 1.0, frac_h])
        ins.imshow(cv2.cvtColor(s["left"], cv2.COLOR_BGR2RGB), extent=(0, W, H, 0))
        ins.set_xlim(c0, c1)
        ins.set_ylim(y1, y0)
        ins.add_patch(plt.Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, ec="red", lw=0.8))
        ins.set_xticks([])
        ins.set_yticks([])
        for sp in ins.spines.values():
            sp.set_edgecolor("red")
        axi.indicate_inset_zoom(ins, edgecolor="red", lw=0.6)
        if ri == 0:
            axi.set_title("Left image\n(red box: rows analysed)", fontsize=6)
        gx, gz = (uu[g] - cx) * gt[g] / f, gt[g]
        zlo, zhi = np.percentile(gz, 0.5) * 0.9, np.percentile(gz, 99.5) * 1.1
        for ci, (title, spec) in enumerate(cols):
            ax = axes[ri, ci]
            ax.scatter(gx, gz, s=0.3, c="0.8", lw=0, rasterized=True)
            if spec is None:
                ax.scatter(gx, gz, s=0.3, c="k", lw=0, rasterized=True)
            else:
                _, tag, kind, source = spec
                _, zn = load_method(s, tag, kind, source, gt, nocc)
                h, w = zn.shape
                # native pixel centres in full-resolution coordinates
                un = (np.arange(w) + 0.5) * W / w - 0.5
                vn = (np.arange(h) + 0.5) * H / h - 0.5
                U, V = np.meshgrid(un, vn)
                sel = (V >= r0) & (V < r1) & (U >= c0) & (U < c1) & (zn > 0)
                ax.scatter((U[sel] - cx) * zn[sel] / f, zn[sel], s=0.6 if w < W else 0.3, c="C3", lw=0, rasterized=True)
            ax.set_ylim(zhi, zlo)  # near at the top of each panel
            ax.set_xticks([])
            if ri == 0:
                short = (title.replace("Depth Anything 3 Metric-L", "DA3 Metric-L")
                         .replace("Depth Anything V2-L", "DA V2-L"))
                ax.set_title(short.replace(", ", "\n"), fontsize=6)
            if ci == 0:
                ax.set_ylabel("depth Z [m]", fontsize=6)
            else:
                ax.tick_params(labelleft=False)
            ax.tick_params(labelsize=5)
    fig.text(0.5, 0.01, "lateral position X (top-down view of a band of image rows across object edges)",
             ha="center", fontsize=6)
    fig.subplots_adjust(left=0.03, right=0.995, top=0.86, bottom=0.07, hspace=0.12)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, f"paper/figures/{out}.{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
