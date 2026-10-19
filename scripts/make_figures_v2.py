"""Tables and figure for the main evaluation (results/v2/summary.csv from score_v2.py).

  paper/tables/v2_main.tex       accuracy + cost per frame, both scene sets
  paper/tables/v2_cost.tex       Orins needed and system cost (cameras + compute)
  paper/figures/v2_absrel_vs_gpu.{pdf,png}   raw results only (hole-filled rows are table-only)
  results/v2/main_table.md, results/v2/breakeven.csv
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CAMS = [("\\$20 per camera", 20), ("\\$500 per camera", 500)]  # range of camera prices
STYLE = {"libSGM": ("#008080", "P"),
         "C-Fast-FoundationStereo": ("#e7298a", "D"),
         "ESS (NVIDIA)": ("#76b900", "X"), "Light ESS (NVIDIA)": ("#4a7300", "X"),
         "Metric3D v2-S (GT focal)": ("#f4a6c6", "h"), "Metric3D v2-L (GT focal)": ("#c2185b", "h"),
         "UniDepth V2-S (GT intrinsics)": ("#9ecae1", "o"), "UniDepth V2-B (GT intrinsics)": ("#4292c6", "o"),
         "UniDepth V2-L (GT intrinsics)": ("#08306b", "o"),
         "Fast-FoundationStereo 23-36-37": ("C4", "D"), "Fast-FoundationStereo 20-30-48": ("C1", "D"),
         "Depth Pro (GT focal)": ("C0", "s"), "Depth Pro (est. focal)": ("C9", "s"),
         "Depth Anything V2-S": ("C5", "^"), "Depth Anything V2-B": ("C6", "^"), "Depth Anything V2-L": ("C7", "^"),
         # native 518x770 input: same colour per encoder, downward triangle
         "Depth Anything V2-S (native)": ("C5", "v"), "Depth Anything V2-B (native)": ("C6", "v"),
         "Depth Anything V2-L (native)": ("C7", "v"),
         "Depth Anything 3 Metric-L (GT focal)": ("k", "*"),
         # MoGe-2: olive shades by size; pentagon = GT focal, hexagon = own focal estimate
         **{f"MoGe-2-{n} ({lab})": (c, mk) for n, c in [("S", "#d4d46a"), ("B", "#bcbd22"), ("L", "#6b6c0e")]
            for lab, mk in [("GT focal", "p"), ("est. focal", "h")]}}


def fmt(v, nd=1):
    return "--" if pd.isna(v) else f"{v:,.{nd}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--module_price", type=float, default=2999.0)
    p.add_argument("--fps", type=float, nargs="*", default=[10, 30])
    a = p.parse_args()
    df = pd.read_csv(os.path.join(ROOT, "results/v2/summary.csv"))
    df["density"] *= 100
    figdir, tabdir = os.path.join(ROOT, "paper/figures"), os.path.join(ROOT, "paper/tables")
    df = df[df.scene_set == "lt255"]  # the paper uses only the 10 scenes with disparity < 256 px
    order = df[(df.variant == "raw")].sort_values(["sensor", "gpu_ms"], ascending=[False, True])
    keys = [(r.method, v) for _, r in order.iterrows() for v in ("raw", "filled")
            if ((df.method == r.method) & (df.variant == v)).any()]

    def row(m, v):
        return df[(df.method == m) & (df.variant == v)].iloc[0]

    # ---- main table -----------------------------------------------------------------------------
    with open(os.path.join(tabdir, "v2_main.tex"), "w") as f:
        f.write("\\begin{tabular}{llrrrrrrr}\n\\toprule\n")
        hdr = "Dens. & bad-2 & AbsRel & $\\delta_{1.05}$ & AbsRel$^{\\dagger}$"
        f.write(f"Method & Resolution & {hdr} & GPU ms & J/frame \\\\\n")
        f.write(" & & [\\%] & [\\%] & [\\%] & [\\%] & [\\%] & & \\\\\n\\midrule\n")
        last_sensor = None
        for m, v in keys:
            A = row(m, v)
            if last_sensor and A.sensor != last_sensor:
                f.write("\\midrule\n")
            last_sensor = A.sensor
            lab = A.label + (" (hole-filled)" if v == "filled" else "")
            cells = [lab, A.resolution.replace("×", "$\\times$").replace("≈", "$\\approx$")]
            cells += [fmt(A.density, 0), fmt(A.bad2), fmt(A.absrel, 2), fmt(A["d_1.05"]),
                      fmt(A.aligned_absrel, 2) if A.sensor == "mono" else "--"]
            cells += [fmt(A.gpu_ms), fmt(A.mJ / 1000, 2)]
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(ROOT, "results/v2/main_table.md"), "w") as f:
        f.write("| Method | Resolution | Dens. % | bad-2 % | AbsRel % | AbsRel % oracle scale | GPU ms | J/frame |\n"
                "|---|---|---:|---:|---:|---:|---:|---:|\n")
        for m, v in keys:
            A = row(m, v)
            orc = fmt(A.aligned_absrel, 2) if A.sensor == "mono" else "--"
            f.write(f"| {A.label}{' (hole-filled)' if v == 'filled' else ''} | {A.resolution} | "
                    f"{fmt(A.density, 0)} | {fmt(A.bad2)} | {fmt(A.absrel, 2)} | {orc} | "
                    f"{fmt(A.gpu_ms)} | {fmt(A.mJ / 1000, 2)} |\n")

    # ---- figure: raw only -----------------------------------------------------------------------
    # labels that would collide with a neighbouring point: (dx, dy, ha) in points; None = no label
    # (FFS 20-30-48 shares 23-36-37's resolutions)
    LABEL_OFFSET = {"MoGe-2-S (GT focal)": None, "MoGe-2-B (GT focal)": None,  # same input as MoGe-2-L
                    "UniDepth V2-S (GT intrinsics)": None, "UniDepth V2-B (GT intrinsics)": None,  # as V2-L
                    "Metric3D v2-S (GT focal)": None,  # same input as Metric3D v2-L
                    "libSGM": (0, 5, "center"),
                    "Fast-FoundationStereo 20-30-48": None,
                    "Fast-FoundationStereo 23-36-37": (0, -9, "center"),
                    "C-Fast-FoundationStereo": None}
    plt.rcParams.update({"font.size": 8, "figure.dpi": 200})
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6))
    axes = [ax]
    for ax, title in zip(axes, ["10 Middlebury 2014 scenes with disparity < 256 px"]):
        sub = df[df.variant == "raw"]
        # MoGe-2 with its own focal estimate is in the tables only (keeps the plot readable)
        sub = sub[~sub.label.str.match(r"MoGe-2-. \(est\. focal\)")]
        # resolution sweeps are plotted for the largest models only (MoGe-2-L, UniDepth V2-L, Metric3D v2-L,
        # Depth Anything 3)
        sub = sub[~(sub.label.str.match(r"(MoGe-2-[SB]|UniDepth V2-[SB]|Metric3D v2-S) ") & ~sub.resolution.str.contains("native"))]
        for _, r in sub.iterrows():
            c, mk = STYLE.get(r.label, ("k", "o"))
            # C-FFS coincides with FFS 23-36-37: draw it smaller and on top so both stay visible
            cffs = r.label == "C-Fast-FoundationStereo"
            ax.plot(r.gpu_ms, r.absrel, marker=mk, color=c, ms=3 if cffs else 5, ls="none", label=r.label,
                    zorder=3 if cffs else 2)
            if r.sensor == "mono":
                ax.plot(r.gpu_ms, r.aligned_absrel, marker=mk, color=c, mfc="none", ms=5, ls="none")
                ax.plot([r.gpu_ms] * 2, [r.aligned_absrel, r.absrel], color=c, lw=0.5, ls=":")
            res = r.resolution
            res = res[res.index("(") + 1:res.index(")")] if res.startswith("full") else res.split(" (")[0]
            off = LABEL_OFFSET.get(r.label, (4, 2, "left"))
            if off is None:
                continue
            dx, dy, ha = off
            ax.annotate(res, (r.gpu_ms, r.absrel), fontsize=4.5, color=c, ha=ha,
                        xytext=(dx, dy), textcoords="offset points")
        # connect input resolutions of the same network to show how it scales
        for lab_, g in sub[sub.label.str.startswith(("Fast-FoundationStereo", "C-Fast-FoundationStereo", "Depth Anything 3", "MoGe-2", "libSGM",
                                                     "Metric3D", "UniDepth"))].groupby("label"):
            g = g.sort_values("gpu_ms")
            ax.plot(g.gpu_ms, g.absrel, color=STYLE[lab_][0], lw=0.6, alpha=0.6, zorder=1)
        # ESS and Light ESS: the same network family at two input sizes
        g = sub[sub.label.isin(["ESS (NVIDIA)", "Light ESS (NVIDIA)"])].sort_values("gpu_ms")
        if len(g) == 2:
            ax.plot(g.gpu_ms, g.absrel, color=STYLE["ESS (NVIDIA)"][0], lw=0.6, alpha=0.6, zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.2)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("GPU time per frame on Jetson AGX Orin [ms]")
        ax.grid(alpha=0.3, which="both")
        for x, txt in [(1000 / 30, "30 fps"), (1000 / 10, "10 fps")]:
            ax.axvline(x, color="gray", lw=0.6, ls="-.")
            ax.text(x * 1.04, 70, txt, fontsize=5.5, color="gray", va="top")
    axes[0].set_ylabel("AbsRel depth error, non-occluded [%]")
    h, l = axes[0].get_legend_handles_labels()
    uniq = dict(zip(l, h))
    uniq["oracle scale (hollow)"] = plt.Line2D([], [], marker="s", color="0.4", mfc="none", ls="none", ms=5)
    fig.legend(uniq.values(), uniq.keys(), fontsize=5.5, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.2, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"v2_absrel_vs_gpu.{ext}"))
    plt.close(fig)

    # ---- cost -----------------------------------------------------------------------------------
    raw = df[df.variant == "raw"].copy()
    stereo = raw[raw.sensor == "stereo"]
    cheapest = stereo.sort_values("gpu_ms").iloc[0]
    be = []
    for _, r in raw[raw.sensor == "mono"].iterrows():
        for _, s_ in stereo.iterrows():
            for F in a.fps:
                be.append({"mono": r.label, "stereo": s_.label, "fps": F,
                           "breakeven_camera_usd": (r.gpu_ms - s_.gpu_ms) * F / 1000 * a.module_price})
    pd.DataFrame(be).to_csv(os.path.join(ROOT, "results/v2/breakeven.csv"), index=False)
    with open(os.path.join(tabdir, "v2_cost.tex"), "w") as f:
        f.write("\\begin{tabular}{llrr" + "r" * len(CAMS) + "r}\n\\toprule\n")
        f.write("Method & Resolution & Orins @10\\,fps & Orins @30\\,fps & " + " & ".join(
            f"\\${{}}@30\\,fps, {c}" for c, _ in CAMS) + " & Break-even camera @30\\,fps \\\\\n\\midrule\n")
        # the estimated-focal-length variants (Depth Pro, MoGe-2) have the same GPU time as the GT-focal rows
        for _, r in raw[~raw.label.str.contains("est. focal")].sort_values(
                ["sensor", "gpu_ms"], ascending=[False, True]).iterrows():
            u = [r.gpu_ms * F / 1000 for F in a.fps]
            ncam = 2 if r.sensor == "stereo" else 1
            tot = [u[-1] * a.module_price + ncam * cp for _, cp in CAMS]
            bev = ((r.gpu_ms - cheapest.gpu_ms) * a.fps[-1] / 1000 * a.module_price) if r.sensor == "mono" else None
            res = r.resolution.replace("×", "$\\times$").replace("≈", "$\\approx$")
            f.write(" & ".join([r.label, res] + [f"{x:.2f}" for x in u] + [f"{x:,.0f}" for x in tot]
                               + ["--" if bev is None else (f"\\${bev:,.0f}" if bev >= 0
                                                            else f"$-$\\${-bev:,.0f}")]) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("cheapest stereo:", cheapest.label, cheapest.gpu_ms)
    print(pd.DataFrame(be)[lambda d: d.stereo == cheapest.label].round(0).to_string(index=False))


if __name__ == "__main__":
    main()
