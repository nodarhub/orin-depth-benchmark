"""Resolution scaling (paper Figure 1 and the scale-drift table): AbsRel against the number of pixels
each method processes, for every method run at more than one input size, from results/v2/summary.csv
and results/v2/per_scene.csv (scripts/score_v2.py).

  paper/figures/scaling.{pdf,png}   (a) as deployed, (b) monocular with per-image oracle scale
  paper/tables/scale_drift.tex      median GT/prediction scale of the monocular models per input size
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# label -> (colour, marker); monocular with the true focal length where the model takes one
SERIES = {"libSGM": ("#008080", "P"),
          "Fast-FoundationStereo 23-36-37": ("C4", "D"), "Fast-FoundationStereo 20-30-48": ("C1", "D"),
          "C-Fast-FoundationStereo": ("#e7298a", "D"),
          "Depth Anything V2-S": ("C5", "^"), "Depth Anything V2-B": ("C6", "^"), "Depth Anything V2-L": ("C7", "^"),
          "Depth Anything 3 Metric-L (GT focal)": ("k", "*"),
          "MoGe-2-S (GT focal)": ("#d4d46a", "p"), "MoGe-2-B (GT focal)": ("#bcbd22", "p"),
          "MoGe-2-L (GT focal)": ("#6b6c0e", "p"),
          "Metric3D v2-S (GT focal)": ("#f4a6c6", "h"), "Metric3D v2-L (GT focal)": ("#c2185b", "h"),
          "UniDepth V2-S (GT intrinsics)": ("#9ecae1", "o"), "UniDepth V2-B (GT intrinsics)": ("#4292c6", "o"),
          "UniDepth V2-L (GT intrinsics)": ("#08306b", "o")}
SHORT = {"Fast-FoundationStereo 23-36-37": "FFS 23-36-37", "Fast-FoundationStereo 20-30-48": "FFS 20-30-48",
         "C-Fast-FoundationStereo": "C-FFS (commercial)",
         "Depth Anything 3 Metric-L (GT focal)": "Depth Anything 3 Metric-L",
         "MoGe-2-S (GT focal)": "MoGe-2-S", "MoGe-2-B (GT focal)": "MoGe-2-B", "MoGe-2-L (GT focal)": "MoGe-2-L",
         "Metric3D v2-S (GT focal)": "Metric3D v2-S", "Metric3D v2-L (GT focal)": "Metric3D v2-L",
         "UniDepth V2-S (GT intrinsics)": "UniDepth V2-S", "UniDepth V2-B (GT intrinsics)": "UniDepth V2-B",
         "UniDepth V2-L (GT intrinsics)": "UniDepth V2-L"}
FULL_MP = 1988 * 2884 / 1e6


def megapixels(res):
    """Processed pixels in MP from a resolution label ('full (≈1990×2880)', '≈995×1440 (half)', '518×770 (native)')."""
    h, w = (int(x) for x in re.findall(r"(\d+)×(\d+)", res)[0])
    return h * w / 1e6


def main():
    df = pd.read_csv(os.path.join(ROOT, "results/v2/summary.csv"))
    df = df[(df.variant == "raw") & df.label.str.replace(" (native)", "", regex=False).isin(SERIES)].copy()
    df["series"] = df.label.str.replace(" (native)", "", regex=False)
    df["mp"] = df.resolution.map(megapixels)
    plt.rcParams.update({"font.size": 8, "figure.dpi": 200})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    for ax, oracle, title in [(axes[0], False, "(a) as deployed"),
                              (axes[1], True, "(b) monocular with oracle per-image scale")]:
        for name, (c, mk) in SERIES.items():
            g = df[df.series == name].sort_values("mp")
            if g.empty:
                continue
            mono = g.sensor.iloc[0] == "mono"
            y = g.aligned_absrel if (oracle and mono) else g.absrel
            ax.plot(g.mp, y, color=c, marker=mk, ms=4.5, lw=1.0, ls="-" if not mono else "--",
                    mfc=c if not (oracle and mono) else "none", label=SHORT.get(name, name))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(FULL_MP, color="gray", lw=0.6, ls="-.")
        ax.text(FULL_MP * 0.97, 0.22, "full resolution", rotation=90, fontsize=5.5, color="gray", ha="right", va="bottom")
        ax.set_xlabel("pixels processed per image [MP]")
        ax.set_title(title, fontsize=8)
        ax.grid(alpha=0.3, which="both")
        ax.set_ylim(0.2, 250)
    axes[0].set_ylabel("AbsRel depth error, non-occluded [%]")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, f"paper/figures/scaling.{ext}"))
    plt.close(fig)

    # ---- scale drift: median over scenes of the per-image GT/prediction median ratio -------------
    ps = pd.read_csv(os.path.join(ROOT, "results/v2/per_scene.csv"))
    ps = ps[ps.variant == "raw"]
    mono = df[df.sensor == "mono"]
    with open(os.path.join(ROOT, "paper/tables/scale_drift.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n")
        f.write("Model & Input & MP & Scale GT/pred. & AbsRel [\\%] & Oracle AbsRel [\\%] & Density [\\%] \\\\\n\\midrule\n")
        last = None
        for name in SERIES:
            g = mono[mono.series == name].sort_values("mp")
            if g.empty:
                continue
            if last is not None:
                f.write("\\addlinespace\n")
            last = name
            for _, r in g.iterrows():
                sc = ps[ps.method == r.method].median_scale.median()
                res = r.resolution.split(" (")[0].replace("×", "$\\times$")
                f.write(f"{SHORT.get(name, name)} & {res} & {r.mp:.2f} & {sc:.2f} & {r.absrel:.1f} & "
                        f"{r.aligned_absrel:.1f} & {r.density * 100:.0f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(df[["series", "resolution", "mp", "absrel", "aligned_absrel"]].sort_values(["series", "mp"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
