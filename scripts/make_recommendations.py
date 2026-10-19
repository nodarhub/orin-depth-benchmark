"""Recommendations table (paper Table 1 of the results): for each frame-rate budget on one Jetson AGX
Orin, the most accurate stereo and monocular configuration whose GPU time fits the budget, from
results/v2/summary.csv (raw outputs, AbsRel over predicted pixels).

  paper/tables/recommendations.tex, results/v2/recommendations.md
"""
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUDGETS = [("30 fps", 1000 / 30), ("10 fps", 100.0), ("1 fps", 1000.0), ("no limit", float("inf"))]
# licences that permit commercial use and modification of the code and weights (Table 1 of the paper)
PERMISSIVE = ("libSGM", "Depth Anything V2-S", "Depth Anything 3", "MoGe-2", "Metric3D")


def fastest_permissive(df, sensor):
    c = df[(df.sensor == sensor) & df.label.str.startswith(PERMISSIVE)].dropna(subset=["gpu_ms"])
    return None if c.empty else c.sort_values("gpu_ms").iloc[0]


def best(df, sensor, budget, col="absrel"):
    c = df[(df.sensor == sensor) & (df.gpu_ms <= budget)].dropna(subset=[col])
    return None if c.empty else c.sort_values(col).iloc[0]


def cell(r, col="absrel", tex=True):
    if r is None:
        return "--"
    res = r.resolution.split(" (")[0]
    if tex:
        res = res.replace("×", "$\\times$").replace("≈", "$\\approx$")
    name = r.label.replace(" (GT focal)", "").replace(" (GT intrinsics)", "")
    return f"{name}, {res}: {r[col]:.2f}\\% ({r.gpu_ms:.0f}\\,ms)" if tex else \
        f"{name}, {res}: {r[col]:.2f}% ({r.gpu_ms:.0f} ms)"


def main():
    df = pd.read_csv(os.path.join(ROOT, "results/v2/summary.csv"))
    df = df[(df.variant == "raw") & ~df.label.str.contains("est. focal")]
    rows = []
    for name, b in BUDGETS:
        rows.append((name, best(df, "stereo", b), best(df, "mono", b), best(df, "mono", b, "aligned_absrel")))
    m = fastest_permissive(df, "mono")
    rows.append(("lowest GPU, open source", fastest_permissive(df, "stereo"), m, m))
    with open(os.path.join(ROOT, "paper/tables/recommendations.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}llll@{}}\n\\toprule\n")
        last = len(rows) - 1
        f.write("Budget & Best stereo & Best monocular & Best monocular, oracle scale \\\\\n"
                " & AbsRel (GPU time) & AbsRel (GPU time) & AbsRel (GPU time) \\\\\n\\midrule\n")
        for i, (name, s, m, o) in enumerate(rows):
            if i == last:
                f.write("\\midrule\n")
            f.write(f"{name} & {cell(s)} & {cell(m)} & {cell(o, 'aligned_absrel')} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(ROOT, "results/v2/recommendations.md"), "w") as f:
        f.write("| Budget (one AGX Orin) | Best stereo | Best monocular | Best monocular, oracle scale |\n|---|---|---|---|\n")
        for name, s, m, o in rows:
            f.write(f"| {name} | {cell(s, tex=False)} | {cell(m, tex=False)} | {cell(o, 'aligned_absrel', tex=False)} |\n")
    print(open(os.path.join(ROOT, "results/v2/recommendations.md")).read())


if __name__ == "__main__":
    main()
