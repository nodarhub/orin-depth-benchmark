"""Appendix table of the 23 Middlebury 2014 scenes with public ground truth: thumbnail, size,
GT disparity statistics, non-occluded maximum, depth range, occlusion statistics.

Writes paper/figures/thumbs/<scene>.jpg and paper/tables/dataset.tex.
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from common.middlebury import SCENES, disp_to_depth_m, load_scene  # noqa: E402


def main():
    data = os.path.join(ROOT, "data/middlebury2014")
    thumbs = os.path.join(ROOT, "paper/figures/thumbs")
    os.makedirs(thumbs, exist_ok=True)
    rows = []
    for n in SCENES:
        s = load_scene(data, n)
        g = s["gt_disp"]
        H, W = g.shape
        v = g[g > 0]
        nocc_max = np.where(s["nonocc"], g, 0).max()
        # occluded / outside the right view, as fractions of GT pixels (see common/middlebury.py)
        xr = np.round(np.arange(W)[None, :] - g)
        outside = (g > 0) & (xr < 0)
        occluded = (g > 0) & ~outside & ~s["nonocc"]
        z = disp_to_depth_m(g, s["f"], s["baseline_mm"], s["doffs"])
        zv = z[z > 0]
        th = cv2.resize(s["left"], (240, int(round(240 * H / W))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(thumbs, f"{n}.jpg"), th, [cv2.IMWRITE_JPEG_QUALITY, 85])
        rows.append(dict(name=n, W=W, H=H, ndisp=int(s["calib"]["ndisp"]), dmin=v.min(),
                         dmed=np.median(v), d99=np.percentile(v, 99), dmax=v.max(), nocc_max=nocc_max,
                         zmin=zv.min(), zmax=zv.max(), occ=occluded.sum() / (g > 0).sum() * 100,
                         out=outside.sum() / (g > 0).sum() * 100, sub=nocc_max < 255))
    rows = [r for r in rows if r["sub"]]  # the paper uses only the scenes with disparity < 256 px
    with open(os.path.join(ROOT, "paper/tables/dataset.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}cl r r rrrr r r rr@{}}\n\\toprule\n")
        f.write(" & & & & \\multicolumn{4}{c}{GT disparity [px]} & Non-occ. & Depth & Occl. & Out \\\\\n")
        f.write("\\cmidrule(lr){5-8}\n")
        f.write("Image & Scene & Size ($W\\times H$) & \\texttt{ndisp} & min & median & 99\\,\\% & max & "
                "max [px] & [m] & [\\%] & [\\%] \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"\\raisebox{{-0.45\\height}}{{\\includegraphics[height=0.72cm]{{figures/thumbs/{r['name']}.jpg}}}} & "
                    f"{r['name']} & {r['W']}$\\times${r['H']} & {r['ndisp']} & {r['dmin']:.0f} & {r['dmed']:.0f} & "
                    f"{r['d99']:.0f} & {r['dmax']:.0f} & {r['nocc_max']:.0f} & {r['zmin']:.2f}--{r['zmax']:.2f} & "
                    f"{r['occ']:.1f} & {r['out']:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"{len(rows)} scenes; depth {min(r['zmin'] for r in rows):.2f}-{max(r['zmax'] for r in rows):.2f} m; "
          f"occluded {np.mean([r['occ'] for r in rows]):.1f}%, outside {np.mean([r['out'] for r in rows]):.1f}%; "
          f"non-occ max {max(r['nocc_max'] for r in rows):.0f} px")


if __name__ == "__main__":
    main()
