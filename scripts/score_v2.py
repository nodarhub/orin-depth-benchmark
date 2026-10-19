"""Second evaluation protocol (paper v2): one configuration per method at full resolution, or at
the method's maximum input size if it cannot run at full resolution; ground truth restricted to
non-occluded pixels (including the oracle-scale fit of monocular methods); one scene set (the
scenes, and the scenes whose non-occluded GT disparity stays below 255 px).

Sparse stereo outputs are scored raw and MiddEval3-style hole-filled; the figure shows raw only.
Reads the float32 predictions in results/preds/<tag>/<scene>.npy and the speed JSONs.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from common.metrics import all_metrics, fill_holes  # noqa: E402
from common.middlebury import SCENES, load_scene  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "methods/libsgm"))
from tuned import tuned_tag  # noqa: E402

# tag -> (label, sensor, processing resolution, sparse?)
METHODS = {
    # Fixstars libSGM, 4 paths, P1/P2 tuned per scale (methods/libsgm/tune.py, run.py --p1 --p2)
    **{tuned_tag(k): ("libSGM", "stereo", r, True)
       for k, r in [(1, "full (≈1990×2880)"), (2, "≈995×1440 (half)"), (4, "≈497×720 (quarter)")]},
    "ffs_23-36-37_it8_768x1152": ("Fast-FoundationStereo 23-36-37", "stereo", "768×1152 (max)", False),
    "ffs_20-30-48_it4_768x1152": ("Fast-FoundationStereo 20-30-48", "stereo", "768×1152 (max)", False),
    "ffs_23-36-37_it8_512x768": ("Fast-FoundationStereo 23-36-37", "stereo", "512×768", False),
    "ffs_20-30-48_it4_512x768": ("Fast-FoundationStereo 20-30-48", "stereo", "512×768", False),
    "ffs_23-36-37_it8_384x576": ("Fast-FoundationStereo 23-36-37", "stereo", "384×576", False),
    "ffs_20-30-48_it4_384x576": ("Fast-FoundationStereo 20-30-48", "stereo", "384×576", False),
    # commercially licensed checkpoint (nvidia/c-fast-foundationstereo), same architecture as 23-36-37
    "ffs_c-ffs_it8_768x1152": ("C-Fast-FoundationStereo", "stereo", "768×1152 (max)", False),
    "ffs_c-ffs_it8_512x768": ("C-Fast-FoundationStereo", "stereo", "512×768", False),
    "ffs_c-ffs_it8_384x576": ("C-Fast-FoundationStereo", "stereo", "384×576", False),
    # NVIDIA ESS / Light ESS (Isaac ROS, methods/ess/run.py): fixed input, dense output
    "ess_576x960": ("ESS (NVIDIA)", "stereo", "576×960 (fixed)", False),
    "light_ess_288x480": ("Light ESS (NVIDIA)", "stereo", "288×480 (fixed)", False),
    # Metric3D v2 (methods/metric3d/run.py), true focal length
    **{f"metric3d_{v}_{r}_fp32": (f"Metric3D v2-{n} (GT focal)", "mono", lab, False)
       for v, n in [("vit_small", "S"), ("vit_large", "L")]
       for r, lab in [("616x1064", "616×1064 (native)"), ("1232x2128", "1232×2128"), ("1988x2884", "1988×2884 (full)")]},
    "depthpro_1536_fp32": ("Depth Pro (GT focal)", "mono", "1536×1536 (fixed)", False),
    "depthpro_1536_fp32_estf": ("Depth Pro (est. focal)", "mono", "1536×1536 (fixed)", False),
    # Depth Anything 3 metric (methods/da3/run.py --res 504/1008/0): GT focal for metric scale
    **{f"da3metric_large_{r}_fp32": ("Depth Anything 3 Metric-L (GT focal)", "mono", lab, False)
       for r, lab in [("350x504", "350×504 (native)"), ("700x1008", "700×1008"), ("1988x2884", "1988×2884 (full)")]},
    # MoGe-2 (methods/moge2/run.py --num_tokens 3600/14400/0): default level 3600 tokens = native
    **{f"moge2_{v}_{r}_fp32{sfx}": (f"MoGe-2-{n} ({lab})", "mono", rl, False)
       for v, n in [("vits", "S"), ("vitb", "B"), ("vitl", "L")]
       for sfx, lab in [("", "GT focal"), ("_estf", "est. focal")]
       for r, rl in [("700x1008", "700×1008 (native)"), ("1400x2030", "1400×2030"), ("1988x2884", "1988×2884 (full)")]},
}
SPEED_KEY = {"depthpro_1536_fp32": "depthpro_1536", "depthpro_1536_fp32_estf": "depthpro_1536",
             "ess_576x960": "ess", "light_ess_288x480": "light_ess",
             **{f"metric3d_{v}_{r}_fp32": f"metric3d_{v}_{r}" for v in ("vit_small", "vit_large")
                for r in ("616x1064", "1232x2128", "1988x2884")},
             **{f"da3metric_large_{r}_fp32": f"da3metric_large_{r}" for r in ("350x504", "700x1008", "1988x2884")},
             **{f"moge2_{v}_{r}_fp32{sfx}": f"moge2_{v}_{r}"
                for v in ("vits", "vitb", "vitl") for sfx in ("", "_estf")
                for r in ("700x1008", "1400x2030", "1988x2884")}}


def add_dav2(res):
    for t in [t for t in list(METHODS) if not os.path.isdir(os.path.join(ROOT, "results/preds", t))]:
        del METHODS[t]  # skip configurations that were not run
    for enc, name in [("vits", "S"), ("vitb", "B"), ("vitl", "L")]:
        tag = f"dav2_{enc}_{res}"
        # prefer FP32 accuracy where it exists (FP16 engine numerically unreliable; see FP16 check)
        if os.path.isdir(os.path.join(ROOT, "results/preds", tag + "_fp32")):
            SPEED_KEY[tag + "_fp32"] = tag
            tag = tag + "_fp32"
        if os.path.isdir(os.path.join(ROOT, "results/preds", tag)):
            h, w = res.split("x")
            METHODS[tag] = (f"Depth Anything V2-{name}", "mono",
                            f"{h}×{w}" + (" (full)" if res == "1988x2884" else ""), False)


def add_unidepth():
    """UniDepth V2 chooses its processing size from a pixel budget; read it from run.py's output."""
    for v, n in [("vits14", "S"), ("vitb14", "B"), ("vitl14", "L")]:
        for sz in ("native", "2x", "full"):
            tag = f"unidepthv2_{v}_{sz}_fp32"
            acc = os.path.join(ROOT, "results/accuracy", tag + ".json")
            if not (os.path.isdir(os.path.join(ROOT, "results/preds", tag)) and os.path.exists(acc)):
                continue
            r0 = json.load(open(acc))[0]
            h, w = r0["proc_h"], r0["proc_w"]
            res = f"{h}×{w}" + {"native": " (native)", "2x": "", "full": " (full)"}[sz]
            METHODS[tag] = (f"UniDepth V2-{n} (GT intrinsics)", "mono", res, False)
            SPEED_KEY[tag] = f"unidepthv2_{v}_{h}x{w}"


def register_methods(dav2_res="1988x2884", dav2_native="518x770"):
    """Complete METHODS with the Depth Anything V2 runs that exist and drop configurations not run."""
    add_dav2(dav2_res)
    add_unidepth()
    add_dav2("1008x1512")  # twice the native size (S and L only)
    if dav2_native:
        for enc, name in [("vits", "S"), ("vitb", "B"), ("vitl", "L")]:
            tag = f"dav2_{enc}_{dav2_native}"
            if os.path.isdir(os.path.join(ROOT, "results/preds", tag)):
                h, w = dav2_native.split("x")
                METHODS[tag] = (f"Depth Anything V2-{name} (native)", "mono", f"{h}×{w} (native)", False)


def score(tag, scene_names, data):
    rows = []
    mono = METHODS[tag][1] == "mono"
    for name in scene_names:
        s = load_scene(data, name)
        s_nocc = dict(s, gt_disp=np.where(s["nonocc"], s["gt_disp"], 0))
        s_nocc.pop("nonocc")  # score once, on the non-occluded GT only
        pred = np.load(os.path.join(ROOT, "results/preds", tag, name + ".npy")).astype(np.float32)
        if mono:
            m = all_metrics(None, pred, s_nocc, mono=True)
        else:
            m = all_metrics(pred, None, s_nocc)
        m.update(scene=name, method=tag, variant="raw")
        rows.append(m)
        if METHODS[tag][3]:
            mf = all_metrics(fill_holes(pred), None, s_nocc)
            mf.update(scene=name, method=tag, variant="filled")
            rows.append(mf)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(ROOT, "data/middlebury2014"))
    p.add_argument("--dav2_res", default="1988x2884")
    p.add_argument("--dav2_native", default="518x770", help="also score Depth Anything V2 at its native input")
    p.add_argument("--out", default=os.path.join(ROOT, "results/v2"))
    a = p.parse_args()
    register_methods(a.dav2_res, a.dav2_native)
    os.makedirs(a.out, exist_ok=True)
    subset = []
    for n in SCENES:
        s = load_scene(a.data, n)
        if np.where(s["nonocc"], s["gt_disp"], 0).max() < 255:
            subset.append(n)
    json.dump({"subset_lt255": subset}, open(os.path.join(a.out, "scene_sets.json"), "w"), indent=1)
    rows = []
    for tag in METHODS:  # the paper uses only the scenes within a 256 px disparity range
        rows += score(tag, subset, a.data)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "per_scene.csv"), index=False)
    speed = {}
    for f in ["networks.json", "dav2_full.json", "ffs_scaling.json", "mono_v3.json", "mono_hires.json", "mono_hires2.json",
              "new_methods.json", "ess_engines.json", "cffs.json"]:
        fp = os.path.join(ROOT, "results/speed", f)
        if not os.path.exists(fp):
            continue
        for r in json.load(open(fp))["rows"]:
            speed[r["method"]] = r
    for t in list(METHODS) + list(SPEED_KEY.values()):
        f = os.path.join(ROOT, "results/speed", t + ".json")
        if os.path.exists(f):
            speed[t] = json.load(open(f))
    out = []
    for setname, scenes in [("lt255", subset)]:
        sub = df[df.scene.isin(scenes)]
        agg = sub.drop(columns=["scene"]).groupby(["method", "variant"]).mean(numeric_only=True).reset_index()
        agg["scene_set"] = setname
        out.append(agg)
    res = pd.concat(out)
    lab = {t: v for t, v in METHODS.items()}
    res["label"] = res.method.map(lambda t: lab[t][0])
    res["sensor"] = res.method.map(lambda t: lab[t][1])
    res["resolution"] = res.method.map(lambda t: lab[t][2])
    sp = res.method.map(lambda t: speed.get(SPEED_KEY.get(t, t), {}))
    res["gpu_ms"] = sp.map(lambda d: d.get("gpu_ms_median", np.nan))
    res["mJ"] = sp.map(lambda d: d.get("dyn_energy_mJ", np.nan))
    res.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    cols = ["scene_set", "label", "variant", "resolution", "density", "bad2", "bad2_nonocc_pred", "absrel",
            "d_1.05", "aligned_absrel", "aligned_d_1.05", "gpu_ms", "mJ"]
    pd.set_option("display.width", 250)
    print(res[cols].round(2).sort_values(["scene_set", "gpu_ms"]).to_string(index=False))
    print("subset (<255 px):", subset)


if __name__ == "__main__":
    main()
