"""Grid search of libSGM's smoothness penalties P1 and P2, separately per input scale, minimising the
mean AbsRel (non-occluded GT, predicted pixels, as in scripts/score_v2.py) over the evaluation scenes.

  python methods/libsgm/tune.py --scale 1 --paths 4

There is no separate training set, so the selected parameters are tuned on the evaluation scenes;
the output also reports leave-one-scene-out cross-validation (parameters chosen on 9 scenes, scored on
the 10th) to show how much of the gain is due to that.
Writes results/libsgm_tuning/scale<k>.json with every configuration's per-scene metrics.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from sgm import LibSGM  # noqa: E402
from run import prepare, to_full_res  # noqa: E402
from common.metrics import all_metrics  # noqa: E402
from common.middlebury import load_scene  # noqa: E402

P1S = [1, 2, 3, 4, 6, 10, 15, 20, 30]
P2S = [40, 60, 90, 120, 160, 200, 224]  # 224: 8-bit aggregated cost limit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=int, default=1, choices=[1, 2, 4])
    p.add_argument("--paths", type=int, default=4, choices=[4, 8])
    p.add_argument("--data", default=os.path.join(ROOT, "data/middlebury2014"))
    p.add_argument("--out", default=os.path.join(ROOT, "results/libsgm_tuning"))
    a = p.parse_args()
    scenes = json.load(open(os.path.join(ROOT, "results/v2/scene_sets.json")))["subset_lt255"]
    grid = [(p1, p2) for p1, p2 in itertools.product(P1S, P2S) if p1 < p2]
    res = {f"{p1},{p2}": {} for p1, p2 in grid}
    for name in scenes:
        s = load_scene(a.data, name)
        s_nocc = dict(s, gt_disp=np.where(s["nonocc"], s["gt_disp"], 0))
        s_nocc.pop("nonocc")
        g = prepare(s, a.scale)
        h, w = g[0].shape
        for p1, p2 in grid:
            sgm = LibSGM(w, h, 256, a.paths, P1=p1, P2=p2)
            disp = to_full_res(sgm(*g), s["gt_disp"].shape)
            del sgm
            m = all_metrics(disp, None, s_nocc)
            res[f"{p1},{p2}"][name] = {k: float(m[k]) for k in ("absrel", "density", "bad2", "d_1.05")}
        print(name, "done", flush=True)
    mean = {k: np.mean([v[n]["absrel"] for n in scenes]) for k, v in res.items()}
    best = min(mean, key=mean.get)
    # leave-one-scene-out: choose on 9 scenes, score the 10th
    loo = []
    for held in scenes:
        rest = [n for n in scenes if n != held]
        k = min(res, key=lambda k: np.mean([res[k][n]["absrel"] for n in rest]))
        loo.append(dict(held_out=held, chosen=k, absrel=res[k][held]["absrel"]))
    out = dict(scale=a.scale, paths=a.paths, scenes=scenes, grid=grid, best=best,
               best_mean_absrel=mean[best], default_mean_absrel=mean.get("10,120"),
               best_mean_density=np.mean([res[best][n]["density"] for n in scenes]),
               default_mean_density=np.mean([res["10,120"][n]["density"] for n in scenes]),
               loo_mean_absrel=float(np.mean([x["absrel"] for x in loo])), loo=loo, per_config=res)
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, f"scale{a.scale}.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_config", "grid", "scenes")}, indent=1, default=float))
    print("top 5:", sorted(mean.items(), key=lambda kv: kv[1])[:5])


if __name__ == "__main__":
    main()
