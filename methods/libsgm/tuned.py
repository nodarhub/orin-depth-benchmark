"""libSGM parameters selected by tune.py (results/libsgm_tuning/scale<k>.json) and the matching
prediction tags written by run.py."""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
NAMES = {1: "full", 2: "half", 4: "quarter"}


def tuned_params(scale):
    """(P1, P2) with the lowest mean AbsRel for this scale (libSGM defaults if not tuned)."""
    f = os.path.join(ROOT, "results/libsgm_tuning", f"scale{scale}.json")
    if not os.path.exists(f):
        return 10, 120
    p1, p2 = json.load(open(f))["best"].split(",")
    return int(p1), int(p2)


def tag(scale, paths=4, p1=10, p2=120):
    return f"libsgm_{NAMES[scale]}" + ("" if paths == 8 else "_4path") + \
        ("" if (p1, p2) == (10, 120) else f"_p1_{p1}_p2_{p2}")


def tuned_tag(scale):
    return tag(scale, 4, *tuned_params(scale))
