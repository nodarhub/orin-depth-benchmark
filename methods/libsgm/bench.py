"""Speed/power of libSGM: per-frame GPU-busy time
(kernels + host<->device copies) from an Nsight Systems trace, plus tegrastats power in a loop.

  python methods/libsgm/bench.py --scale 1     # orchestrates both (Adirondack)
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))


def workload(a, gap_s, seconds):
    from sgm import LibSGM
    from run import prepare
    from common.middlebury import load_scene
    g = prepare(load_scene(os.path.join(ROOT, "data/middlebury2014"), "Adirondack"), a.scale)
    h, w = g[0].shape
    sgm = LibSGM(w, h, 256, a.paths, P1=a.p1, P2=a.p2)
    for _ in range(5):
        sgm(*g)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        sgm(*g)
        n += 1
        if gap_s:
            time.sleep(gap_s)
    return n / (time.perf_counter() - t0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=int, default=1, choices=[1, 2, 4])
    p.add_argument("--paths", type=int, default=8, choices=[4, 8])
    p.add_argument("--p1", type=int, default=10)
    p.add_argument("--p2", type=int, default=120)
    p.add_argument("--tuned", action="store_true", help="use the P1/P2 selected by tune.py for this scale")
    p.add_argument("--loop", type=float, default=0, help="internal: run workload for N seconds")
    p.add_argument("--gap_ms", type=float, default=0, help="internal: idle time between frames")
    p.add_argument("--out", default=os.path.join(ROOT, "results/speed"))
    a = p.parse_args()
    if a.tuned:
        from tuned import tuned_params
        a.p1, a.p2 = tuned_params(a.scale)
    if a.loop:
        print(json.dumps({"fps": workload(a, a.gap_ms / 1000, a.loop)}))
        return
    from common.power import PowerSampler, measure_idle
    from tuned import tag as make_tag
    tag = make_tag(a.scale, a.paths, a.p1, a.p2)
    os.makedirs(os.path.join(a.out, "nsys"), exist_ok=True)
    rep = os.path.join(a.out, "nsys", tag)
    me = [sys.executable, os.path.abspath(__file__), "--scale", str(a.scale), "--paths", str(a.paths),
          "--p1", str(a.p1), "--p2", str(a.p2)]
    # frames separated by idle gaps so nsys_frame_stats can segment them
    subprocess.run(["nsys", "profile", "-t", "cuda", "--cuda-memory-usage=false", "-f", "true", "-o", rep]
                   + me + ["--loop", "15", "--gap_ms", "50"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["nsys", "export", "-t", "sqlite", "-f", "true", "-o", rep + ".sqlite", rep + ".nsys-rep"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    stats = json.loads(subprocess.run([sys.executable, os.path.join(ROOT, "scripts/nsys_frame_stats.py"),
                                       rep + ".sqlite", "--gap_ms", "20", "--skip", "5"],
                                      capture_output=True, text=True).stdout)
    idle = measure_idle()
    with PowerSampler() as ps:
        fps = json.loads(subprocess.run(me + ["--loop", "20"], capture_output=True, text=True).stdout)["fps"]
    pw = ps.summary()
    r = {"method": tag, "gpu_ms_median": stats["busy_ms_median"], "gpu_ms_p90": stats["busy_ms_p90"],
         "gpu_span_ms_median": stats["span_ms_median"], "loop_fps": fps, **pw,
         "throughput_fps": 1000.0 / stats["busy_ms_median"],
         "dyn_energy_mJ": (pw["SOC_mW"] - idle["SOC_mW"]) / fps, "idle": idle}
    print(json.dumps(r, indent=1))
    json.dump(r, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
