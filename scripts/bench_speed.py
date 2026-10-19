"""Latency / throughput / power benchmark of TensorRT engines on Jetson.

For each engine: trtexec back-to-back inference for --duration seconds while
tegrastats samples power. Reports median GPU compute time (the engine's GPU
cost per frame), median host latency, throughput, mean rail power, and dynamic
energy per frame = (P_load - P_idle) / throughput.

Run with nothing else on the GPU, in MAXN with jetson_clocks if possible.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from common.power import PowerSampler, measure_idle  # noqa: E402

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


def bench(engine, duration, warmup_ms, plugins=()):
    times = engine.replace(".engine", ".times.json")
    cmd = [TRTEXEC, f"--loadEngine={engine}", f"--warmUp={warmup_ms}", f"--duration={duration}",
           "--iterations=20", "--useSpinWait", f"--exportTimes={times}", "--noDataTransfers"] + \
        [f"--staticPlugins={x}" for x in plugins]
    with PowerSampler() as ps:
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
    t = json.load(open(times))
    gpu = np.array([x["computeMs"] for x in t])
    lat = np.array([x["latencyMs"] for x in t])
    thr = float(re.search(r"Throughput: ([\d.]+) qps", out).group(1))
    return {"gpu_ms_median": float(np.median(gpu)), "gpu_ms_p90": float(np.percentile(gpu, 90)),
            "latency_ms_median": float(np.median(lat)), "throughput_fps": thr,
            "n": int(len(gpu)), **ps.summary()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engines", nargs="*", default=sorted(glob.glob(os.path.join(ROOT, "engines/*/*.engine"))))
    p.add_argument("--duration", type=int, default=20)
    p.add_argument("--warmup_ms", type=int, default=3000)
    p.add_argument("--out", default=os.path.join(ROOT, "results/speed/networks.json"))
    p.add_argument("--plugins", nargs="*", default=[], help="TensorRT plugin libraries needed by an engine")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    idle = measure_idle()
    print("idle", idle)
    rows = []
    for e in a.engines:
        r = bench(e, a.duration, a.warmup_ms, a.plugins)
        r["method"] = os.path.splitext(os.path.basename(e))[0]
        r["dyn_energy_mJ"] = (r["SOC_mW"] - idle["SOC_mW"]) / r["throughput_fps"]
        r["dyn_energy_sys_mJ"] = (r["VIN_SYS_5V0_mW"] - idle["VIN_SYS_5V0_mW"]) / r["throughput_fps"]
        rows.append(r)
        print(json.dumps(r))
        json.dump({"idle": idle, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
