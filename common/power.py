"""Sample Jetson power rails with tegrastats in the background."""
import re
import subprocess
import tempfile
import time

import numpy as np

RAILS = ("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0")


class PowerSampler:
    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms

    def __enter__(self):
        self.log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
        self.proc = subprocess.Popen(["tegrastats", "--interval", str(self.interval_ms)],
                                     stdout=self.log, stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait()
        self.log.flush()
        self.log.seek(0)
        self.samples = {r: [] for r in RAILS}
        self.gpu_util = []
        for line in self.log:
            for r in RAILS:
                m = re.search(rf"{r} (\d+)mW", line)
                if m:
                    self.samples[r].append(int(m.group(1)))
            m = re.search(r"GR3D_FREQ (\d+)%", line)
            if m:
                self.gpu_util.append(int(m.group(1)))

    def summary(self, trim=0.1):
        out = {}
        for r, v in self.samples.items():
            v = np.array(v, float)
            k = int(len(v) * trim)
            v = v[k:len(v) - k] if len(v) > 2 * k + 2 else v
            out[f"{r}_mW"] = float(v.mean()) if len(v) else float("nan")
        out["SOC_mW"] = out["VDD_GPU_SOC_mW"] + out["VDD_CPU_CV_mW"]
        out["gpu_util"] = float(np.mean(self.gpu_util)) if self.gpu_util else float("nan")
        return out


def measure_idle(seconds=15):
    with PowerSampler() as ps:
        time.sleep(seconds)
    return ps.summary()
