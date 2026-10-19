"""NVIDIA ESS and Light ESS stereo networks (Isaac ROS, NGC model nvidia/isaac/dnn_stereo_disparity
v4.1.0, NVIDIA TAO model EULA): accuracy on Middlebury from the TensorRT FP16 engines built with the
ESS plugin library. Both images are resized to the fixed network input (ESS 576x960, Light ESS
288x480; this changes the aspect ratio from 3:2 to 5:3, which does not affect horizontal disparity),
normalised with mean 0.5 / std 0.5 as on the model card; the dense disparity is rescaled to
full-resolution pixels and upsampled bilinearly like the other networks.

  python methods/ess/run.py --model ess --save_preds --save_native
"""
import argparse
import ctypes
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
ESS_DIR = os.path.join(ROOT, "weights/ess/dnn_stereo_disparity_v4.1.0_onnx")
PLUGIN = os.path.join(ESS_DIR, "plugins/aarch64/ess_plugins.so")
ctypes.CDLL(PLUGIN, mode=ctypes.RTLD_GLOBAL)  # registers the ESS TensorRT plugins
import tensorrt as trt  # noqa: E402
trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), "")
from common.metrics import all_metrics  # noqa: E402
from common.middlebury import SCENES, load_scene  # noqa: E402
from common.trt_runner import TRTRunner  # noqa: E402


def prep(bgr, h, w):
    rgb = cv2.cvtColor(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return ((rgb.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["ess", "light_ess"], default="ess")
    p.add_argument("--data", default=os.path.join(ROOT, "data/middlebury2014"))
    p.add_argument("--save_preds", action="store_true")
    p.add_argument("--save_native", action="store_true")
    p.add_argument("--scenes", nargs="*", default=SCENES)
    a = p.parse_args()
    runner = TRTRunner(os.path.join(ROOT, "engines/ess", f"{a.model}.engine"))
    _, _, h, w = runner.input_shape()
    tag = f"{a.model}_{h}x{w}"
    rows = []
    for name in a.scenes:
        s = load_scene(a.data, name)
        H, W = s["left"].shape[:2]
        out = runner(prep(s["left"], h, w), prep(s["right"], h, w))
        disp = out["output_left"] if "output_left" in out else out[list(out)[0]]
        native = disp.reshape(h, w) * (W / w)  # disparity in full-resolution pixels
        full = cv2.resize(native, (W, H), interpolation=cv2.INTER_LINEAR)
        m = all_metrics(full, None, s)
        m.update(scene=name, method=tag)
        rows.append(m)
        print(name, {k: round(v, 3) for k, v in m.items() if k in ("density", "bad2", "absrel")}, flush=True)
        for flag, sub, arr in ((a.save_preds, "preds", full), (a.save_native, "preds_native", native)):
            if flag:
                d = os.path.join(ROOT, "results", sub, tag)
                os.makedirs(d, exist_ok=True)
                np.save(os.path.join(d, f"{name}.npy"), arr.astype(np.float32))
    os.makedirs(os.path.join(ROOT, "results/accuracy"), exist_ok=True)
    json.dump(rows, open(os.path.join(ROOT, "results/accuracy", f"{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
