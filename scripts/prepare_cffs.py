"""Complete the stored settings of the commercial C-Fast-FoundationStereo checkpoint for this repo's
pinned Fast-FoundationStereo code.

The checkpoint (nvidia/c-fast-foundationstereo) was saved by newer training code and lacks the
`normalize` flag of the group-wise correlation volume that the pinned code reads; the research
checkpoints store normalize=True, which is also the code's default. With normalize=True our export
matches NVIDIA's own ONNX export (onnx/c_fast_foundationstereo_bp2_dynbatch_dynhw.onnx) to a median
of 0.016 px (0.08 %, FP16 TensorRT, Adirondack at 384x576), so the flag is True for this model too.
The original file is kept as model_original.pth.

  python scripts/prepare_cffs.py
"""
import os
import shutil
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "third_party/Fast-FoundationStereo"))
D = os.path.join(ROOT, "weights/ffs/c-ffs")


def main():
    pth = os.path.join(D, "model_best_bp2_serialize.pth")
    m = torch.load(pth, map_location="cpu", weights_only=False)
    if "normalize" in m.args:
        print("already complete:", pth)
        return
    shutil.copy(pth, os.path.join(D, "model_original.pth"))
    m.args["normalize"] = True
    torch.save(m, pth)
    print("added normalize=True to", pth)


if __name__ == "__main__":
    main()
