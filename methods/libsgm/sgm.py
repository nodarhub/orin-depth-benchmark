"""Fixstars libSGM (https://github.com/fixstars/libSGM) through the C shim built by build_libsgm.sh.
The aggregated cost volume is stored per path with 64-bit offsets, so it runs at full resolution."""
import ctypes
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
LIB = os.path.join(ROOT, "third_party/libSGM/build/libsgm_c.so")


class LibSGM:
    """Left disparity in pixels (float32, 0 = invalid). One instance per image size."""

    def __init__(self, width, height, disp_size=256, paths=8, subpixel=True, P1=10, P2=120):
        self.lib = ctypes.CDLL(LIB)
        self.lib.sgm_create.restype = ctypes.c_void_p
        self.lib.sgm_execute.argtypes = [ctypes.c_void_p] * 4
        self.lib.sgm_invalid.argtypes = [ctypes.c_void_p]
        self.lib.sgm_destroy.argtypes = [ctypes.c_void_p]
        # aggregated costs are 8-bit: census cost (<=31) + P2 must stay below 256
        assert P2 <= 224, "P2 > 224 overflows libSGM's 8-bit aggregated cost"
        self.h = self.lib.sgm_create(width, height, disp_size, int(paths == 8), int(subpixel), P1, P2)
        self.invalid = self.lib.sgm_invalid(self.h)
        self.scale = 16.0 if subpixel else 1.0
        self.out = np.empty((height, width), np.uint16)

    def __call__(self, left_gray, right_gray):
        l, r = (np.ascontiguousarray(x, dtype=np.uint8) for x in (left_gray, right_gray))
        self.lib.sgm_execute(self.h, l.ctypes.data, r.ctypes.data, self.out.ctypes.data)
        raw = self.out.astype(np.int32)
        disp = raw.astype(np.float32) / self.scale
        disp[(raw == (self.invalid & 0xFFFF)) | (raw == self.invalid)] = 0
        return disp

    def __del__(self):
        if getattr(self, "h", None):
            self.lib.sgm_destroy(self.h)
