#!/usr/bin/env bash
# Build Fixstars libSGM (Apache-2.0) for the Orin GPU (sm_87) plus a small C shim for ctypes.
# Output: third_party/libSGM/build/libsgm_c.so
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/third_party/libSGM"
COMMIT=e4c669b  # v3.1.0 + fixes
[ -d "$SRC" ] || git clone https://github.com/fixstars/libSGM.git "$SRC"
git -C "$SRC" checkout -q "$COMMIT"
cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DLIBSGM_SHARED=OFF > /dev/null
cmake --build "$SRC/build" -j"$(nproc)" > /dev/null
lib=$(find "$SRC/build" -name "libsgm.a" | head -1)
g++ -O2 -shared -fPIC -o "$SRC/build/libsgm_c.so" "$(dirname "$0")/sgm_c.cpp" -I"$SRC/include" \
  -Wl,--whole-archive "$lib" -Wl,--no-whole-archive -L/usr/local/cuda/lib64 -lcudart
echo "built $SRC/build/libsgm_c.so"
