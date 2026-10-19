#!/usr/bin/env bash
# Download the 23 Middlebury 2014 scenes that have public ground truth (full ~6 MP, "perfect" rectification).
# ~2.7 GB unpacked. Scharstein et al., GCPR 2014. https://vision.middlebury.edu/stereo/data/scenes2014/
set -euo pipefail
DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/data/middlebury2014}"
mkdir -p "$DEST" && cd "$DEST"
for s in Adirondack Backpack Bicycle1 Cable Classroom1 Couch Flowers Jadeplant Mask Motorcycle Piano Pipes \
         Playroom Playtable Recycle Shelves Shopvac Sticks Storage Sword1 Sword2 Umbrella Vintage; do
  [ -d "$s-perfect" ] && continue
  curl -sSfL -o "$s-perfect.zip" "https://vision.middlebury.edu/stereo/data/scenes2014/zip/$s-perfect.zip"
  unzip -q "$s-perfect.zip" && rm "$s-perfect.zip" && echo "downloaded $s"
done
