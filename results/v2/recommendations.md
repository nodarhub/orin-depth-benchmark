| Budget (one AGX Orin) | Best stereo | Best monocular | Best monocular, oracle scale |
|---|---|---|---|
| 30 fps | ESS (NVIDIA), 576×960: 0.70% (9 ms) | Depth Anything 3 Metric-L, 350×504: 12.34% (33 ms) | Depth Anything 3 Metric-L, 350×504: 5.97% (33 ms) |
| 10 fps | Fast-FoundationStereo 20-30-48, 384×576: 0.64% (68 ms) | Depth Anything 3 Metric-L, 350×504: 12.34% (33 ms) | UniDepth V2-B, 644×938: 3.90% (85 ms) |
| 1 fps | C-Fast-FoundationStereo, 768×1152: 0.33% (760 ms) | Depth Pro, 1536×1536: 11.98% (803 ms) | MoGe-2-L, 700×1008: 3.78% (212 ms) |
| no limit | C-Fast-FoundationStereo, 768×1152: 0.33% (760 ms) | Depth Pro, 1536×1536: 11.98% (803 ms) | MoGe-2-L, 700×1008: 3.78% (212 ms) |
| lowest GPU, open source | libSGM, ≈497×720: 1.18% (6 ms) | Depth Anything V2-S (native), 518×770: 29.88% (12 ms) | Depth Anything V2-S (native), 518×770: 8.93% (12 ms) |
