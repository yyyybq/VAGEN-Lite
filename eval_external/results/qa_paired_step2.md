**逐题配对比较**（差 = B − A，单位是点；95% 区间来自题内配对 bootstrap，2000 次、种子 20261021；只有 A 对 / 只有 B 对和 McNemar p 只对每题非对即错的 benchmark 给出）

| A | B | benchmark | 题数 | A 的分 | B 的分 | 差 [95% 区间] | 只有 A 对 / 只有 B 对 | p |
|---|---|---|---|---|---|---|---|---|
| base-7B | ours-SFT | SAT-Real | 150 | 58.0 | 68.0 | +10.0 [+0.7, +19.3] | 17 / 32 | 0.04 |
| base-7B | ours-SFT | SAT-Syn | 4001 | 59.1 | 58.8 | -0.3 [-2.4, +1.8] | 871 / 859 | 0.79 |
| base-7B | ours-SFT | ViewSpatial | 5712 | 37.7 | 39.3 | +1.6 [+0.6, +2.7] | 387 / 480 | 0.0018 |
| base-7B | ours-SFT | MindCube-Tiny | 1040 | 34.2 | 30.3 | -3.9 [-7.4, -0.4] | 209 / 168 | 0.04 |
| base-7B | ours-SFT | MindCube | 21154 | 29.4 | 31.2 | +1.8 [+1.1, +2.6] | 3291 / 3672 | <0.0001 |
| base-7B | ours-SFT | VSI-Bench | 5130 | 37.6 | 30.7 | -6.9 [-8.4, -5.2] | — | — |
| base-7B | ours-SFT | VSTI-Bench | 5736 | 41.5 | 30.1 | -11.4 [-12.6, -10.2] | — | — |
| base-7B | ours-SFT | MMSI | 1000 | 28.0 | 25.9 | -2.1 [-4.9, +0.6] | 114 / 93 | 0.16 |
| base-7B | ours-SFT | SPBench-MV | 319 | 42.7 | 38.2 | -4.5 [-8.8, -0.4] | — | — |
| base-7B | ours-RL-50 | SAT-Real | 150 | 58.0 | 67.3 | +9.3 [+0.7, +18.0] | 17 / 31 | 0.06 |
| base-7B | ours-RL-50 | SAT-Syn | 4001 | 59.1 | 58.8 | -0.2 [-2.3, +1.8] | 860 / 850 | 0.83 |
| base-7B | ours-RL-50 | ViewSpatial | 5712 | 37.7 | 39.1 | +1.4 [+0.4, +2.5] | 388 / 471 | 0.0051 |
| base-7B | ours-RL-50 | MindCube-Tiny | 1040 | 34.2 | 31.1 | -3.1 [-6.4, +0.3] | 197 / 165 | 0.10 |
| base-7B | ours-RL-50 | MindCube | 21154 | 29.4 | 32.8 | +3.4 [+2.6, +4.2] | 3141 / 3861 | <0.0001 |
| base-7B | ours-RL-50 | VSI-Bench | 5130 | 37.6 | 30.9 | -6.7 [-8.3, -5.1] | — | — |
| base-7B | ours-RL-50 | VSTI-Bench | 5736 | 41.5 | 30.3 | -11.2 [-12.4, -10.0] | — | — |
| base-7B | ours-RL-50 | MMSI | 1000 | 28.0 | 26.3 | -1.7 [-4.4, +1.0] | 111 / 94 | 0.26 |
| base-7B | ours-RL-50 | SPBench-MV | 319 | 42.7 | 38.4 | -4.3 [-8.6, -0.2] | — | — |
| base-7B | ours-RL-150 | SAT-Real | 150 | 58.0 | 64.7 | +6.7 [-2.0, +15.3] | 16 / 26 | 0.16 |
| base-7B | ours-RL-150 | SAT-Syn | 4001 | 59.1 | 58.4 | -0.7 [-2.7, +1.4] | 887 / 860 | 0.53 |
| base-7B | ours-RL-150 | ViewSpatial | 5712 | 37.7 | 39.5 | +1.8 [+0.8, +2.9] | 386 / 491 | 0.0004 |
| base-7B | ours-RL-150 | MindCube-Tiny | 1040 | 34.2 | 31.2 | -3.0 [-6.3, +0.5] | 194 / 163 | 0.11 |
| base-7B | ours-RL-150 | MindCube | 21154 | 29.4 | 32.6 | +3.2 [+2.5, +4.0] | 3130 / 3814 | <0.0001 |
| base-7B | ours-RL-150 | VSI-Bench | 5130 | 37.6 | 31.1 | -6.5 [-8.1, -4.9] | — | — |
| base-7B | ours-RL-150 | VSTI-Bench | 5736 | 41.5 | 30.0 | -11.5 [-12.7, -10.3] | — | — |
| base-7B | ours-RL-150 | MMSI | 1000 | 28.0 | 25.8 | -2.2 [-4.9, +0.4] | 113 / 91 | 0.14 |
| base-7B | ours-RL-150 | SPBench-MV | 319 | 42.7 | 38.5 | -4.1 [-8.6, +0.1] | — | — |
| base-7B | ours-SFT-r11 | SAT-Real | 150 | 58.0 | 70.7 | +12.7 [+4.7, +20.7] | 11 / 30 | 0.0043 |
| base-7B | ours-SFT-r11 | SAT-Syn | 4001 | 59.1 | 58.3 | -0.8 [-2.8, +1.3] | 857 / 826 | 0.46 |
| base-7B | ours-SFT-r11 | ViewSpatial | 5712 | 37.7 | 39.4 | +1.7 [+0.6, +2.9] | 512 / 610 | 0.0038 |
| base-7B | ours-SFT-r11 | MindCube-Tiny | 1040 | 34.2 | 12.4 | -21.8 [-25.3, -18.6] | 300 / 73 | <0.0001 |
| base-7B | ours-SFT-r11 | MindCube | 21154 | 29.4 | 13.8 | -15.5 [-16.3, -14.7] | 5016 / 1729 | <0.0001 |
| base-7B | ours-SFT-r11 | VSI-Bench | 5130 | 37.6 | 28.0 | -9.5 [-11.2, -7.9] | — | — |
| base-7B | ours-SFT-r11 | VSTI-Bench | 5736 | 41.5 | 29.2 | -12.3 [-13.6, -11.0] | — | — |
| base-7B | ours-SFT-r11 | MMSI | 1000 | 28.0 | 25.1 | -2.9 [-5.9, +0.2] | 140 / 111 | 0.08 |
| base-7B | ours-SFT-r11 | SPBench-MV | 319 | 42.7 | 34.7 | -8.0 [-13.9, -2.6] | — | — |
| base-7B | ours-RL-r11 | SAT-Real | 150 | 58.0 | 74.7 | +16.7 [+8.0, +24.7] | 11 / 36 | 0.0003 |
| base-7B | ours-RL-r11 | SAT-Syn | 4001 | 59.1 | 58.8 | -0.3 [-2.3, +1.8] | 881 / 870 | 0.81 |
| base-7B | ours-RL-r11 | ViewSpatial | 5712 | 37.7 | 39.2 | +1.6 [+0.5, +2.8] | 507 / 597 | 0.0074 |
| base-7B | ours-RL-r11 | MindCube-Tiny | 1040 | 34.2 | 13.6 | -20.7 [-23.9, -17.3] | 290 / 75 | <0.0001 |
| base-7B | ours-RL-r11 | MindCube | 21154 | 29.4 | 15.2 | -14.2 [-14.9, -13.4] | 4882 / 1874 | <0.0001 |
| base-7B | ours-RL-r11 | VSI-Bench | 5130 | 37.6 | 28.2 | -9.4 [-10.9, -7.6] | — | — |
| base-7B | ours-RL-r11 | VSTI-Bench | 5736 | 41.5 | 29.4 | -12.1 [-13.4, -10.8] | — | — |
| base-7B | ours-RL-r11 | MMSI | 1000 | 28.0 | 25.5 | -2.5 [-5.5, +0.5] | 137 / 112 | 0.13 |
| base-7B | ours-RL-r11 | SPBench-MV | 319 | 42.7 | 32.5 | -10.2 [-15.9, -4.8] | — | — |
| ours-SFT | ours-RL-50 | SAT-Real | 150 | 68.0 | 67.3 | -0.7 [-5.3, +3.3] | 6 / 5 | 1.00 |
| ours-SFT | ours-RL-50 | SAT-Syn | 4001 | 58.8 | 58.8 | +0.1 [-0.5, +0.6] | 57 / 59 | 0.93 |
| ours-SFT | ours-RL-50 | ViewSpatial | 5712 | 39.3 | 39.1 | -0.2 [-0.5, +0.1] | 48 / 38 | 0.33 |
| ours-SFT | ours-RL-50 | MindCube-Tiny | 1040 | 30.3 | 31.1 | +0.9 [-0.4, +2.1] | 18 / 27 | 0.23 |
| ours-SFT | ours-RL-50 | MindCube | 21154 | 31.2 | 32.8 | +1.6 [+1.3, +1.9] | 263 / 602 | <0.0001 |
| ours-SFT | ours-RL-50 | VSI-Bench | 5130 | 30.7 | 30.9 | +0.2 [-0.3, +0.6] | — | — |
| ours-SFT | ours-RL-50 | VSTI-Bench | 5736 | 30.1 | 30.3 | +0.2 [-0.1, +0.6] | — | — |
| ours-SFT | ours-RL-50 | MMSI | 1000 | 25.9 | 26.3 | +0.4 [-0.4, +1.3] | 7 / 11 | 0.48 |
| ours-SFT | ours-RL-50 | SPBench-MV | 319 | 38.2 | 38.4 | +0.2 [-1.2, +1.7] | — | — |
| ours-SFT | ours-RL-150 | SAT-Real | 150 | 68.0 | 64.7 | -3.3 [-7.3, +0.7] | 7 / 2 | 0.18 |
| ours-SFT | ours-RL-150 | SAT-Syn | 4001 | 58.8 | 58.4 | -0.4 [-1.0, +0.2] | 86 / 71 | 0.26 |
| ours-SFT | ours-RL-150 | ViewSpatial | 5712 | 39.3 | 39.5 | +0.2 [-0.1, +0.6] | 48 / 60 | 0.29 |
| ours-SFT | ours-RL-150 | MindCube-Tiny | 1040 | 30.3 | 31.2 | +1.0 [-0.5, +2.5] | 26 / 36 | 0.25 |
| ours-SFT | ours-RL-150 | MindCube | 21154 | 31.2 | 32.6 | +1.4 [+1.1, +1.7] | 364 / 667 | <0.0001 |
| ours-SFT | ours-RL-150 | VSI-Bench | 5130 | 30.7 | 31.1 | +0.3 [-0.2, +0.9] | — | — |
| ours-SFT | ours-RL-150 | VSTI-Bench | 5736 | 30.1 | 30.0 | -0.1 [-0.5, +0.3] | — | — |
| ours-SFT | ours-RL-150 | MMSI | 1000 | 25.9 | 25.8 | -0.1 [-1.2, +1.0] | 17 / 16 | 1.00 |
| ours-SFT | ours-RL-150 | SPBench-MV | 319 | 38.2 | 38.5 | +0.4 [-0.7, +1.7] | — | — |
| ours-SFT-r11 | ours-RL-r11 | SAT-Real | 150 | 70.7 | 74.7 | +4.0 [-0.7, +9.3] | 5 / 11 | 0.21 |
| ours-SFT-r11 | ours-RL-r11 | SAT-Syn | 4001 | 58.3 | 58.8 | +0.5 [-0.5, +1.4] | 181 / 201 | 0.33 |
| ours-SFT-r11 | ours-RL-r11 | ViewSpatial | 5712 | 39.4 | 39.2 | -0.1 [-0.7, +0.4] | 143 / 135 | 0.67 |
| ours-SFT-r11 | ours-RL-r11 | MindCube-Tiny | 1040 | 12.4 | 13.6 | +1.1 [+0.1, +2.2] | 11 / 23 | 0.06 |
| ours-SFT-r11 | ours-RL-r11 | MindCube | 21154 | 13.8 | 15.2 | +1.3 [+1.0, +1.6] | 335 / 614 | <0.0001 |
| ours-SFT-r11 | ours-RL-r11 | VSI-Bench | 5130 | 28.0 | 28.2 | +0.2 [-0.6, +0.9] | — | — |
| ours-SFT-r11 | ours-RL-r11 | VSTI-Bench | 5736 | 29.2 | 29.4 | +0.1 [-0.4, +0.7] | — | — |
| ours-SFT-r11 | ours-RL-r11 | MMSI | 1000 | 25.1 | 25.5 | +0.4 [-1.4, +2.2] | 41 / 45 | 0.75 |
| ours-SFT-r11 | ours-RL-r11 | SPBench-MV | 319 | 34.7 | 32.5 | -2.2 [-7.0, +2.8] | — | — |
| ours-SFT | ours-SFT-r11 | SAT-Real | 150 | 68.0 | 70.7 | +2.7 [-4.7, +10.7] | 15 / 19 | 0.61 |
| ours-SFT | ours-SFT-r11 | SAT-Syn | 4001 | 58.8 | 58.3 | -0.5 [-1.9, +0.9] | 430 / 411 | 0.53 |
| ours-SFT | ours-SFT-r11 | ViewSpatial | 5712 | 39.3 | 39.4 | +0.1 [-0.8, +1.0] | 345 / 350 | 0.88 |
| ours-SFT | ours-SFT-r11 | MindCube-Tiny | 1040 | 30.3 | 12.4 | -17.9 [-20.8, -15.1] | 220 / 34 | <0.0001 |
| ours-SFT | ours-SFT-r11 | MindCube | 21154 | 31.2 | 13.8 | -17.3 [-18.0, -16.7] | 4382 / 714 | <0.0001 |
| ours-SFT | ours-SFT-r11 | VSI-Bench | 5130 | 30.7 | 28.0 | -2.7 [-3.9, -1.5] | — | — |
| ours-SFT | ours-SFT-r11 | VSTI-Bench | 5736 | 30.1 | 29.2 | -0.8 [-1.8, +0.1] | — | — |
| ours-SFT | ours-SFT-r11 | MMSI | 1000 | 25.9 | 25.1 | -0.8 [-3.2, +1.6] | 74 / 66 | 0.55 |
| ours-SFT | ours-SFT-r11 | SPBench-MV | 319 | 38.2 | 34.7 | -3.5 [-8.3, +0.8] | — | — |
| base-7B | SI-7B | SAT-Real | 150 | 58.0 | 68.7 | +10.7 [+0.7, +20.7] | 23 / 39 | 0.06 |
| base-7B | SI-7B | SAT-Syn | 4001 | 59.1 | 61.1 | +2.0 [-0.5, +4.4] | 1187 / 1267 | 0.11 |
| base-7B | SI-7B | ViewSpatial | 5712 | 37.7 | 43.2 | +5.6 [+4.1, +7.0] | 716 / 1034 | <0.0001 |
| base-7B | SI-7B | MindCube-Tiny | 1040 | 34.2 | 73.5 | +39.2 [+35.0, +43.1] | 106 / 514 | <0.0001 |
| base-7B | SI-7B | MindCube | 21154 | 29.4 | 82.8 | +53.4 [+52.6, +54.2] | 1291 / 12588 | <0.0001 |
| base-7B | SI-7B | VSI-Bench | 5130 | 37.6 | 47.5 | +10.0 [+8.3, +11.7] | — | — |
| base-7B | SI-7B | VSTI-Bench | 5736 | 41.5 | 50.8 | +9.3 [+8.0, +10.7] | — | — |
| base-7B | SI-7B | VSTI-Bench-prefix | 6042 | 40.1 | 49.1 | +9.0 [+7.7, +10.3] | — | — |
| base-7B | SI-7B | MMSI | 1000 | 28.0 | 31.2 | +3.2 [-0.2, +7.0] | 149 / 181 | 0.09 |
| base-7B | SI-7B | SPBench-MV | 319 | 42.7 | 67.2 | +24.5 [+19.0, +30.3] | — | — |
| base-7B | SpaceR-7B | SAT-Real | 150 | 58.0 | 55.3 | -2.7 [-10.7, +5.3] | 21 / 17 | 0.63 |
| base-7B | SpaceR-7B | SAT-Syn | 4001 | 59.1 | 61.4 | +2.3 [+1.3, +3.3] | 178 / 271 | <0.0001 |
| base-7B | SpaceR-7B | ViewSpatial | 5712 | 37.7 | 38.0 | +0.4 [-0.3, +1.2] | 244 / 266 | 0.35 |
| base-7B | SpaceR-7B | MindCube-Tiny | 1040 | 34.2 | 32.6 | -1.6 [-4.0, +0.7] | 89 / 72 | 0.21 |
| base-7B | SpaceR-7B | MindCube | 21154 | 29.4 | 26.6 | -2.8 [-3.3, -2.2] | 1834 / 1251 | <0.0001 |
| base-7B | SpaceR-7B | VSI-Bench | 5130 | 37.6 | 44.6 | +7.1 [+5.9, +8.3] | — | — |
| base-7B | SpaceR-7B | VSTI-Bench | 5736 | 41.5 | 45.4 | +3.9 [+2.7, +5.0] | — | — |
| base-7B | SpaceR-7B | MMSI | 1000 | 28.0 | 23.1 | -4.9 [-7.0, -2.8] | 84 / 35 | <0.0001 |
| base-7B | SpaceR-7B | SPBench-MV | 319 | 42.7 | 56.7 | +14.0 [+8.4, +19.4] | — | — |
| base-7B | VideoR1-7B | SAT-Real | 150 | 58.0 | 48.0 | -10.0 [-18.0, -2.0] | 26 / 11 | 0.02 |
| base-7B | VideoR1-7B | SAT-Syn | 4001 | 59.1 | 55.3 | -3.8 [-4.8, -2.7] | 325 / 175 | <0.0001 |
| base-7B | VideoR1-7B | ViewSpatial | 5712 | 37.7 | 37.9 | +0.2 [-0.7, +1.2] | 370 / 382 | 0.69 |
| base-7B | VideoR1-7B | MindCube-Tiny | 1040 | 34.2 | 34.6 | +0.4 [-2.2, +3.1] | 101 / 105 | 0.83 |
| base-7B | VideoR1-7B | MindCube | 21154 | 29.4 | 31.9 | +2.5 [+2.0, +3.1] | 1702 / 2236 | <0.0001 |
| base-7B | VideoR1-7B | VSI-Bench | 5130 | 37.6 | 32.6 | -5.0 [-6.4, -3.7] | — | — |
| base-7B | VideoR1-7B | VSTI-Bench | 5736 | 41.5 | 44.6 | +3.1 [+1.9, +4.3] | — | — |
| base-7B | VideoR1-7B | MMSI | 1000 | 28.0 | 24.2 | -3.8 [-6.8, -0.8] | 137 / 99 | 0.02 |
| base-7B | VideoR1-7B | SPBench-MV | 319 | 42.7 | 40.7 | -2.0 [-7.4, +3.2] | — | — |
| base-7B | ViLASR-7B | SAT-Real | 150 | 58.0 | 56.0 | -2.0 [-10.7, +6.0] | 21 / 18 | 0.75 |
| base-7B | ViLASR-7B | SAT-Syn | 4001 | 59.1 | 64.4 | +5.3 [+3.9, +6.8] | 318 / 532 | <0.0001 |
| base-7B | ViLASR-7B | ViewSpatial | 5712 | 37.7 | 35.6 | -2.0 [-3.1, -1.1] | 503 / 386 | <0.0001 |
| base-7B | ViLASR-7B | MindCube-Tiny | 1040 | 34.2 | 35.1 | +0.9 [-2.2, +4.1] | 130 / 139 | 0.63 |
| base-7B | ViLASR-7B | MindCube | 21154 | 29.4 | 30.6 | +1.2 [+0.6, +1.9] | 2320 / 2576 | 0.0003 |
| base-7B | ViLASR-7B | VSI-Bench | 5130 | 37.6 | 43.7 | +6.2 [+4.6, +7.7] | — | — |
| base-7B | ViLASR-7B | VSTI-Bench | 5736 | 41.5 | 47.1 | +5.6 [+4.4, +6.8] | — | — |
| base-7B | ViLASR-7B | MMSI | 1000 | 28.0 | 30.7 | +2.7 [-0.4, +5.6] | 97 / 124 | 0.08 |
| base-7B | ViLASR-7B | SPBench-MV | 319 | 42.7 | 58.2 | +15.5 [+9.9, +21.0] | — | — |
| base-7B | VST-7B-RL | SAT-Real | 150 | 58.0 | 55.3 | -2.7 [-12.7, +8.0] | 35 / 31 | 0.71 |
| base-7B | VST-7B-RL | SAT-Syn | 4001 | 59.1 | 64.3 | +5.2 [+3.5, +6.8] | 484 / 694 | <0.0001 |
| base-7B | VST-7B-RL | ViewSpatial | 5712 | 37.7 | 49.9 | +12.2 [+10.9, +13.6] | 532 / 1232 | <0.0001 |
| base-7B | VST-7B-RL | MindCube-Tiny | 1040 | 34.2 | 40.9 | +6.6 [+3.3, +9.9] | 131 / 200 | 0.0002 |
| base-7B | VST-7B-RL | MindCube | 21154 | 29.4 | 37.3 | +7.9 [+7.2, +8.6] | 2230 / 3899 | <0.0001 |
| base-7B | VST-7B-RL | VSI-Bench | 5130 | 37.6 | 56.0 | +18.4 [+16.6, +20.3] | — | — |
| base-7B | VST-7B-RL | VSTI-Bench | 5736 | 41.5 | 48.6 | +7.1 [+5.7, +8.6] | — | — |
| base-7B | VST-7B-RL | MMSI | 1000 | 28.0 | 33.1 | +5.1 [+1.6, +8.7] | 153 / 204 | 0.0080 |
| base-7B | VST-7B-RL | SPBench-MV | 319 | 42.7 | 66.2 | +23.5 [+17.5, +29.4] | — | — |
| base-7B | SenseNovaSI-7B | SAT-Real | 150 | 58.0 | 61.3 | +3.3 [-5.3, +12.0] | 19 / 24 | 0.54 |
| base-7B | SenseNovaSI-7B | SAT-Syn | 4001 | 59.1 | 58.8 | -0.3 [-2.1, +1.4] | 681 / 668 | 0.74 |
| base-7B | SenseNovaSI-7B | ViewSpatial | 5712 | 37.7 | 45.1 | +7.4 [+6.2, +8.7] | 450 / 875 | <0.0001 |
| base-7B | SenseNovaSI-7B | MindCube-Tiny | 1040 | 34.2 | 53.8 | +19.5 [+16.1, +23.4] | 106 / 309 | <0.0001 |
| base-7B | SenseNovaSI-7B | MindCube | 21154 | 29.4 | 56.6 | +27.2 [+26.5, +28.1] | 1787 / 7547 | <0.0001 |
| base-7B | SenseNovaSI-7B | VSI-Bench | 5130 | 37.6 | 49.6 | +12.0 [+10.3, +13.6] | — | — |
| base-7B | SenseNovaSI-7B | VSTI-Bench | 5736 | 41.5 | 49.8 | +8.3 [+7.2, +9.4] | — | — |
| base-7B | SenseNovaSI-7B | MMSI | 1000 | 28.0 | 32.6 | +4.6 [+1.4, +7.8] | 121 / 167 | 0.0079 |
| base-7B | SenseNovaSI-7B | SPBench-MV | 319 | 42.7 | 60.0 | +17.3 [+12.1, +22.4] | — | — |
| base-3B | SI-3B | SAT-Real | 150 | 50.0 | 64.0 | +14.0 [+4.0, +24.0] | 19 / 40 | 0.0086 |
| base-3B | SI-3B | SAT-Syn | 4001 | 54.6 | 50.5 | -4.2 [-6.2, -2.1] | 953 / 786 | <0.0001 |
| base-3B | SI-3B | ViewSpatial | 5712 | 35.7 | 44.7 | +9.0 [+7.5, +10.4] | 648 / 1160 | <0.0001 |
| base-3B | SI-3B | MindCube-Tiny | 1040 | 41.5 | 60.2 | +18.6 [+14.5, +22.7] | 158 / 352 | <0.0001 |
| base-3B | SI-3B | MindCube | 21154 | 39.8 | 66.6 | +26.8 [+26.0, +27.7] | 2673 / 8347 | <0.0001 |
| base-3B | SI-3B | VSI-Bench | 5130 | 26.8 | 49.3 | +22.5 [+20.8, +24.3] | — | — |
| base-3B | SI-3B | VSTI-Bench | 5736 | 40.2 | 50.4 | +10.2 [+8.8, +11.5] | — | — |
| base-3B | SI-3B | VSTI-Bench-prefix | 6042 | 39.8 | 48.6 | +8.8 [+7.5, +10.1] | — | — |
| base-3B | SI-3B | MMSI | 1000 | 28.2 | 29.0 | +0.8 [-2.8, +4.1] | 159 / 167 | 0.70 |
| base-3B | SI-3B | SPBench-MV | 319 | 40.6 | 61.2 | +20.7 [+13.3, +27.5] | — | — |
| SpaceR-7B | SpaceR-7B-ownpx | VSI-Bench | 5130 | 44.6 | 45.2 | +0.5 [-0.2, +1.3] | — | — |
| VideoR1-7B | VideoR1-7B-ownpx | VSI-Bench | 5130 | 32.6 | 33.0 | +0.3 [-0.5, +1.1] | — | — |
| SenseNovaSI-7B | SenseNovaSI-7B-ownpx | VSI-Bench | 5130 | 49.6 | 51.3 | +1.8 [+0.7, +2.9] | — | — |

**VSI-Bench 分题型**（VSTI 按 VLM-3R 的五类；差 = B − A，点）

| A | B | 题型 | 题数 | A | B | 差 [95% 区间] |
|---|---|---|---|---|---|---|
| base-7B | ours-SFT | obj_appearance_order | 618 | 34.6 | 30.3 | -4.4 [-7.8, -1.0] |
| base-7B | ours-SFT | object_abs_distance | 834 | 22.6 | 5.4 | -17.2 [-19.8, -14.7] |
| base-7B | ours-SFT | object_counting | 565 | 41.4 | 46.0 | +4.6 [+1.5, +8.0] |
| base-7B | ours-SFT | object_rel_direction_easy | 217 | 46.5 | 51.1 | +4.6 [-1.4, +10.6] |
| base-7B | ours-SFT | object_rel_direction_hard | 373 | 30.0 | 23.9 | -6.2 [-11.3, -1.3] |
| base-7B | ours-SFT | object_rel_direction_medium | 378 | 43.9 | 43.9 | +0.0 [-5.0, +5.0] |
| base-7B | ours-SFT | object_rel_distance | 710 | 37.3 | 35.9 | -1.4 [-4.9, +2.2] |
| base-7B | ours-SFT | object_size_estimation | 953 | 49.0 | 29.7 | -19.3 [-21.9, -16.6] |
| base-7B | ours-SFT | room_size_estimation | 288 | 46.2 | 32.0 | -14.2 [-18.4, -9.9] |
| base-7B | ours-SFT | route_planning | 194 | 29.4 | 26.8 | -2.6 [-11.9, +6.7] |
| base-7B | ours-RL-50 | obj_appearance_order | 618 | 34.6 | 31.1 | -3.6 [-7.0, +0.0] |
| base-7B | ours-RL-50 | object_abs_distance | 834 | 22.6 | 5.3 | -17.2 [-19.8, -14.7] |
| base-7B | ours-RL-50 | object_counting | 565 | 41.4 | 46.2 | +4.8 [+1.9, +8.0] |
| base-7B | ours-RL-50 | object_rel_direction_easy | 217 | 46.5 | 50.7 | +4.2 [-1.8, +10.1] |
| base-7B | ours-RL-50 | object_rel_direction_hard | 373 | 30.0 | 25.5 | -4.6 [-9.9, +0.3] |
| base-7B | ours-RL-50 | object_rel_direction_medium | 378 | 43.9 | 43.1 | -0.8 [-5.6, +4.5] |
| base-7B | ours-RL-50 | object_rel_distance | 710 | 37.3 | 35.5 | -1.8 [-5.2, +1.6] |
| base-7B | ours-RL-50 | object_size_estimation | 953 | 49.0 | 29.7 | -19.3 [-21.7, -16.6] |
| base-7B | ours-RL-50 | room_size_estimation | 288 | 46.2 | 33.2 | -13.0 [-17.2, -8.7] |
| base-7B | ours-RL-50 | route_planning | 194 | 29.4 | 26.3 | -3.1 [-12.4, +6.2] |
| base-7B | ours-RL-150 | obj_appearance_order | 618 | 34.6 | 30.7 | -3.9 [-7.3, -0.2] |
| base-7B | ours-RL-150 | object_abs_distance | 834 | 22.6 | 5.5 | -17.1 [-19.8, -14.5] |
| base-7B | ours-RL-150 | object_counting | 565 | 41.4 | 45.2 | +3.8 [+1.0, +7.0] |
| base-7B | ours-RL-150 | object_rel_direction_easy | 217 | 46.5 | 51.1 | +4.6 [-1.4, +10.6] |
| base-7B | ours-RL-150 | object_rel_direction_hard | 373 | 30.0 | 25.2 | -4.8 [-9.9, +0.0] |
| base-7B | ours-RL-150 | object_rel_direction_medium | 378 | 43.9 | 43.4 | -0.5 [-5.3, +4.5] |
| base-7B | ours-RL-150 | object_rel_distance | 710 | 37.3 | 34.4 | -3.0 [-6.3, +0.6] |
| base-7B | ours-RL-150 | object_size_estimation | 953 | 49.0 | 30.5 | -18.5 [-21.0, -15.8] |
| base-7B | ours-RL-150 | room_size_estimation | 288 | 46.2 | 36.0 | -10.2 [-14.3, -6.0] |
| base-7B | ours-RL-150 | route_planning | 194 | 29.4 | 26.3 | -3.1 [-12.4, +6.2] |
| base-7B | ours-SFT-r11 | obj_appearance_order | 618 | 34.6 | 30.4 | -4.2 [-8.4, +0.2] |
| base-7B | ours-SFT-r11 | object_abs_distance | 834 | 22.6 | 6.7 | -15.9 [-18.5, -13.4] |
| base-7B | ours-SFT-r11 | object_counting | 565 | 41.4 | 43.5 | +2.2 [-0.7, +5.1] |
| base-7B | ours-SFT-r11 | object_rel_direction_easy | 217 | 46.5 | 49.3 | +2.8 [-4.2, +9.2] |
| base-7B | ours-SFT-r11 | object_rel_direction_hard | 373 | 30.0 | 30.6 | +0.5 [-3.8, +4.8] |
| base-7B | ours-SFT-r11 | object_rel_direction_medium | 378 | 43.9 | 43.1 | -0.8 [-6.3, +5.3] |
| base-7B | ours-SFT-r11 | object_rel_distance | 710 | 37.3 | 30.6 | -6.8 [-10.3, -3.1] |
| base-7B | ours-SFT-r11 | object_size_estimation | 953 | 49.0 | 25.2 | -23.8 [-26.3, -21.1] |
| base-7B | ours-SFT-r11 | room_size_estimation | 288 | 46.2 | 19.1 | -27.1 [-31.6, -22.8] |
| base-7B | ours-SFT-r11 | route_planning | 194 | 29.4 | 27.8 | -1.6 [-10.8, +7.7] |
| base-7B | ours-RL-r11 | obj_appearance_order | 618 | 34.6 | 29.8 | -4.8 [-8.9, -0.8] |
| base-7B | ours-RL-r11 | object_abs_distance | 834 | 22.6 | 7.0 | -15.6 [-18.0, -13.1] |
| base-7B | ours-RL-r11 | object_counting | 565 | 41.4 | 46.8 | +5.5 [+2.3, +8.7] |
| base-7B | ours-RL-r11 | object_rel_direction_easy | 217 | 46.5 | 45.6 | -0.9 [-7.8, +6.0] |
| base-7B | ours-RL-r11 | object_rel_direction_hard | 373 | 30.0 | 26.8 | -3.2 [-7.8, +1.3] |
| base-7B | ours-RL-r11 | object_rel_direction_medium | 378 | 43.9 | 42.9 | -1.1 [-6.6, +5.3] |
| base-7B | ours-RL-r11 | object_rel_distance | 710 | 37.3 | 32.5 | -4.8 [-8.7, -0.8] |
| base-7B | ours-RL-r11 | object_size_estimation | 953 | 49.0 | 26.2 | -22.8 [-25.4, -19.9] |
| base-7B | ours-RL-r11 | room_size_estimation | 288 | 46.2 | 18.7 | -27.5 [-32.1, -22.9] |
| base-7B | ours-RL-r11 | route_planning | 194 | 29.4 | 26.3 | -3.1 [-12.4, +5.7] |
| ours-SFT | ours-RL-50 | obj_appearance_order | 618 | 30.3 | 31.1 | +0.8 [-0.7, +2.3] |
| ours-SFT | ours-RL-50 | object_abs_distance | 834 | 5.4 | 5.3 | -0.1 [-0.7, +0.6] |
| ours-SFT | ours-RL-50 | object_counting | 565 | 46.0 | 46.2 | +0.2 [-0.7, +1.2] |
| ours-SFT | ours-RL-50 | object_rel_direction_easy | 217 | 51.1 | 50.7 | -0.5 [-1.8, +0.9] |
| ours-SFT | ours-RL-50 | object_rel_direction_hard | 373 | 23.9 | 25.5 | +1.6 [+0.0, +3.2] |
| ours-SFT | ours-RL-50 | object_rel_direction_medium | 378 | 43.9 | 43.1 | -0.8 [-2.4, +0.8] |
| ours-SFT | ours-RL-50 | object_rel_distance | 710 | 35.9 | 35.5 | -0.4 [-1.6, +0.7] |
| ours-SFT | ours-RL-50 | object_size_estimation | 953 | 29.7 | 29.7 | +0.0 [-1.0, +1.0] |
| ours-SFT | ours-RL-50 | room_size_estimation | 288 | 32.0 | 33.2 | +1.2 [-0.3, +2.9] |
| ours-SFT | ours-RL-50 | route_planning | 194 | 26.8 | 26.3 | -0.5 [-2.6, +1.6] |
| ours-SFT | ours-RL-150 | obj_appearance_order | 618 | 30.3 | 30.7 | +0.5 [-1.0, +1.9] |
| ours-SFT | ours-RL-150 | object_abs_distance | 834 | 5.4 | 5.5 | +0.1 [-0.6, +0.7] |
| ours-SFT | ours-RL-150 | object_counting | 565 | 46.0 | 45.2 | -0.8 [-1.9, +0.4] |
| ours-SFT | ours-RL-150 | object_rel_direction_easy | 217 | 51.1 | 51.1 | +0.0 [-1.4, +1.4] |
| ours-SFT | ours-RL-150 | object_rel_direction_hard | 373 | 23.9 | 25.2 | +1.3 [+0.0, +3.0] |
| ours-SFT | ours-RL-150 | object_rel_direction_medium | 378 | 43.9 | 43.4 | -0.5 [-2.1, +0.8] |
| ours-SFT | ours-RL-150 | object_rel_distance | 710 | 35.9 | 34.4 | -1.6 [-2.8, -0.3] |
| ours-SFT | ours-RL-150 | object_size_estimation | 953 | 29.7 | 30.5 | +0.8 [-0.1, +1.8] |
| ours-SFT | ours-RL-150 | room_size_estimation | 288 | 32.0 | 36.0 | +4.0 [+1.7, +6.4] |
| ours-SFT | ours-RL-150 | route_planning | 194 | 26.8 | 26.3 | -0.5 [-2.6, +1.6] |
| ours-SFT-r11 | ours-RL-r11 | obj_appearance_order | 618 | 30.4 | 29.8 | -0.7 [-2.8, +1.5] |
| ours-SFT-r11 | ours-RL-r11 | object_abs_distance | 834 | 6.7 | 7.0 | +0.4 [-0.5, +1.3] |
| ours-SFT-r11 | ours-RL-r11 | object_counting | 565 | 43.5 | 46.8 | +3.3 [+1.5, +5.1] |
| ours-SFT-r11 | ours-RL-r11 | object_rel_direction_easy | 217 | 49.3 | 45.6 | -3.7 [-7.4, +0.5] |
| ours-SFT-r11 | ours-RL-r11 | object_rel_direction_hard | 373 | 30.6 | 26.8 | -3.8 [-6.7, -1.1] |
| ours-SFT-r11 | ours-RL-r11 | object_rel_direction_medium | 378 | 43.1 | 42.9 | -0.3 [-3.2, +2.6] |
| ours-SFT-r11 | ours-RL-r11 | object_rel_distance | 710 | 30.6 | 32.5 | +2.0 [-0.1, +4.1] |
| ours-SFT-r11 | ours-RL-r11 | object_size_estimation | 953 | 25.2 | 26.2 | +1.0 [-0.4, +2.5] |
| ours-SFT-r11 | ours-RL-r11 | room_size_estimation | 288 | 19.1 | 18.7 | -0.5 [-3.3, +2.7] |
| ours-SFT-r11 | ours-RL-r11 | route_planning | 194 | 27.8 | 26.3 | -1.6 [-4.7, +1.6] |
| ours-SFT | ours-SFT-r11 | obj_appearance_order | 618 | 30.3 | 30.4 | +0.2 [-3.2, +3.4] |
| ours-SFT | ours-SFT-r11 | object_abs_distance | 834 | 5.4 | 6.7 | +1.3 [-0.1, +2.8] |
| ours-SFT | ours-SFT-r11 | object_counting | 565 | 46.0 | 43.5 | -2.5 [-5.1, +0.1] |
| ours-SFT | ours-SFT-r11 | object_rel_direction_easy | 217 | 51.1 | 49.3 | -1.8 [-8.3, +4.6] |
| ours-SFT | ours-SFT-r11 | object_rel_direction_hard | 373 | 23.9 | 30.6 | +6.7 [+2.4, +11.3] |
| ours-SFT | ours-SFT-r11 | object_rel_direction_medium | 378 | 43.9 | 43.1 | -0.8 [-5.3, +4.0] |
| ours-SFT | ours-SFT-r11 | object_rel_distance | 710 | 35.9 | 30.6 | -5.3 [-8.4, -2.1] |
| ours-SFT | ours-SFT-r11 | object_size_estimation | 953 | 29.7 | 25.2 | -4.5 [-6.8, -2.3] |
| ours-SFT | ours-SFT-r11 | room_size_estimation | 288 | 32.0 | 19.1 | -12.9 [-17.2, -8.9] |
| ours-SFT | ours-SFT-r11 | route_planning | 194 | 26.8 | 27.8 | +1.0 [-5.2, +6.7] |
| base-7B | SI-7B | obj_appearance_order | 618 | 34.6 | 64.4 | +29.8 [+24.4, +35.0] |
| base-7B | SI-7B | object_abs_distance | 834 | 22.6 | 23.3 | +0.7 [-2.5, +3.7] |
| base-7B | SI-7B | object_counting | 565 | 41.4 | 65.1 | +23.7 [+20.0, +27.5] |
| base-7B | SI-7B | object_rel_direction_easy | 217 | 46.5 | 53.0 | +6.5 [-1.8, +15.2] |
| base-7B | SI-7B | object_rel_direction_hard | 373 | 30.0 | 43.2 | +13.1 [+6.2, +19.9] |
| base-7B | SI-7B | object_rel_direction_medium | 378 | 43.9 | 48.9 | +5.0 [-2.9, +13.2] |
| base-7B | SI-7B | object_rel_distance | 710 | 37.3 | 49.4 | +12.1 [+7.3, +17.0] |
| base-7B | SI-7B | object_size_estimation | 953 | 49.0 | 60.5 | +11.5 [+8.9, +14.2] |
| base-7B | SI-7B | room_size_estimation | 288 | 46.2 | 38.9 | -7.3 [-11.9, -2.6] |
| base-7B | SI-7B | route_planning | 194 | 29.4 | 30.4 | +1.0 [-6.7, +9.3] |
| base-7B | SpaceR-7B | obj_appearance_order | 618 | 34.6 | 49.5 | +14.9 [+11.2, +18.6] |
| base-7B | SpaceR-7B | object_abs_distance | 834 | 22.6 | 31.8 | +9.2 [+6.5, +11.8] |
| base-7B | SpaceR-7B | object_counting | 565 | 41.4 | 49.9 | +8.5 [+5.7, +11.4] |
| base-7B | SpaceR-7B | object_rel_direction_easy | 217 | 46.5 | 51.1 | +4.6 [-2.8, +12.4] |
| base-7B | SpaceR-7B | object_rel_direction_hard | 373 | 30.0 | 40.8 | +10.7 [+6.4, +15.0] |
| base-7B | SpaceR-7B | object_rel_direction_medium | 378 | 43.9 | 46.0 | +2.1 [-4.2, +8.2] |
| base-7B | SpaceR-7B | object_rel_distance | 710 | 37.3 | 39.3 | +2.0 [-1.3, +5.1] |
| base-7B | SpaceR-7B | object_size_estimation | 953 | 49.0 | 61.5 | +12.5 [+10.2, +15.0] |
| base-7B | SpaceR-7B | room_size_estimation | 288 | 46.2 | 47.3 | +1.1 [-3.4, +5.6] |
| base-7B | SpaceR-7B | route_planning | 194 | 29.4 | 32.0 | +2.6 [-1.6, +6.7] |
| base-7B | VideoR1-7B | obj_appearance_order | 618 | 34.6 | 32.2 | -2.4 [-6.2, +1.5] |
| base-7B | VideoR1-7B | object_abs_distance | 834 | 22.6 | 29.8 | +7.2 [+4.2, +10.3] |
| base-7B | VideoR1-7B | object_counting | 565 | 41.4 | 24.9 | -16.5 [-19.5, -13.3] |
| base-7B | VideoR1-7B | object_rel_direction_easy | 217 | 46.5 | 48.4 | +1.8 [-3.7, +7.4] |
| base-7B | VideoR1-7B | object_rel_direction_hard | 373 | 30.0 | 21.4 | -8.6 [-13.9, -3.5] |
| base-7B | VideoR1-7B | object_rel_direction_medium | 378 | 43.9 | 36.5 | -7.4 [-12.2, -2.9] |
| base-7B | VideoR1-7B | object_rel_distance | 710 | 37.3 | 38.9 | +1.6 [-2.0, +5.1] |
| base-7B | VideoR1-7B | object_size_estimation | 953 | 49.0 | 36.2 | -12.8 [-15.7, -9.8] |
| base-7B | VideoR1-7B | room_size_estimation | 288 | 46.2 | 35.2 | -11.0 [-14.8, -7.3] |
| base-7B | VideoR1-7B | route_planning | 194 | 29.4 | 28.4 | -1.0 [-7.2, +5.2] |
| base-7B | ViLASR-7B | obj_appearance_order | 618 | 34.6 | 54.7 | +20.1 [+15.9, +24.8] |
| base-7B | ViLASR-7B | object_abs_distance | 834 | 22.6 | 33.9 | +11.3 [+8.5, +14.3] |
| base-7B | ViLASR-7B | object_counting | 565 | 41.4 | 52.9 | +11.6 [+8.8, +14.6] |
| base-7B | ViLASR-7B | object_rel_direction_easy | 217 | 46.5 | 49.3 | +2.8 [-3.7, +9.2] |
| base-7B | ViLASR-7B | object_rel_direction_hard | 373 | 30.0 | 42.9 | +12.9 [+8.0, +17.7] |
| base-7B | ViLASR-7B | object_rel_direction_medium | 378 | 43.9 | 45.5 | +1.6 [-2.6, +5.8] |
| base-7B | ViLASR-7B | object_rel_distance | 710 | 37.3 | 41.1 | +3.8 [-0.1, +7.8] |
| base-7B | ViLASR-7B | object_size_estimation | 953 | 49.0 | 59.5 | +10.5 [+8.0, +13.2] |
| base-7B | ViLASR-7B | room_size_estimation | 288 | 46.2 | 32.5 | -13.7 [-20.0, -7.7] |
| base-7B | ViLASR-7B | route_planning | 194 | 29.4 | 29.4 | +0.0 [-6.2, +6.2] |
| base-7B | VST-7B-RL | obj_appearance_order | 618 | 34.6 | 61.5 | +26.9 [+21.8, +31.9] |
| base-7B | VST-7B-RL | object_abs_distance | 834 | 22.6 | 41.4 | +18.8 [+15.5, +22.0] |
| base-7B | VST-7B-RL | object_counting | 565 | 41.4 | 64.8 | +23.4 [+19.6, +27.4] |
| base-7B | VST-7B-RL | object_rel_direction_easy | 217 | 46.5 | 53.0 | +6.5 [-1.9, +16.1] |
| base-7B | VST-7B-RL | object_rel_direction_hard | 373 | 30.0 | 45.0 | +15.0 [+7.5, +22.8] |
| base-7B | VST-7B-RL | object_rel_direction_medium | 378 | 43.9 | 51.6 | +7.7 [+2.4, +13.2] |
| base-7B | VST-7B-RL | object_rel_distance | 710 | 37.3 | 50.9 | +13.5 [+8.9, +18.2] |
| base-7B | VST-7B-RL | object_size_estimation | 953 | 49.0 | 71.9 | +22.9 [+20.6, +25.4] |
| base-7B | VST-7B-RL | room_size_estimation | 288 | 46.2 | 64.0 | +17.8 [+13.4, +22.2] |
| base-7B | VST-7B-RL | route_planning | 194 | 29.4 | 43.8 | +14.4 [+4.6, +23.7] |
| base-7B | SenseNovaSI-7B | obj_appearance_order | 618 | 34.6 | 60.4 | +25.7 [+21.4, +30.1] |
| base-7B | SenseNovaSI-7B | object_abs_distance | 834 | 22.6 | 31.2 | +8.6 [+5.4, +11.8] |
| base-7B | SenseNovaSI-7B | object_counting | 565 | 41.4 | 59.6 | +18.2 [+14.7, +21.8] |
| base-7B | SenseNovaSI-7B | object_rel_direction_easy | 217 | 46.5 | 49.8 | +3.2 [-4.2, +11.1] |
| base-7B | SenseNovaSI-7B | object_rel_direction_hard | 373 | 30.0 | 44.5 | +14.5 [+9.1, +19.8] |
| base-7B | SenseNovaSI-7B | object_rel_direction_medium | 378 | 43.9 | 48.4 | +4.5 [-1.9, +10.6] |
| base-7B | SenseNovaSI-7B | object_rel_distance | 710 | 37.3 | 46.3 | +9.0 [+5.1, +13.0] |
| base-7B | SenseNovaSI-7B | object_size_estimation | 953 | 49.0 | 68.5 | +19.4 [+17.3, +21.7] |
| base-7B | SenseNovaSI-7B | room_size_estimation | 288 | 46.2 | 51.2 | +5.0 [+0.7, +9.5] |
| base-7B | SenseNovaSI-7B | route_planning | 194 | 29.4 | 32.0 | +2.6 [-6.2, +10.8] |
| base-3B | SI-3B | obj_appearance_order | 618 | 29.1 | 68.1 | +39.0 [+34.0, +44.0] |
| base-3B | SI-3B | object_abs_distance | 834 | 21.7 | 29.3 | +7.6 [+4.7, +10.4] |
| base-3B | SI-3B | object_counting | 565 | 17.4 | 64.9 | +47.5 [+43.7, +51.1] |
| base-3B | SI-3B | object_rel_direction_easy | 217 | 49.3 | 51.1 | +1.8 [-7.4, +10.6] |
| base-3B | SI-3B | object_rel_direction_hard | 373 | 36.7 | 42.9 | +6.2 [+0.5, +12.1] |
| base-3B | SI-3B | object_rel_direction_medium | 378 | 40.0 | 46.3 | +6.3 [-1.3, +14.0] |
| base-3B | SI-3B | object_rel_distance | 710 | 33.7 | 50.3 | +16.6 [+12.0, +21.3] |
| base-3B | SI-3B | object_size_estimation | 953 | 15.1 | 59.6 | +44.6 [+41.9, +47.2] |
| base-3B | SI-3B | room_size_estimation | 288 | 27.2 | 44.1 | +16.9 [+12.2, +21.9] |
| base-3B | SI-3B | route_planning | 194 | 28.4 | 31.4 | +3.1 [-5.7, +11.3] |
| SpaceR-7B | SpaceR-7B-ownpx | obj_appearance_order | 618 | 49.5 | 52.6 | +3.1 [+0.2, +6.2] |
| SpaceR-7B | SpaceR-7B-ownpx | object_abs_distance | 834 | 31.8 | 30.2 | -1.6 [-3.2, -0.0] |
| SpaceR-7B | SpaceR-7B-ownpx | object_counting | 565 | 49.9 | 50.4 | +0.6 [-1.4, +2.5] |
| SpaceR-7B | SpaceR-7B-ownpx | object_rel_direction_easy | 217 | 51.1 | 48.9 | -2.3 [-6.5, +1.4] |
| SpaceR-7B | SpaceR-7B-ownpx | object_rel_direction_hard | 373 | 40.8 | 40.2 | -0.5 [-2.1, +0.8] |
| SpaceR-7B | SpaceR-7B-ownpx | object_rel_direction_medium | 378 | 46.0 | 46.6 | +0.5 [-0.5, +1.6] |
| SpaceR-7B | SpaceR-7B-ownpx | object_rel_distance | 710 | 39.3 | 40.4 | +1.1 [-1.8, +4.1] |
| SpaceR-7B | SpaceR-7B-ownpx | object_size_estimation | 953 | 61.5 | 61.8 | +0.3 [-0.6, +1.2] |
| SpaceR-7B | SpaceR-7B-ownpx | room_size_estimation | 288 | 47.3 | 49.3 | +2.1 [-0.2, +4.5] |
| SpaceR-7B | SpaceR-7B-ownpx | route_planning | 194 | 32.0 | 31.4 | -0.5 [-2.1, +1.0] |
| VideoR1-7B | VideoR1-7B-ownpx | obj_appearance_order | 618 | 32.2 | 35.3 | +3.1 [+0.5, +5.5] |
| VideoR1-7B | VideoR1-7B-ownpx | object_abs_distance | 834 | 29.8 | 29.6 | -0.1 [-0.8, +0.5] |
| VideoR1-7B | VideoR1-7B-ownpx | object_counting | 565 | 24.9 | 30.2 | +5.3 [+3.0, +7.7] |
| VideoR1-7B | VideoR1-7B-ownpx | object_rel_direction_easy | 217 | 48.4 | 50.7 | +2.3 [-2.8, +7.4] |
| VideoR1-7B | VideoR1-7B-ownpx | object_rel_direction_hard | 373 | 21.4 | 15.0 | -6.4 [-9.9, -3.2] |
| VideoR1-7B | VideoR1-7B-ownpx | object_rel_direction_medium | 378 | 36.5 | 36.8 | +0.3 [-2.4, +2.9] |
| VideoR1-7B | VideoR1-7B-ownpx | object_rel_distance | 710 | 38.9 | 38.7 | -0.1 [-3.2, +2.7] |
| VideoR1-7B | VideoR1-7B-ownpx | object_size_estimation | 953 | 36.2 | 35.9 | -0.3 [-1.2, +0.6] |
| VideoR1-7B | VideoR1-7B-ownpx | room_size_estimation | 288 | 35.2 | 32.1 | -3.1 [-4.9, -1.4] |
| VideoR1-7B | VideoR1-7B-ownpx | route_planning | 194 | 28.4 | 27.8 | -0.5 [-4.1, +3.1] |
| SenseNovaSI-7B | SenseNovaSI-7B-ownpx | obj_appearance_order | 618 | 60.4 | 59.4 | -1.0 [-4.5, +2.6] |
| SenseNovaSI-7B | SenseNovaSI-7B-ownpx | object_abs_distance | 834 | 31.2 | 36.0 | +4.8 [+2.7, +6.8] |
| SenseNovaSI-7B | SenseNovaSI-7B-ownpx | object_counting | 565 | 59.6 | 60.9 | +1.4 [-1.3, +3.9] |
| SenseNovaSI-7B | SenseNovaSI-7B-ownpx | object_rel_direction_easy | 217 | 49.8 | 50.2 | +0.5 [-7.4, +8.8] |
| SenseNovaSI-7B | SenseNovaSI-7B-ownpx | object_rel_direction_hard | 373 | 44.5 | 44.2 | -0.3 [-3.8, +3.0] |
| SenseNovaSI-7B | SenseNovaSI-7B-ownpx | object_rel_direction_medium | 378 | 48.4 | 48.1 | -0.3 [-4.5, +4.0] |
| SenseNovaSI-7B | SenseNovaSI-7B-ownpx | object_rel_distance | 710 | 46.3 | 46.8 | +0.4 [-3.1, +3.9] |
| SenseNovaSI-7B | SenseNovaSI-7B-ownpx | object_size_estimation | 953 | 68.5 | 67.5 | -0.9 [-2.1, +0.3] |
| SenseNovaSI-7B | SenseNovaSI-7B-ownpx | room_size_estimation | 288 | 51.2 | 58.6 | +7.4 [+3.9, +11.0] |
| SenseNovaSI-7B | SenseNovaSI-7B-ownpx | route_planning | 194 | 32.0 | 34.0 | +2.1 [-2.6, +6.2] |

**VSTI-Bench 分题型**（VSTI 按 VLM-3R 的五类；差 = B − A，点）

| A | B | 题型 | 题数 | A | B | 差 [95% 区间] |
|---|---|---|---|---|---|---|
| base-7B | ours-SFT | camera_obj_abs_dist | 905 | 31.1 | 4.4 | -26.7 [-29.1, -24.2] |
| base-7B | ours-SFT | camera_displacement | 833 | 9.2 | 5.7 | -3.4 [-5.3, -1.5] |
| base-7B | ours-SFT | camera_movement_direction | 913 | 45.1 | 30.6 | -14.6 [-18.5, -10.5] |
| base-7B | ours-SFT | obj_obj_relative_pos | 1645 | 68.1 | 61.0 | -7.0 [-9.4, -4.9] |
| base-7B | ours-SFT | camera_obj_rel_dist | 1440 | 53.9 | 48.5 | -5.3 [-7.8, -3.0] |
| base-7B | ours-RL-50 | camera_obj_abs_dist | 905 | 31.1 | 4.0 | -27.1 [-29.6, -24.6] |
| base-7B | ours-RL-50 | camera_displacement | 833 | 9.2 | 6.3 | -2.9 [-4.8, -1.0] |
| base-7B | ours-RL-50 | camera_movement_direction | 913 | 45.1 | 29.9 | -15.2 [-19.2, -11.3] |
| base-7B | ours-RL-50 | obj_obj_relative_pos | 1645 | 68.1 | 62.0 | -6.1 [-8.4, -4.0] |
| base-7B | ours-RL-50 | camera_obj_rel_dist | 1440 | 53.9 | 49.2 | -4.7 [-7.1, -2.4] |
| base-7B | ours-RL-150 | camera_obj_abs_dist | 905 | 31.1 | 4.0 | -27.1 [-29.5, -24.7] |
| base-7B | ours-RL-150 | camera_displacement | 833 | 9.2 | 6.2 | -2.9 [-4.9, -1.1] |
| base-7B | ours-RL-150 | camera_movement_direction | 913 | 45.1 | 29.1 | -16.0 [-20.0, -11.9] |
| base-7B | ours-RL-150 | obj_obj_relative_pos | 1645 | 68.1 | 62.1 | -6.0 [-8.3, -3.8] |
| base-7B | ours-RL-150 | camera_obj_rel_dist | 1440 | 53.9 | 48.4 | -5.5 [-7.8, -3.2] |
| base-7B | ours-SFT-r11 | camera_obj_abs_dist | 905 | 31.1 | 4.7 | -26.4 [-28.9, -23.9] |
| base-7B | ours-SFT-r11 | camera_displacement | 833 | 9.2 | 8.8 | -0.4 [-2.6, +1.8] |
| base-7B | ours-SFT-r11 | camera_movement_direction | 913 | 45.1 | 26.6 | -18.5 [-22.9, -14.0] |
| base-7B | ours-SFT-r11 | obj_obj_relative_pos | 1645 | 68.1 | 58.2 | -9.9 [-12.5, -7.5] |
| base-7B | ours-SFT-r11 | camera_obj_rel_dist | 1440 | 53.9 | 47.8 | -6.1 [-8.8, -3.5] |
| base-7B | ours-RL-r11 | camera_obj_abs_dist | 905 | 31.1 | 5.8 | -25.3 [-27.8, -22.8] |
| base-7B | ours-RL-r11 | camera_displacement | 833 | 9.2 | 8.9 | -0.3 [-2.4, +1.9] |
| base-7B | ours-RL-r11 | camera_movement_direction | 913 | 45.1 | 25.9 | -19.3 [-23.6, -14.9] |
| base-7B | ours-RL-r11 | obj_obj_relative_pos | 1645 | 68.1 | 59.1 | -9.0 [-11.4, -6.5] |
| base-7B | ours-RL-r11 | camera_obj_rel_dist | 1440 | 53.9 | 47.2 | -6.7 [-9.2, -4.1] |
| ours-SFT | ours-RL-50 | camera_obj_abs_dist | 905 | 4.4 | 4.0 | -0.4 [-1.0, +0.1] |
| ours-SFT | ours-RL-50 | camera_displacement | 833 | 5.7 | 6.3 | +0.6 [-0.2, +1.4] |
| ours-SFT | ours-RL-50 | camera_movement_direction | 913 | 30.6 | 29.9 | -0.7 [-1.8, +0.4] |
| ours-SFT | ours-RL-50 | obj_obj_relative_pos | 1645 | 61.0 | 62.0 | +0.9 [+0.2, +1.6] |
| ours-SFT | ours-RL-50 | camera_obj_rel_dist | 1440 | 48.5 | 49.2 | +0.6 [-0.2, +1.5] |
| ours-SFT | ours-RL-150 | camera_obj_abs_dist | 905 | 4.4 | 4.0 | -0.4 [-0.9, +0.1] |
| ours-SFT | ours-RL-150 | camera_displacement | 833 | 5.7 | 6.2 | +0.5 [-0.2, +1.2] |
| ours-SFT | ours-RL-150 | camera_movement_direction | 913 | 30.6 | 29.1 | -1.4 [-2.4, -0.3] |
| ours-SFT | ours-RL-150 | obj_obj_relative_pos | 1645 | 61.0 | 62.1 | +1.0 [+0.2, +1.9] |
| ours-SFT | ours-RL-150 | camera_obj_rel_dist | 1440 | 48.5 | 48.4 | -0.1 [-1.0, +0.6] |
| ours-SFT-r11 | ours-RL-r11 | camera_obj_abs_dist | 905 | 4.7 | 5.8 | +1.1 [+0.2, +1.9] |
| ours-SFT-r11 | ours-RL-r11 | camera_displacement | 833 | 8.8 | 8.9 | +0.1 [-1.1, +1.5] |
| ours-SFT-r11 | ours-RL-r11 | camera_movement_direction | 913 | 26.6 | 25.9 | -0.8 [-2.3, +0.7] |
| ours-SFT-r11 | ours-RL-r11 | obj_obj_relative_pos | 1645 | 58.2 | 59.1 | +0.9 [-0.3, +2.2] |
| ours-SFT-r11 | ours-RL-r11 | camera_obj_rel_dist | 1440 | 47.8 | 47.2 | -0.6 [-1.9, +0.8] |
| ours-SFT | ours-SFT-r11 | camera_obj_abs_dist | 905 | 4.4 | 4.7 | +0.3 [-0.7, +1.3] |
| ours-SFT | ours-SFT-r11 | camera_displacement | 833 | 5.7 | 8.8 | +3.0 [+1.3, +4.7] |
| ours-SFT | ours-SFT-r11 | camera_movement_direction | 913 | 30.6 | 26.6 | -3.9 [-6.6, -1.3] |
| ours-SFT | ours-SFT-r11 | obj_obj_relative_pos | 1645 | 61.0 | 58.2 | -2.9 [-5.1, -0.6] |
| ours-SFT | ours-SFT-r11 | camera_obj_rel_dist | 1440 | 48.5 | 47.8 | -0.8 [-2.9, +1.3] |
| base-7B | SI-7B | camera_obj_abs_dist | 905 | 31.1 | 33.7 | +2.6 [-0.1, +5.4] |
| base-7B | SI-7B | camera_displacement | 833 | 9.2 | 30.1 | +20.9 [+18.3, +23.5] |
| base-7B | SI-7B | camera_movement_direction | 913 | 45.1 | 49.4 | +4.3 [+0.6, +8.2] |
| base-7B | SI-7B | obj_obj_relative_pos | 1645 | 68.1 | 70.3 | +2.2 [-0.6, +4.9] |
| base-7B | SI-7B | camera_obj_rel_dist | 1440 | 53.9 | 70.6 | +16.7 [+13.7, +19.6] |
| base-7B | SpaceR-7B | camera_obj_abs_dist | 905 | 31.1 | 37.2 | +6.1 [+2.9, +9.3] |
| base-7B | SpaceR-7B | camera_displacement | 833 | 9.2 | 12.7 | +3.5 [+1.0, +5.9] |
| base-7B | SpaceR-7B | camera_movement_direction | 913 | 45.1 | 47.9 | +2.7 [+0.0, +5.4] |
| base-7B | SpaceR-7B | obj_obj_relative_pos | 1645 | 68.1 | 67.4 | -0.7 [-2.4, +1.2] |
| base-7B | SpaceR-7B | camera_obj_rel_dist | 1440 | 53.9 | 61.6 | +7.7 [+5.7, +9.6] |
| base-7B | VideoR1-7B | camera_obj_abs_dist | 905 | 31.1 | 37.6 | +6.6 [+3.5, +9.6] |
| base-7B | VideoR1-7B | camera_displacement | 833 | 9.2 | 9.5 | +0.4 [-1.5, +2.1] |
| base-7B | VideoR1-7B | camera_movement_direction | 913 | 45.1 | 51.3 | +6.1 [+2.7, +9.6] |
| base-7B | VideoR1-7B | obj_obj_relative_pos | 1645 | 68.1 | 65.5 | -2.6 [-5.0, -0.4] |
| base-7B | VideoR1-7B | camera_obj_rel_dist | 1440 | 53.9 | 59.1 | +5.2 [+2.5, +7.8] |
| base-7B | ViLASR-7B | camera_obj_abs_dist | 905 | 31.1 | 37.2 | +6.1 [+2.8, +9.5] |
| base-7B | ViLASR-7B | camera_displacement | 833 | 9.2 | 15.5 | +6.3 [+3.7, +8.9] |
| base-7B | ViLASR-7B | camera_movement_direction | 913 | 45.1 | 48.4 | +3.3 [+0.7, +5.9] |
| base-7B | ViLASR-7B | obj_obj_relative_pos | 1645 | 68.1 | 66.1 | -2.0 [-3.9, -0.1] |
| base-7B | ViLASR-7B | camera_obj_rel_dist | 1440 | 53.9 | 68.2 | +14.3 [+11.8, +16.7] |
| base-7B | VST-7B-RL | camera_obj_abs_dist | 905 | 31.1 | 36.9 | +5.8 [+2.3, +9.2] |
| base-7B | VST-7B-RL | camera_displacement | 833 | 9.2 | 24.9 | +15.7 [+13.1, +18.3] |
| base-7B | VST-7B-RL | camera_movement_direction | 913 | 45.1 | 36.7 | -8.4 [-12.7, -3.9] |
| base-7B | VST-7B-RL | obj_obj_relative_pos | 1645 | 68.1 | 67.5 | -0.6 [-3.3, +2.1] |
| base-7B | VST-7B-RL | camera_obj_rel_dist | 1440 | 53.9 | 77.1 | +23.2 [+20.3, +26.0] |
| base-7B | SenseNovaSI-7B | camera_obj_abs_dist | 905 | 31.1 | 44.6 | +13.5 [+11.0, +16.1] |
| base-7B | SenseNovaSI-7B | camera_displacement | 833 | 9.2 | 23.4 | +14.2 [+12.1, +16.5] |
| base-7B | SenseNovaSI-7B | camera_movement_direction | 913 | 45.1 | 45.5 | +0.3 [-2.7, +3.3] |
| base-7B | SenseNovaSI-7B | obj_obj_relative_pos | 1645 | 68.1 | 67.5 | -0.6 [-2.8, +1.6] |
| base-7B | SenseNovaSI-7B | camera_obj_rel_dist | 1440 | 53.9 | 68.0 | +14.1 [+11.6, +16.6] |
| base-3B | SI-3B | camera_obj_abs_dist | 905 | 14.2 | 33.5 | +19.4 [+16.9, +21.8] |
| base-3B | SI-3B | camera_displacement | 833 | 21.6 | 28.6 | +7.0 [+3.9, +10.4] |
| base-3B | SI-3B | camera_movement_direction | 913 | 48.1 | 44.0 | -4.0 [-7.5, -0.7] |
| base-3B | SI-3B | obj_obj_relative_pos | 1645 | 57.7 | 74.6 | +16.9 [+14.0, +19.8] |
| base-3B | SI-3B | camera_obj_rel_dist | 1440 | 59.7 | 71.4 | +11.7 [+8.6, +14.7] |
