#!/bin/bash
# Review site of REPORT_20261014.md. Old questions replaced by later runs (bias fixes, old M3 code) are excluded;
# S7 (3DSRBench parallel / perpendicular, 3D-11) is dropped entirely (user decision of 2026-10-15, during labelling);
# so are the MMSI questions that fix the compass by one stated direction ("P is to the northeast of Q; where is R
# from Q?"): object-object (M2), object-region and region-region (M3) (user decision of 2026-10-19, during labelling).
cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite
S=exps/interiorgs_qa_sample_20261014
EXCL="interiorgs_qa_sample_20261014/single:S7,interiorgs_qa_sample_20261014/multi:M2:MMSI-position-object-object,topup_m3:M3:MMSI-position-object-region,topup_m3:M3:MMSI-position-region-region,interiorgs_qa_sample_20261014/multi:M6,interiorgs_qa_sample_20261014/multi:M11,topup_multi_a:M6,topup_multi_a:M11,interiorgs_qa_sample_20261014/single:S1:BL-03,interiorgs_qa_sample_20261014/single:S2:BL-03,interiorgs_qa_sample_20261014/single:S2:3D-02,interiorgs_qa_sample_20261014/single:S4:BL-03,interiorgs_qa_sample_20261014/multi:M1:BL-04,qid:8704d4e110bd0a20,qid:5d1ee32b9f7ff454"
PYTHONPATH=. /mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python -m data_gen.interiorgs_qa.site \
  --run $S/single --run $S/multi --run $S/video --run $S/topup_multi_a --run $S/topup_multi_b --run $S/topup_m3 \
  --run $S/topup_video --run $S/topup_pan2 --run $S/rebalance4 \
  --trials exps/interiorgs_qa_trials/merged --exclude "$EXCL" --out "${1:-exps/interiorgs_qa_site_20261014}"
