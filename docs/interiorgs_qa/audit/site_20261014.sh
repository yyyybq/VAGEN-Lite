#!/bin/bash
# Review site of REPORT_20261014.md. Old questions replaced by later runs (bias fixes, old M3 code) are excluded.
cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite
S=exps/interiorgs_qa_sample_20261014
EXCL="interiorgs_qa_sample_20261014/multi:M6,interiorgs_qa_sample_20261014/multi:M11,topup_multi_a:M6,topup_multi_a:M11,interiorgs_qa_sample_20261014/single:S1:BL-03,interiorgs_qa_sample_20261014/single:S2:BL-03,interiorgs_qa_sample_20261014/single:S2:3D-02,interiorgs_qa_sample_20261014/single:S4:BL-03,interiorgs_qa_sample_20261014/multi:M1:BL-04,qid:8704d4e110bd0a20,qid:5d1ee32b9f7ff454"
PYTHONPATH=. /mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python -m data_gen.interiorgs_qa.site \
  --run $S/single --run $S/multi --run $S/video --run $S/topup_multi_a --run $S/topup_multi_b --run $S/topup_m3 \
  --run $S/topup_video --run $S/topup_pan2 --run $S/rebalance4 \
  --trials exps/interiorgs_qa_trials/merged --exclude "$EXCL" --out "${1:-exps/interiorgs_qa_site_20261014}"
