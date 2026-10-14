#!/bin/bash
# Numbers in REPORT_20261014.md (run from anywhere on fujian). Old questions replaced by later runs are excluded.
cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite
S=exps/interiorgs_qa_sample_20261014; T=exps/interiorgs_qa_stats_20261014
EXCL="interiorgs_qa_sample_20261014/multi:M6,interiorgs_qa_sample_20261014/multi:M11,topup_multi_a:M6,topup_multi_a:M11,interiorgs_qa_sample_20261014/single:S1:BL-03,interiorgs_qa_sample_20261014/single:S2:BL-03,interiorgs_qa_sample_20261014/single:S2:3D-02,interiorgs_qa_sample_20261014/single:S4:BL-03,interiorgs_qa_sample_20261014/multi:M1:BL-04,interiorgs_qa_stats_20261014/single:S1:BL-03,interiorgs_qa_stats_20261014/single:S2:BL-03,interiorgs_qa_stats_20261014/single:S2:3D-02,interiorgs_qa_stats_20261014/single:S4:BL-03,interiorgs_qa_stats_20261014/multi:M1:BL-04,interiorgs_qa_stats_20261014/multi:M6,interiorgs_qa_stats_20261014/multi:M11,qid:8704d4e110bd0a20,qid:5d1ee32b9f7ff454"
PYTHONPATH=. /mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python -m data_gen.interiorgs_qa.report \
  --census exps/interiorgs_qa_census_20261014/census.json \
  --runs $S/single,$S/multi,$S/video,$S/topup_multi_a,$S/topup_multi_b,$S/topup_m3,$S/topup_video \
  --sample-extra $S/topup_pan2,$S/rebalance4 \
  --stats-runs $T/single,$T/multi,$T/video --stats-extra $T/rebalance4_a,$T/rebalance4_b,$T/rebalance4_c,$T/rebalance4_d \
  --rate-override "M3=$S/topup_m3" \
  --scan "M3=$S/topup_m3;M6=$S/topup_pan2;M8=$S/topup_multi_a;M9=$S/topup_multi_b;M10=$S/topup_multi_a;M11=$S/topup_pan2;V5=$S/topup_video;V6=$S/topup_video" \
  --exclude "$EXCL" --out "${1:-exps/interiorgs_qa_report_20261014/report_numbers.json}"
