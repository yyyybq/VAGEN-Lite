"""Read-only comparison of two completed CPU capacity audits; writes comparison reports only."""
import json,csv
from pathlib import Path
x=json.loads(Path('/tmp/interiorgs_qa_cpu_audit.json').read_text()); y=json.loads(Path('/tmp/interiorgs_qa_cpu_v3.json').read_text())
a=json.loads(Path('/tmp/interiorgs_qa_ref_identity_v2.json').read_text()); b=json.loads(Path('/tmp/interiorgs_qa_ref_identity_v3.json').read_text())
control={'same_20_selected_scenes':x['selected_train_scenes']==y['selected_train_scenes'],'same_metadata_digest':x['schema']['metadata_manifest_sha256']==y['schema']['metadata_manifest_sha256'],'same_sampling_config':x['sampling']==y['sampling'],'same_all_ref_targets_text_anchor_and_geometry':a==b,'all_same_sampled_positions':all(i['sampled_positions_xy']==j['sampled_positions_xy'] for i,j in zip(x['scenes'],y['scenes'])),'all_same_sampled_target_ids':all(i['sampled_target_ids']==j['sampled_target_ids'] for i,j in zip(x['scenes'],y['scenes'])),'v2_source_hashes_unchanged':x['source_snapshot_unchanged'],'v3_source_hashes_unchanged':y['source_snapshot_unchanged']}
assert all(control.values()),control
metrics=['visible_unique_targets','lateral_unique_pairs','depth_order_unique_pairs','aabb_size_unique_pairs','noncollinear_unique_triples','object_seen_from_ge2_positions','multiview_camera_pairs_ge3_anchors']
totals={k:{'v2':x['totals'][k],'v3':y['totals'][k],'delta':y['totals'][k]-x['totals'][k],'change_percent':round(100*(y['totals'][k]/x['totals'][k]-1),2) if x['totals'][k] else None} for k in metrics}
view_totals={k:{'v2':x['total_view_counts'][k],'v3':y['total_view_counts'][k],'delta':y['total_view_counts'][k]-x['total_view_counts'][k]} for k in x['total_view_counts']}
rows=[]
for i,j in zip(x['scenes'],y['scenes']):
 r={'scene_id':i['scene_id'],'ref_targets':j['eligible_ref_targets'],'views':j['sampled_views']}
 for k in metrics:
  r[k+'_v2']=i[k];r[k+'_v3']=j[k];r[k+'_delta']=j[k]-i[k]
 for k in x['total_view_counts']:
  r['view_'+k+'_v2']=i['view_counts'][k];r['view_'+k+'_v3']=j['view_counts'][k];r['view_'+k+'_delta']=j['view_counts'][k]-i['view_counts'][k]
 rows.append(r)
notes=['Both audits use r5 split and v10_r5 referring rules, same 20 scenes, same targets, same ref_anchor identities, same grid positions and yaws.','V3 code is git archive of 3a964c795f82e4014572d7dc61425cd22d7816f3 from VAGEN-Lite; no r4 checkout was used.','V3 replaces the room-outline visibility walls with structure-wall panels and door holes and adds curtain occlusion. It is not a monotonic reject-only filter; some scene counts can increase.','Geometric candidate counts remain unvalidated by RGB. Neither the before nor after counts prove benchmark question correctness.','Multiview numbers remain structural upper bounds without true-multiview-necessity filtering. Video/appearance-order QA remains unestimated.']
out={'controls':control,'v2_rule':x['visibility_rule'],'v3_rule':y['visibility_rule'],'v2_run_commit':x['r5_recorded_commit'],'v3_source_commit':y['analysis_source_commit'],'v2_seconds':x['elapsed_seconds'],'v3_seconds':y['elapsed_seconds'],'totals':totals,'view_totals':view_totals,'per_scene':rows,'notes':notes}
Path('/tmp/interiorgs_qa_cpu_v2_v3_comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
with open('/tmp/interiorgs_qa_cpu_v2_v3_comparison.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
labels={'visible_unique_targets':'可见独立目标','lateral_unique_pairs':'左右独立对象对','depth_order_unique_pairs':'深度独立对象对','aabb_size_unique_pairs':'AABB尺寸比较对','noncollinear_unique_triples':'非共线三物体组合','object_seen_from_ge2_positions':'≥2位置可见目标','multiview_camera_pairs_ge3_anchors':'≥3共享目标视图对（结构上界）'}
lines=['# face_visible_v2 / v3 CPU 敏感性核验','','保持 r5 split、v10_r5 指代、seed 20261014、20 场景、467 相机位置、1868 视图及336被采样目标一致。完整指代文本、ref_anchor 与目标几何的逐场景 SHA 也一致。','v2 使用 r5 原运行源码；v3 使用指定提交 3a964c795f82e4014572d7dc61425cd22d7816f3 的 /tmp 快照。两次源文件前后 SHA 均未变。没有使用 r4。',f'v3 CPU 分析耗时 {y["elapsed_seconds"]} 秒，20/20 场景成功。','','|共享几何能力|v2|v3|变化|变化比例|','|---|---:|---:|---:|---:|']
for k in metrics:
 d=totals[k];lines.append(f'|{labels[k]}|{d["v2"]}|{d["v3"]}|{d["delta"]:+d}|{d["change_percent"]:+.2f}%|')
lines += ['','以下逐场景写作 v2→v3，均为几何候选，不是验收问答。','','|场景|左右对|深度对|尺寸对|三物体组合|≥3共享目标视图对|','|---|---:|---:|---:|---:|---:|']
for r in rows:
 ks=['lateral_unique_pairs','depth_order_unique_pairs','aabb_size_unique_pairs','noncollinear_unique_triples','multiview_camera_pairs_ge3_anchors']
 lines.append('|'+r['scene_id']+'|'+'|'.join(f'{r[k+"_v2"]}→{r[k+"_v3"]}' for k in ks)+'|')
lines += ['','v3 改为 structure 墙面与门洞，并把窗帘作为遮挡物；并非简单从 v2 中删除样例，因此少数场景有新增候选。没有 RGB 验收，不能把 v2 全部减少项称为已证明错误，也不能把 v3 剩余项称为正确题。','多图仍缺必须融合多图的筛选；视频路线、appearance order 未估。原始 OBB、房间类型和尺度歧义等资产限制不随可见性规则改变。']
Path('/tmp/interiorgs_qa_cpu_v2_v3_comparison.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'controls':control,'totals':totals},ensure_ascii=False,indent=2))
