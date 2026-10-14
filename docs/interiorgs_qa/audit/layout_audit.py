#!/usr/bin/env python3
"""Read-only CPU census of raw OBB and triple layouts for the 20 frozen sampled target sets.
No rendering, visibility recomputation, QA generation, or production writes.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,itertools,math,time,hashlib,csv,datetime
from pathlib import Path
import numpy as np
START=time.perf_counter()
BASE=Path('/tmp/interiorgs_qa_cpu_v3.json')
base=json.loads(BASE.read_text());root=Path(base['metadata_root'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
thresholds={'corner_reconstruction_abs_tol_m':1e-5,'orthogonality_cos_abs_tol':1e-5,'min_edge_length_m':1e-5,'max_edge_size_ratio_min':1.2,'triple_min_xy_pair_distance_m':0.5,'near_linear_max_point_to_other_two_line_distance_m':0.15,'near_linear_min_max_xy_pair_distance_m':1.0,'rightangle_min_deg':80.0,'rightangle_max_deg':100.0}
result={'audit_kind':'raw_OBB_and_layout_structural_precursors_only','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'input_audit':str(BASE),'input_rule':base['visibility_rule'],'same_selected_scenes':base['selected_train_scenes'],'thresholds':thresholds,'inputs_sha256':{str(BASE):sha(BASE),__file__:sha(__file__)},'scenes':[],'scope':['Only the exact sampled_target_ids from the v3 audit are enumerated (not all eligible or raw objects).','Original eight vertices are used; none of these counts are filtered by visibility.','The prior audit did not persist per-view visible target IDs. No visibility was recomputed.','Counts are unordered distinct instance pairs/triples, not questions, images, trajectories or multi-view-valid cases.','Near-collinear and near-right-angle are world XY configurations of object centers; neither proves semantic object fronts or viewpoint transformation ambiguity.','Five-object relative-distance benchmark filtering is not implemented.']}
for row in base['scenes']:
 sid=row['scene_id'];lp=root/sid/'labels.json';result['inputs_sha256'][str(lp)]=sha(lp)
 records={str(o.get('ins_id',o.get('id',''))):o for o in json.loads(lp.read_text())}
 valid=[];invalid=[];rotated=[];details=[]
 for oid in row['sampled_target_ids']:
  obj=records.get(oid,{});bb=obj.get('bounding_box') or []
  if len(bb)!=8:invalid.append({'id':oid,'reason':'missing_8_vertices'});continue
  xyz=np.array([[q['x'],q['y'],q['z']] for q in bb],float)
  e1,e2,e3=xyz[1]-xyz[0],xyz[3]-xyz[0],xyz[4]-xyz[0]
  edges=[e1,e2,e3];lengths=np.array([np.linalg.norm(e) for e in edges])
  pred=np.array([xyz[0],xyz[0]+e1,xyz[0]+e1+e2,xyz[0]+e2,xyz[0]+e3,xyz[0]+e1+e3,xyz[0]+e1+e2+e3,xyz[0]+e2+e3])
  ordered=bool(np.max(np.abs(xyz-pred))<=1e-5);nondeg=bool(lengths.min()>1e-5)
  orth=bool(nondeg and max(abs(float(np.dot(a,b)))/(np.linalg.norm(a)*np.linalg.norm(b)) for a,b in [(e1,e2),(e1,e3),(e2,e3)])<=1e-5)
  if not(ordered and orth):invalid.append({'id':oid,'reason':'corner_order_or_orthogonality_or_degeneracy','ordered':ordered,'nondegenerate':nondeg,'orthogonal':orth});continue
  lo=xyz.min(0);hi=xyz.max(0);axis=bool(np.all(np.minimum(abs(xyz-lo),abs(xyz-hi))<=1e-6))
  d={'id':oid,'label':obj.get('label'),'center':xyz.mean(0).tolist(),'obb_edge_lengths_m':lengths.tolist(),'obb_longest_edge_m':float(lengths.max()),'aabb_longest_edge_m':float((hi-lo).max()),'axis_aligned':axis}
  valid.append(d);details.append(d)
  if not axis:rotated.append(oid)
 pairs=list(itertools.combinations(valid,2))
 obb_pairs=[(a['id'],b['id']) for a,b in pairs if max(a['obb_longest_edge_m'],b['obb_longest_edge_m'])>=1.2*min(a['obb_longest_edge_m'],b['obb_longest_edge_m'])]
 aabb_pairs=[(a['id'],b['id']) for a,b in pairs if max(a['aabb_longest_edge_m'],b['aabb_longest_edge_m'])>=1.2*min(a['aabb_longest_edge_m'],b['aabb_longest_edge_m'])]
 sep_count=0;linear=[];right=[]
 for aa,bb,cc in itertools.combinations(valid,3):
  pts=np.array([aa['center'][:2],bb['center'][:2],cc['center'][:2]])
  dis=[float(np.linalg.norm(pts[a]-pts[b])) for a,b in [(0,1),(0,2),(1,2)]]
  if min(dis)<0.5:continue
  sep_count+=1
  area2=abs(float(np.cross(pts[1]-pts[0],pts[2]-pts[0])))
  maxline=area2/min(dis) # maximum of all three point-to-other-two infinite-line distances
  ids=[aa['id'],bb['id'],cc['id']]
  if maxline<=0.15 and max(dis)>=1.0:linear.append(ids)
  angles=[]
  for j in range(3):
   others=[k for k in range(3) if k!=j];u,v=pts[others[0]]-pts[j],pts[others[1]]-pts[j]
   angles.append(math.degrees(math.acos(float(np.clip(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)),-1,1)))))
  if any(80<=a<=100 for a in angles):right.append(ids)
 result['scenes'].append({'scene_id':sid,'sampled_targets':len(row['sampled_target_ids']),'valid_raw_obb_targets':len(valid),'rotated_raw_obb_targets':len(rotated),'invalid_raw_obb_targets':len(invalid),'obb_maxedge_ratio_pairs':len(obb_pairs),'aabb_maxedge_ratio_pairs_same_objects':len(aabb_pairs),'obb_vs_aabb_size_pair_symmetric_difference':len(set(obb_pairs)^set(aabb_pairs)),'xy_separated_triples':sep_count,'near_collinear_triples':len(linear),'near_rightangle_triples':len(right),'no_view_filter':True,'sampled_target_ids':row['sampled_target_ids'],'invalid_details':invalid,'rotated_target_ids':rotated,'raw_obb_details':details,'near_collinear_instance_triples':linear,'near_rightangle_instance_triples':right,'obb_size_instance_pairs':obb_pairs})
keys=['sampled_targets','valid_raw_obb_targets','rotated_raw_obb_targets','invalid_raw_obb_targets','obb_maxedge_ratio_pairs','aabb_maxedge_ratio_pairs_same_objects','obb_vs_aabb_size_pair_symmetric_difference','xy_separated_triples','near_collinear_triples','near_rightangle_triples']
result['totals']={k:sum(r[k] for r in result['scenes']) for k in keys};result['seconds']=round(time.perf_counter()-START,4)
result['inputs_unchanged']=all(sha(p)==h for p,h in result['inputs_sha256'].items())
Path('/tmp/interiorgs_qa_cpu_layout.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
with open('/tmp/interiorgs_qa_cpu_layout.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['scene_id']+keys+['no_view_filter'],extrasaction='ignore');w.writeheader();w.writerows(result['scenes'])
lines=['# 原始 OBB 与三物体构型：CPU 前置容量补充','','沿用 v3 审计抽样的20个train场景和336个目标ID；只读原始labels八顶点。未使用可见性筛选、未重跑渲染/可见性、未生成问答。以下是资产与构型前置候选，不能称合格图像或多图题。',f'CPU耗时 {result["seconds"]} 秒；全部输入SHA前后相同：{result["inputs_unchanged"]}。','','OBB检查：p0/p1/p3/p4构造三条边，重建8点误差≤1e-5m、边长>1e-5m、归一化点积绝对值≤1e-5；尺寸对按真实OBB最大边长比≥1.2。','三元组都要求中心XY两两距离≥0.5m。近共线：三个点各自到另外两点无限直线的距离最大值≤0.15m，最大两点跨度≥1m；近直角：至少一个顶点角80°–100°。这些是本次探查阈值，不冒充原benchmark采样标准。','','|场景|采样目标|有效OBB|旋转OBB|OBB尺寸对|AABB尺寸对（同集合）|两种尺寸对集合差异|XY间隔合格三元组|近共线|近直角|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in result['scenes']:lines.append('|'+r['scene_id']+'|'+'|'.join(str(r[k]) for k in ['sampled_targets','valid_raw_obb_targets','rotated_raw_obb_targets','obb_maxedge_ratio_pairs','aabb_maxedge_ratio_pairs_same_objects','obb_vs_aabb_size_pair_symmetric_difference','xy_separated_triples','near_collinear_triples','near_rightangle_triples'])+'|')
lines+=['|总计|'+'|'.join(str(result['totals'][k]) for k in ['sampled_targets','valid_raw_obb_targets','rotated_raw_obb_targets','obb_maxedge_ratio_pairs','aabb_maxedge_ratio_pairs_same_objects','obb_vs_aabb_size_pair_symmetric_difference','xy_separated_triples','near_collinear_triples','near_rightangle_triples'])+'|','','此处OBB尺寸对没有视图约束，不能与v3主审计246个可见AABB尺寸对直接比较。原始OBB轴不提供语义正面。近共线/直角组合未生成拍摄位姿，也没验证真多图必要性。五对象relative distance、表面最短距离和benchmark专用选择题干扰项仍未估。']
Path('/tmp/interiorgs_qa_cpu_layout.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'totals':result['totals'],'seconds':result['seconds'],'inputs_unchanged':result['inputs_unchanged']},ensure_ascii=False))
