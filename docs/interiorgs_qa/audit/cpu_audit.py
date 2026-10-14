#!/usr/bin/env python3
"""One-off read-only CPU metadata/candidate audit. No rendering, QA or training-data generation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,json,time,hashlib,random,itertools,collections,math,subprocess,traceback
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/umm/users/yinbaiqiao/VAGEN-Lite')
RUN=ROOT/'exps/vagen_active_spatial/interiorgs_v10_gen_20261014'
SRC=Path('/tmp/interiorgs_qa_cpu_v3_snapshot')
META=Path('/mnt/umm/users/yinbaiqiao/InteriorGS_meta')
sys.path.insert(0,str(SRC))
sys.path.insert(1,str(ROOT))
from data_gen.interiorgs_v10.scene import load_scene
from data_gen.interiorgs_v10.objects import candidate_objects
from data_gen.interiorgs_v10.contract import pose_from_xy_yaw
from data_gen.interiorgs_v10.referring import display_label,infer_room_type
from data_gen.interiorgs_v10.occlusion import canonical_intrinsics
from vagen.envs.active_spatial import target_visibility as TV
assert TV.VISIBILITY_RULE_VERSION=='face_visible_v3', TV.__file__
SEED=20261014
GRID=0.8
CAP_GRID=2000
CAP_POS=24
CAP_OBJS=24
START=time.perf_counter()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(d):Path('/tmp/interiorgs_qa_cpu_v3.json').write_text(json.dumps(d,ensure_ascii=False,indent=2))
def number(x):return int(x)
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(str(type(x)))
split=json.load((RUN/'split_manifest.json').open())
selected=sorted(random.Random(SEED).sample(sorted(split['train_scenes']),20))
Path('/tmp/interiorgs_qa_cpu_v3_scenes.txt').write_text('\n'.join(selected)+'\n')
source_files={}
for name,m in list(sys.modules.items()):
 p=getattr(m,'__file__',None)
 if p and (name.startswith('data_gen.interiorgs_v10') or name.startswith('vagen.envs.active_spatial')):
  assert str(Path(p)).startswith(str(SRC)),(name,p)
  if p.endswith('.py'):source_files[p]=sha(p)
# Hash all geometry source files before any late imports, including parser dependencies.
for folder in [SRC/'data_gen/interiorgs_v10',SRC/'vagen/envs']:
 for f in folder.rglob('*.py'):source_files[str(f)]=sha(f)
source_files[str(RUN/'split_manifest.json')]=sha(RUN/'split_manifest.json')
source_files[__file__]=sha(__file__)
no_task_path=SRC/'data_gen/interiorgs_v10/configs/no_task_objects_20261013.json'
source_files[str(no_task_path)]=sha(no_task_path)
no_task_data=json.load(no_task_path.open()).get('scenes',{})
result={'audit_kind':'read_only_cpu_geometric_capacity_not_QA_generation','seed':SEED,'selected_train_scenes':selected,'selection_method':'sorted(random.Random(20261014).sample(sorted(r5 split train_scenes),20)); no task/quality filtering','metadata_root':str(META),'r5_run':str(RUN),'source_snapshot':str(SRC),'r5_recorded_commit':(RUN/'code_commit.txt').read_text().strip(),'current_repository_HEAD_at_start':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256_at_start':source_files,'analysis_source_commit':'3a964c795f82e4014572d7dc61425cd22d7816f3','visibility_rule':TV.VISIBILITY_RULE_VERSION,'sampling':{'grid_spacing_m':GRID,'max_raw_grid_points_per_scene':CAP_GRID,'max_legal_positions_per_scene':CAP_POS,'yaw_degrees':[0,90,180,270],'max_ref_targets_per_scene':CAP_OBJS,'position_height_m':1.5,'wall_clearance_m':0.5,'camera_radius_m':0.15,'safety_margin_m':0.05,'target_selection':'uniform seeded sample if >24 eligible targets; all otherwise','position_selection':'uniform seeded sample of eligible fixed-grid positions if >24; all otherwise','face_visible_fraction':TV.MIN_FACE_VISIBLE_FRACTION,'face_image_area_fraction':TV.MIN_FACE_AREA_FRACTION,'face_grid':TV.FACE_SAMPLES_PER_AXIS,'overlap_margin_m':TV.OVERLAP_MARGIN_M,'lateral_pixel_margin':12,'lateral_max_angle_deg':45,'vertical_pixel_margin':12,'depth_gap_m':0.5,'pair_min_center_separation_m':0.5,'size_ratio_min':1.2,'triple_xy_area_min_m2':0.1,'multiview_min_translation_m':0.5},'schema':{},'scenes':[]}
# Full 1,000-scene schema census. Does not instantiate geometry or read splats.
all_scenes=sorted(set(split['train_scenes']+split['val_scenes']+split['test_scenes']))
counts=collections.Counter(); label_keys=collections.Counter(); room_keys=collections.Counter(); wall_keys=collections.Counter(); hole_keys=collections.Counter(); struct_keys=collections.Counter(); occ_keys=collections.Counter(); category_counts=collections.Counter(); examples={}; metadata_hash=hashlib.sha256()
for sid in all_scenes:
 for name in ['labels.json','structure.json','occupancy.json']:
  p=META/sid/name
  if not p.exists():counts['missing_'+name]+=1;continue
  raw=p.read_bytes();metadata_hash.update(sid.encode()+name.encode()+hashlib.sha256(raw).digest())
  d=json.loads(raw);counts['scenes_with_'+name]+=1
  if name=='labels.json':
   counts['label_records']+=len(d)
   for o in d:
    label_keys.update(o.keys());category_counts[o.get('label','')]+=1
    bb=o.get('bounding_box') or [];counts['bbox_length_'+str(len(bb))]+=1
    if len(bb)==8:
     xyz=np.array([[q['x'],q['y'],q['z']] for q in bb],dtype=float)
     lo=xyz.min(0);hi=xyz.max(0)
     aabb=bool(np.all(np.minimum(np.abs(xyz-lo),np.abs(xyz-hi))<=1e-6))
     counts['eight_corner_axis_aligned' if aabb else 'eight_corner_non_axis_aligned']+=1
     e1,e2,e3=xyz[1]-xyz[0],xyz[3]-xyz[0],xyz[4]-xyz[0]
     pred=np.array([xyz[0],xyz[0]+e1,xyz[0]+e1+e2,xyz[0]+e2,xyz[0]+e3,xyz[0]+e1+e3,xyz[0]+e1+e2+e3,xyz[0]+e2+e3])
     order_ok=bool(np.max(np.abs(pred-xyz))<=1e-5)
     lens=np.array([np.linalg.norm(e) for e in [e1,e2,e3]])
     nondeg=bool(lens.min()>1e-5)
     orth=bool(nondeg and max(abs(float(np.dot(a,b)))/(np.linalg.norm(a)*np.linalg.norm(b)) for a,b in [(e1,e2),(e1,e3),(e2,e3)])<=1e-5)
     counts['ordered_box_corner_pattern_valid' if order_ok else 'ordered_box_corner_pattern_invalid']+=1
     counts['orthogonal_box_edges' if orth else 'nonorthogonal_or_degenerate_box_edges']+=1
     if order_ok and orth:counts['recoverable_geometric_OBB']+=1
     if not aabb and order_ok and orth:counts['recoverable_rotated_OBB']+=1
     if not aabb and 'non_axis_aligned_box' not in examples:examples['non_axis_aligned_box']={'scene':sid,'id':o.get('ins_id'),'bbox':bb}
    for k in ['orientation','rotation','heading','front','front_direction','semantic_front','quaternion','room_id','room_type']:
     if k in o:counts['label_field_'+k]+=1
  elif name=='structure.json':
   struct_keys.update(d.keys())
   for key,kc in [('rooms',room_keys),('walls',wall_keys),('holes',hole_keys)]:
    rr=d.get(key,[]);counts[key]+=len(rr)
    if rr:counts['scenes_with_nonempty_'+key]+=1
    for o in rr:
     kc.update(o.keys())
     if key=='rooms' and any(o.get(k) for k in ['name','type','room_type']):counts['rooms_with_semantic_name_or_type']+=1
     if key=='holes':counts['hole_type_'+str(o.get('type'))]+=1
  elif name=='occupancy.json':occ_keys.update(d.keys())
result['schema']={'scene_count':len(all_scenes),'counts':dict(counts),'label_key_occurrences':dict(label_keys),'structure_key_scene_occurrences':dict(struct_keys),'room_key_occurrences':dict(room_keys),'wall_key_occurrences':dict(wall_keys),'hole_key_occurrences':dict(hole_keys),'occupancy_key_scene_occurrences':dict(occ_keys),'label_category_count':len(category_counts),'top30_categories':category_counts.most_common(30),'examples':examples,'metadata_manifest_sha256':metadata_hash.hexdigest(),'seconds':round(time.perf_counter()-START,3)}
print('schema_complete',result['schema']['counts'],flush=True);dump(result)
# Existing R5 trajectory metadata, no replay / new trajectories.
r5_tasks={s:[] for s in selected}
with (RUN/'output/tasks.jsonl').open() as f:
 for line in f:
  row=json.loads(line)
  if row['scene_id'] in r5_tasks:r5_tasks[row['scene_id']].append(row)
for sid in selected:
 t=time.perf_counter();out={'scene_id':sid}
 try:
  rng=random.Random(str(SEED)+':'+sid)
  nt=[str(x['ins_id']) for x in no_task_data.get(sid,[])]
  scene=load_scene(META/sid,sid,no_task_ids=nt)
  objs,funnel=candidate_objects(scene,'v10_r5')
  rawcats=collections.Counter(display_label(o.label) for o in scene.objects)
  cats=collections.Counter(display_label(o.label) for o in objs)
  room_areas=[abs(float(np.dot(r.polygon[:,0],np.roll(r.polygon[:,1],-1))-np.dot(r.polygon[:,1],np.roll(r.polygon[:,0],-1))))/2 for r in scene.rooms]
  roomtypes={str(r.index):infer_room_type([display_label(o.label) for o in scene.objects if o.room_index==r.index]) for r in scene.rooms}
  pp=list(itertools.combinations(objs,2));metric_pairs=[(a,b) for a,b in pp if np.linalg.norm(a.center-b.center)>=0.5]
  size_pairs=[(a,b) for a,b in pp if max(a.dims.max(),b.dims.max())>=1.2*min(a.dims.max(),b.dims.max())]
  rows=r5_tasks[sid]
  out.update(raw_bbox_objects=len(scene.objects),r5_no_task_objects=len(scene.no_task_object_ids),eligible_ref_targets=len(objs),eligible_categories=len(cats),ref_methods=dict(collections.Counter(o.ref_method for o in objs)),ref_anchors=sum(bool((o.ref_extra or {}).get('ref_anchor')) for o in objs),raw_duplicate_categories=sum(v>=2 for v in rawcats.values()),eligible_duplicate_categories=sum(v>=2 for v in cats.values()),metric_pair_structures=len(metric_pairs),aabb_size_comparison_structures=len(size_pairs),room_polygons=len(scene.rooms),room_areas_m2=room_areas,inferred_room_types=roomtypes,inferred_typed_rooms=sum(v is not None for v in roomtypes.values()),coordinate_status=scene.convention_v2.get('status'),coordinate_usable=scene.convention_v2.get('usable'),object_funnel=funnel.to_dict(),existing_r5_tasks=len(rows),existing_r5_trajectory_ge2=sum(r['difficulty']['d_star']>=2 for r in rows),existing_r5_trajectory_ge4=sum(r['difficulty']['d_star']>=4 for r in rows),existing_r5_max_steps=max([r['difficulty']['d_star'] for r in rows],default=0))
  if not scene.convention_v2.get('usable',False):out['sampling_status']='not_estimated_coordinate_unusable';result['scenes'].append(out);dump(result);continue
  # Bounded per-room fixed lattice; cap before legality and report truncation.
  grid=[]
  for room in scene.rooms:
   lo=room.polygon.min(0);hi=room.polygon.max(0)
   for x in np.arange(math.ceil(lo[0]/GRID)*GRID,hi[0]+1e-8,GRID):
    for y in np.arange(math.ceil(lo[1]/GRID)*GRID,hi[1]+1e-8,GRID):grid.append((round(float(x),6),round(float(y),6)))
  grid=sorted(set(grid));out['raw_grid_points']=len(grid)
  if len(grid)>CAP_GRID:grid=sorted(rng.sample(grid,CAP_GRID))
  g=np.asarray(grid).reshape(-1,2);mask=scene.legal_mask(g);legal=g[mask]
  out.update(grid_points_checked=len(grid),legal_grid_points_found=len(legal),grid_was_capped=out['raw_grid_points']>CAP_GRID)
  if len(legal)>CAP_POS:legal=legal[sorted(rng.sample(range(len(legal)),CAP_POS))]
  sampled=sorted(objs,key=lambda x:x.id)
  if len(sampled)>CAP_OBJS:sampled=sorted(rng.sample(sampled,CAP_OBJS),key=lambda x:x.id)
  out.update(sampled_positions=len(legal),sampled_targets=len(sampled),sampled_target_ids=[o.id for o in sampled],sampled_positions_xy=legal.tolist())
  poses=np.array([pose_from_xy_yaw(x,y,yaw) for x,y in legal for yaw in [0,math.pi/2,math.pi,3*math.pi/2]])
  n=len(poses);m=len(sampled);out['sampled_views']=n
  if n==0 or m==0:out['sampling_status']='no_legal_positions_or_targets';result['scenes'].append(out);dump(result);continue
  R=poses[:,:3,:3];C=poses[:,:3,3];K=canonical_intrinsics();occ=TV.OccluderSet.from_detector(scene.detector)
  vis_cache={}
  def visible(entry):
   oid=str(entry['id'])
   if oid in vis_cache:return vis_cache[oid]
   geom=TV.FaceGeometry(entry['bbox_min'],entry['bbox_max'])
   # Occlusion depends on camera position, not yaw; evaluate only 24 origins then repeat.
   blocked=TV.blocked_mask(occ,C[::4],geom.points,geom.lo,geom.hi,exclude_ids=(oid,))
   stats=TV.face_visibility(R,C,geom,np.repeat(blocked,4,axis=0),K,256,256)
   vis_cache[oid]=stats['visible'];return vis_cache[oid]
  V=np.zeros((n,m),bool);centers=np.array([o.center for o in sampled]);dims=np.array([o.dims for o in sampled]);camera=np.empty((n,m,3));uv=np.empty((n,m,2))
  for j,o in enumerate(sampled):
   e={'id':o.id,'bbox_min':o.aabb_min.tolist(),'bbox_max':o.aabb_max.tolist()}
   V[:,j]=visible(e)
   anchor=(o.ref_extra or {}).get('ref_anchor')
   if anchor:V[:,j]&=visible(anchor)
  for k,p in enumerate(poses):
   camera[k]=(centers-p[:3,3])@p[:3,:3]
   with np.errstate(divide='ignore',invalid='ignore'):uv[k]=camera[k,:,:2]/camera[k,:,2,None]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
  s=collections.Counter();unique_pairs=set();unique_lr=set();unique_depth=set();unique_size=set();unique_trip=set();pair_signs=collections.defaultdict(set)
  selected_ids={o.id for o in sampled}
  complete_categories={label for label in cats if {o.id for o in scene.objects if display_label(o.label)==label}<=selected_ids}
  for k in range(n):
   ids=np.flatnonzero(V[k]);s['visible_object_view_slots']+=len(ids);s['views_with_ge2_targets']+=int(len(ids)>=2);s['views_with_ge3_targets']+=int(len(ids)>=3)
   for a,b in itertools.combinations(ids,2):
    key=tuple(sorted([sampled[a].id,sampled[b].id]));sep=float(np.linalg.norm(centers[a]-centers[b]));dx=abs(camera[k,a,0]-camera[k,b,0]);dz=abs(camera[k,a,2]-camera[k,b,2]);pix=float(uv[k,a,0]-uv[k,b,0])
    unique_pairs.add(key);s['covisible_pair_view_slots']+=1
    if abs(pix)>=12 and dx>=dz and sep>=0.5:unique_lr.add(key);s['lateral_pair_view_slots']+=1;pair_signs[key].add(1 if pix>0 else -1)
    if abs(float(uv[k,a,1]-uv[k,b,1]))>=12 and abs(centers[a,2]-centers[b,2])>=0.2:s['vertical_pair_view_slots']+=1
    if dz>=0.5 and dz>dx:unique_depth.add(key);s['depth_order_pair_view_slots']+=1
    if sep>=0.5:s['metric_object_pair_view_slots']+=1
    if max(dims[a].max(),dims[b].max())>=1.2*min(dims[a].max(),dims[b].max()):unique_size.add(key);s['aabb_size_pair_view_slots']+=1
   for a,b,c in itertools.combinations(ids,3):
    aa,bb,cc=centers[[a,b,c],:2];area=abs(float(np.cross(bb-aa,cc-aa)))/2
    if area>=0.1 and min(np.linalg.norm(aa-bb),np.linalg.norm(aa-cc),np.linalg.norm(bb-cc))>=0.5:unique_trip.add(tuple(sorted([sampled[a].id,sampled[b].id,sampled[c].id])));s['noncollinear_triple_view_slots']+=1
   for label in complete_categories:
    group=[j for j,o in enumerate(sampled) if display_label(o.label)==label]
    if len(group)>=2 and V[k,group].all():s['full_scene_category_count_view_slots']+=1
  multiview_objects=0;mv_pose_pairs=0;mv_anchor_pose_pairs=0
  for j in range(m):
   ii=np.flatnonzero(V[:,j]);posset={int(i)//4 for i in ii}
   if any(np.linalg.norm(legal[a]-legal[b])>=0.5 for a,b in itertools.combinations(posset,2)):multiview_objects+=1
  for a,b in itertools.combinations(range(n),2):
   if np.linalg.norm(C[a]-C[b])<0.5:continue
   common=int((V[a]&V[b]).sum())
   if common>=1:mv_pose_pairs+=1
   if common>=3:mv_anchor_pose_pairs+=1
  defaults=['visible_object_view_slots','views_with_ge2_targets','views_with_ge3_targets','covisible_pair_view_slots','lateral_pair_view_slots','vertical_pair_view_slots','depth_order_pair_view_slots','metric_object_pair_view_slots','aabb_size_pair_view_slots','noncollinear_triple_view_slots','full_scene_category_count_view_slots']
  for key in defaults:s.setdefault(key,0)
  out.update(sampling_status='ok',sampled_object_fraction=(m/len(objs) if objs else 0),visible_unique_targets=int(V.any(axis=0).sum()),covisible_unique_pairs=len(unique_pairs),lateral_unique_pairs=len(unique_lr),depth_order_unique_pairs=len(unique_depth),aabb_size_unique_pairs=len(unique_size),noncollinear_unique_triples=len(unique_trip),object_seen_from_ge2_positions=multiview_objects,multiview_camera_pairs_ge1_anchor=mv_pose_pairs,multiview_camera_pairs_ge3_anchors=mv_anchor_pose_pairs,lateral_relation_flip_pairs=sum(len(v)==2 for v in pair_signs.values()),complete_category_count_categories=len(complete_categories),view_counts=dict(s),unestimated={'intrinsic_object_front_and_orientation':'not estimated: no semantic front metadata; AABB axes do not establish facing','video_route_QA':'not estimated: only existing R5 path lengths counted; no route/temporal observation coverage evaluated','appearance_order_QA':'not estimated: requires full temporal visibility sequence and identity audit','room_type_ground_truth':'not available: inferred_room_types are furniture heuristics','render_or_RGB_quality':'not estimated: CPU only','absolute_metric_scale_identifiability':'not estimated: geometry exists but RGB alone may not provide scale'})
 except Exception as exc:out['error']=type(exc).__name__+': '+str(exc);out['traceback']=traceback.format_exc()
 out['seconds']=round(time.perf_counter()-t,3);result['scenes'].append(out);dump(result)
 print('scene_complete',sid,'refs',out.get('eligible_ref_targets'),'views',out.get('sampled_views'),'visible',out.get('visible_unique_targets'),'seconds',out['seconds'],flush=True)
result['elapsed_seconds']=round(time.perf_counter()-START,3)
result['source_sha256_at_end']={p:sha(p) for p in source_files}
result['source_snapshot_unchanged']=result['source_sha256_at_end']==source_files
result['schema_caveats']=['Raw eight-corner boxes include rotated OBBs; corner-pattern/orthogonality census reports recoverability. The reused v10 parser converts every box to AABB and loses that rotation. OBB axes still do not establish semantic front.','Room semantic types are absent unless counts explicitly show otherwise. Inferred types are heuristics, not annotated GT.','Occupancy JSON inspected only for schema; grid/image masks were not loaded.','Current commit snapshot uses face_visible_v3, while scene splits and existing trajectories remain the frozen r5 batch and object selection stays v10_r5. V3 includes structure-wall panels with door holes and curtains; visibility is still a box-based geometry proxy, not RGB identifiability.']
result['count_caveats']=['All counts are bounded CPU geometry candidates, not independent QA examples or validated images.','Counts reuse the same images/objects/pairs across capabilities and MUST NOT be summed into dataset size.','No successful question or answer text was created; candidate_objects was used only to audit existing referring-expression eligibility.','No scenes were replaced based on poor yield.','Existing R5 paths are short action certificates, not newly generated route-navigation/video questions.','Full-scene category count slots require every raw bounding-box instance in that category to be among sampled targets and geometrically visible; incomplete categories are not counted.']
result['metadata_files_per_scene']=json.load(open('/tmp/interiorgs_qa_metadata_files.json'))
result['capacity_layers']={'precise_within_sampling':['visible_unique_targets','lateral_unique_pairs','depth_order_unique_pairs','aabb_size_unique_pairs','noncollinear_unique_triples','view_counts'], 'structural_upper_bounds_only':['metric_pair_structures','aabb_size_comparison_structures','room_polygons','inferred_typed_rooms','object_seen_from_ge2_positions','multiview_camera_pairs_ge1_anchor','multiview_camera_pairs_ge3_anchors','lateral_relation_flip_pairs','existing_r5_tasks','existing_r5_trajectory_ge2','existing_r5_trajectory_ge4'], 'not_estimated':['true_multiview_necessity','video_route_QA','appearance_order_QA','intrinsic_object_semantic_front','absolute_metric_scale_identifiability','RGB_identifiability','pixel_masks_or_depth_GT']}
result['totals']={k:sum(o.get(k,0) for o in result['scenes']) for k in ['raw_bbox_objects','eligible_ref_targets','sampled_positions','sampled_targets','sampled_views','visible_unique_targets','lateral_unique_pairs','depth_order_unique_pairs','aabb_size_unique_pairs','noncollinear_unique_triples','object_seen_from_ge2_positions','multiview_camera_pairs_ge3_anchors','existing_r5_trajectory_ge2','existing_r5_trajectory_ge4','room_polygons']}
result['total_view_counts']=dict(sum((collections.Counter(o.get('view_counts',{})) for o in result['scenes']),collections.Counter()))
import csv
flat=[]
for o in result['scenes']:
 flat.append({**{k:v for k,v in o.items() if isinstance(v,(str,int,float,bool)) or v is None},**{'view_'+k:v for k,v in o.get('view_counts',{}).items()}})
with open('/tmp/interiorgs_qa_cpu_v3.csv','w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for o in flat for k in o)));writer.writeheader();writer.writerows(flat)
dump(result)
lines=['# InteriorGS CPU 空间 QA 候选容量审计','','只读 CPU 元数据与有限几何候选普查；没有渲染、没有生成问答、没有修改生产。',f'总耗时 {result["elapsed_seconds"]} 秒；冻结源码 {SRC}，记录提交 {result["r5_recorded_commit"]}；分析源码提交 {result["analysis_source_commit"]}；可见性 {TV.VISIBILITY_RULE_VERSION}；源文件前后 SHA 一致：{result["source_snapshot_unchanged"]}。','','## 选择与采样',result['selection_method'],f'每场景 0.8 m 网格最多检查 {CAP_GRID} 点，最多 {CAP_POS} 合法位置 × 4 固定朝向；目标最多 {CAP_OBJS}，使用 v10_r5 指代和 no-task 清单。要求目标及 ref_anchor 都通过 face_visible_v3；v10_r5 指代和 r5 split 保持。','位置 z=1.5 m、距墙≥0.5 m、相机半径0.15 m、安全余量0.05 m；可见面比例≥0.4、面积≥图像0.003；关系阈值及源文件 SHA 详见 JSON。','','## 全量元数据 schema','```json',json.dumps(result['schema'],ensure_ascii=False,indent=2),'```','','## 20 个 train 场景（每项不是 QA 成品数量）','','|场景|原始框|可唯一指代目标|房间/推断有类型|合法网格点|采样视图|可见目标|左右独立对|深度独立对|尺寸独立对|非共线三元组|多位置可见目标|≥3共享目标的视图对|现存≥2步路径|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for o in result['scenes']:
 def v(k):return o.get(k,'未估')
 lines.append(f'|{o["scene_id"]}|{v("raw_bbox_objects")}|{v("eligible_ref_targets")}|{v("room_polygons")}/{v("inferred_typed_rooms")}|{v("legal_grid_points_found")}|{v("sampled_views")}|{v("visible_unique_targets")}|{v("lateral_unique_pairs")}|{v("depth_order_unique_pairs")}|{v("aabb_size_unique_pairs")}|{v("noncollinear_unique_triples")}|{v("object_seen_from_ge2_positions")}|{v("multiview_camera_pairs_ge3_anchors")}|{v("existing_r5_trajectory_ge2")}|')
lines += ['','## 单图能力的观测槽计数（同一个目标/对可多视角重复）','','|场景|可见物体槽|左右对槽|上下对槽|深度对槽|世界距离对槽|框尺寸对槽|完整类别计数槽|三物体槽|左右翻转独立对|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for o in result['scenes']:
 vc=o.get('view_counts',{})
 vals=[vc.get(k,'未估') for k in ['visible_object_view_slots','lateral_pair_view_slots','vertical_pair_view_slots','depth_order_pair_view_slots','metric_object_pair_view_slots','aabb_size_pair_view_slots','full_scene_category_count_view_slots','noncollinear_triple_view_slots']]
 lines.append('|'+o['scene_id']+'|'+'|'.join(map(str,vals))+'|'+str(o.get('lateral_relation_flip_pairs','未估'))+'|')
lines += ['','## 解释与未估','左右：中心投影间距≥12 px、地面中心线距相机横轴≤45°；深度：中心相机深度差≥0.5 m且大于横向差；尺寸：AABB最大边比≥1.2，不代表真实 OBB 边长。','三物体：同视图可见、XY 三角形面积≥0.1 m²、两两距≥0.5 m；这是点参考系结构，未证明对象有语义正面。','多图：相机位置差≥0.5 m，共享目标数按表统计；仅结构上界，未做真多图必要性筛选，没有验证 RGB 特征匹配、时序重识别或相机估计可解性。','房间区域：房间多边形面积在 JSON 中；房间名称仅家具规则推断，不能当真实语义标签。','视频路线/appearance order：未估；只报告已有 R5 动作证书长度，未生成视频路径或问答。','绝对距离：只证明元数据有米制几何，不证明模型从有限 RGB 能唯一回答。','']+result['count_caveats']+result['schema_caveats']
Path('/tmp/interiorgs_qa_cpu_v3.md').write_text('\n'.join(lines)+'\n')
print('DONE',result['elapsed_seconds'],result['source_snapshot_unchanged'],flush=True)
