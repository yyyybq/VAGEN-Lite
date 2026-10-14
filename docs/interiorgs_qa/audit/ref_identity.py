import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,hashlib
from pathlib import Path
src=Path(sys.argv[1]);sys.path.insert(0,str(src))
from data_gen.interiorgs_v10.scene import load_scene
from data_gen.interiorgs_v10.objects import candidate_objects
base=json.load(open('/tmp/interiorgs_qa_cpu_audit.json'))
nt=json.load(open(src/'data_gen/interiorgs_v10/configs/no_task_objects_20261013.json'))['scenes']
result={}
for sid in base['selected_train_scenes']:
 s=load_scene(Path(base['metadata_root'])/sid,sid,no_task_ids=[str(o['ins_id']) for o in nt.get(sid,[])])
 oo,_=candidate_objects(s,'v10_r5')
 rows=sorted([{'id':o.id,'ref':o.ref,'ref_method':o.ref_method,'ref_extra':o.ref_extra,'aabb_min':o.aabb_min.tolist(),'aabb_max':o.aabb_max.tolist()} for o in oo],key=lambda x:x['id'])
 result[sid]={'count':len(rows),'full_ref_geometry_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()}
print(json.dumps(result,indent=2))
