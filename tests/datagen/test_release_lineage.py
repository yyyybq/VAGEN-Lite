import sys,json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import pytest
from active_spatial_release_contract import scene_partitions,load_registry,bind_derived
from prepare_active_spatial_sft import qa_record

def test_alias_leak_is_rejected():
 with pytest.raises(ValueError,match='physical'):
  scene_partitions({'train_scenes':['a'],'test_scenes':['alias'],'aliases':{'a':'p','alias':'p'}})
def test_reserved_relabel_is_rejected():
 with pytest.raises(ValueError,match='Reserved'):
  scene_partitions({'train_scenes':['a'],'reserved_scenes':{'a':'val'}})
def test_task_hash_and_explicit_split_are_binding(tmp_path):
 split={'train_scenes':['a'],'val_scenes':['b'],'test_scenes':['c']};versions={x:'v1' for x in ('camera_model_version','canonical_task_metric_version','action_protocol_version')}
 parent={'task_id':'t','source_task_sha256':'hash','scene_id':'c','split':'test','versions':versions};p=tmp_path/'registry';p.write_text(json.dumps(parent)+'\n');registry=load_registry(p,split);lookup=scene_partitions(split)
 row={'parent_task_id':'t','source_task_sha256':'hash','scene_id':'c','split':'test','source_versions':versions}
 assert bind_derived(row,'qa',registry,lookup)==parent
 for change in ({'source_task_sha256':'changed'},{'split':'train'},{'scene_id':'a'},{'parent_task_id':'fake'},{'source_versions':{}}):
  with pytest.raises(ValueError):bind_derived(row|change,'qa',registry,lookup)
def test_unverified_screen_occupancy_cannot_opt_in():
 assert qa_record({'split':'train','task_type':'screen_occupancy','qa_sft_allowed':True},Path('.'))==(None,'screen_occupancy_unverified')


def test_release_sft_requires_observation_review(tmp_path):
 from types import SimpleNamespace
 from PIL import Image
 from prepare_active_spatial_sft import prepare
 image=tmp_path/'frame.png';Image.new('RGB',(8,8),'red').save(image)
 model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{"model_type":"qwen2_5_vl"}')
 versions={k:'fixture_v1' for k in ('camera_model_version','canonical_task_metric_version','action_protocol_version')}
 records=[];registry=[]
 for scene,split,review in [('a','train',True),('b','val',True),('c','train',False)]:
  parent='task_'+scene;digest='fixture_hash_'+scene
  registry.append({'task_id':parent,'source_task_sha256':digest,'scene_id':scene,'split':split,'versions':versions})
  row={'id':'sft_'+parent,'source_task_id':parent,'source_task_sha256':digest,'source_versions':versions,'scene_id':scene,'split':split,'success':True,'score_contract':{'success_source':'canonical_gates'},'image_paths':[str(image)],'conversations':[{'role':'user','content':'<image> Move left','image_path':str(image)},{'role':'assistant','content':'<action>move_left|</action>'}]}
  if review:row['semantic_review']={'status':'PASS'}
  records.append(row)
 source=tmp_path/'sft.jsonl';source.write_text(''.join(json.dumps(x)+'\n' for x in records))
 reg=tmp_path/'registry.jsonl';reg.write_text(''.join(json.dumps(x)+'\n' for x in registry))
 split=tmp_path/'split.json';split.write_text(json.dumps({'version':2,'train_scenes':['a','c'],'val_scenes':['b'],'test_scenes':[]}))
 args=SimpleNamespace(out=str(tmp_path/'out'),model=str(model),trajectory=[str(source)],qa_bank=[],qa_probability=0, val_fraction=.1,seed=42,split_manifest=str(split),task_registry=str(reg),cutoff_len=2048)
 result=prepare(args)
 assert result['counts']['trajectory/train']==1 and result['counts']['trajectory/val']==1
 assert result['excluded']['trajectory_semantic_review_missing']==1
