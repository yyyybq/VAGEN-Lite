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
