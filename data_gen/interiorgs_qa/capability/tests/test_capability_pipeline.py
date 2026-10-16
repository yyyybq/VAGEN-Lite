import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from data_gen.interiorgs_qa.capability.pipeline import training_row, refs_unchanged, overlay, combine, write_rows
from data_gen.interiorgs_qa.capability.evidence import visible_references, wide_center_agrees
import numpy as np

class PipelineTests(unittest.TestCase):
    def test_private_fields_and_history_order_do_not_leak(self):
        q={'images':['/plain/first.png','/plain/second.png'],'question':'Where is the chair?',
           'options':['Left','Right'],'answer':'B','source':{'score':1,'future_image':'secret'},
           'evidence_zh':['secret answer B'],'human_review':'pending'}
        row=training_row(q)
        self.assertEqual(row['images'],q['images'])
        self.assertEqual(row['messages'][0]['content'].count('<image>'),2)
        self.assertNotIn('secret',str(row));self.assertEqual(row['messages'][1],{'role':'assistant','content':'B'})
        q['images']=['/inspection_wide_rgb.png']
        with self.assertRaises(ValueError):training_row(q)
    def test_changed_reference_is_rejected_not_rewritten(self):
        task={'target_object':{'objects':[{'id':'1','ref':'the chair'}]}}
        original=copy.deepcopy(task)
        with patch('data_gen.interiorgs_qa.capability.pipeline.start_refs',return_value={'1':{'ref':'the chair in bedroom'}}):
            self.assertFalse(refs_unchanged(None,task,None))
        self.assertEqual(task,original)
    def test_hidden_target_identifying_anchor_is_required(self):
        anchor=lambda i:{'id':i,'bbox_min':[0,0,0],'bbox_max':[1,1,1]}
        task={'task_type':'occlusion_alignment','target_object':{'objects':[{'id':'a','ref_anchor':anchor('ra')},{'id':'b','ref_anchor':anchor('rb')}]}}
        stat={'ra':{'visible':False},'rb':{'visible':True}}
        self.assertFalse(visible_references(task,stat))
        self.assertFalse(visible_references(task,stat,slots={0}))
        self.assertTrue(visible_references(task,stat,slots={1}))
    def test_wide_center_checks_actual_arrays_not_only_intrinsics(self):
        d=np.ones((256,256));w=np.ones((384,384))
        self.assertTrue(wide_center_agrees(d,d,w,w)[0])
        w[100,100]=2
        self.assertFalse(wide_center_agrees(d,d,w,np.ones_like(w))[0])

    def test_combine_preserves_render_paths_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);a=base/'a';b=base/'b';g=base/'g'
            for x in (a,b,g):x.mkdir()
            for x in (a,b):(x/'tasks.jsonl').write_text('{"task_id":"t"}\n')
            for x,sid,key in ((a,'s1','v1'),(b,'s2','v2')):
                write_rows(x/'states.jsonl',[{'state_id':sid}])
                write_rows(x/'render_manifest.jsonl',[{'view_key':key,'rgb':'/existing/'+key+'.png'}])
                write_rows((g if x==a else b)/'geometry.jsonl',[{'state_id':sid}])
            combine(a,b,g,base/'combined')
            import json
            records=[json.loads(r) for r in (base/'combined'/'render_manifest.jsonl').read_text().splitlines()]
            self.assertEqual([r['rgb'] for r in records],['/existing/v1.png','/existing/v2.png'])
            write_rows(b/'render_manifest.jsonl',[{'view_key':'v1','rgb':'/duplicate.png'}])
            with self.assertRaises(ValueError):combine(a,b,g,base/'invalid')
            self.assertFalse((base/'invalid').exists())

    def test_inspection_svg_does_not_modify_source(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'raw.png';Image.new('RGB',(256,256)).save(p);before=p.read_bytes()
            out=Path(d)/'overlay.svg'
            overlay(p,{'objects':[{'role':'a','projection':{'bbox_raw':[-20,5,80,120]}}]},out)
            self.assertEqual(before,p.read_bytes());self.assertIn('<rect',out.read_text())

if __name__=='__main__':unittest.main()
