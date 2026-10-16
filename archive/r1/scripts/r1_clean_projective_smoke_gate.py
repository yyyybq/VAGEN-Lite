#!/usr/bin/env python3
"""Fail closed on missing PPO update, nonfinite metrics or incomplete checkpoint."""
import argparse
import json
import math
import re
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--frozen',type=Path,required=True);a=p.parse_args()
    smoke=a.run/'smoke';cfg=json.loads((a.frozen/'data_gate.json').read_text())
    reload=json.loads((smoke/'checkpoint_reload.json').read_text())
    assert reload['checkpoint_save_and_load']=='PASS'
    text=(smoke/'train.log').read_text(errors='replace')
    metrics={}
    # verl formats scalar metrics through ``np.float64(...)`` in this runtime.
    # Accept both that representation and ordinary numeric scalars; the old
    # expression silently skipped every wrapped actor/critic metric and made a
    # completed PPO smoke fail only at the post-run evidence gate.
    number=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf)'
    pattern=rf'([A-Za-z_][A-Za-z0-9_./-]+):\s*(?:np\.float(?:32|64)\()?({number})(?:\))?'
    for key,value in re.findall(pattern,text,re.IGNORECASE):
        metrics[key]=float(value)
    required=('trainer/actor_update_performed','actor/grad_norm','actor/pg_loss','critic/grad_norm')
    for key in required:
        assert key in metrics and math.isfinite(metrics[key]),(key,metrics.get(key))
    assert metrics['trainer/actor_update_performed']==1
    assert metrics['actor/grad_norm']>0 and metrics['critic/grad_norm']>0
    advantage={k:v for k,v in metrics.items() if 'advantages/' in k}
    assert advantage and all(math.isfinite(v) for v in advantage.values())
    checkpoint=smoke/'checkpoints/global_step_1'
    assert (checkpoint/'COMPLETE').is_file()
    # Compare actual tensors, not checkpoint/config filenames or timestamps.
    from safetensors import safe_open
    model=Path(cfg['model']['path']);hf=checkpoint/'actor/huggingface'
    exports=list((checkpoint/'actor').glob('**/model.safetensors.index.json'))
    assert len(exports)==1,('missing/ambiguous HF export',exports)
    hf=exports[0].parent
    oldmap=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    newmap=json.loads(exports[0].read_text())['weight_map']
    import torch
    changed=[];checked=[]
    for name in sorted(oldmap):
        if name not in newmap or 'layers.' not in name or not name.endswith('weight'):continue
        with safe_open(str(model/oldmap[name]),framework='pt',device='cpu') as x, safe_open(str(hf/newmap[name]),framework='pt',device='cpu') as y:
            before=x.get_tensor(name);after=y.get_tensor(name)
            delta=float((before.float()-after.float()).abs().max())
            assert math.isfinite(delta)
            checked.append({'parameter':name,'max_abs_delta':delta})
            if delta>0:changed.append(name)
        if len(checked)>=16:break
    assert changed,'no actual actor weight change detected'
    report={'status':'PASS','checkpoint_reload':reload,'metrics':metrics,'checked_parameters':checked,
            'formal_starts_from':str(model),'smoke_weights_reused':False}
    (smoke/'gate.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'smoke_gate':'PASS','changed_parameters':len(changed)}))

if __name__=='__main__':main()
