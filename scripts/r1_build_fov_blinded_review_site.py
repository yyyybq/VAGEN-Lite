#!/usr/bin/env python3
"""Package the frozen 105 cases as a portable human annotation page.

Images are copied byte-for-byte with neutral filenames. Predictions and
historical labels never enter the reviewer page. No annotation is generated.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


PAGE = r'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>FOV blinded review</title><style>
body{font:16px system-ui;max-width:1200px;margin:24px auto;padding:0 16px}
button,input,select{font:inherit;padding:8px;margin:4px} .frames{display:flex;flex-wrap:wrap;gap:12px}
figure{margin:0} img{width:256px;height:256px;image-rendering:auto} .bar{position:sticky;top:0;background:white;padding:8px;border-bottom:1px solid #aaa}
textarea{width:95%;height:64px} #warning{color:#b22}</style>
<h1>FOV RGB 人工盲审：105 cases</h1>
<p>只判断：实际 RGB 中两个指定目标及 FOV 入框关系是否足够可辨认，使任务具有视觉可解性。
检查初态、末态和路径；目标过小、遮挡或低信息导致不可辨认时记 FAIL。不判断 canonical success、动作最优性或程序结果。</p>
<p>PASS：视觉可解；FAIL：至少一个目标/关系不可判断；UNCERTAIN：无法确定。此页面不提供任何默认标签。
图像为原官方 renderer 输出，未添加评分/预测。提交需要真实人工审阅。</p>
<label>Reviewer ID <input id="reviewer" placeholder="输入人工审阅者身份"></label>
<button id="export">导出 annotation CSV</button><button id="jsonExport">导出 JSON 备份</button>
<label>恢复 JSON 备份 <input id="import" type="file" accept="application/json"></label>
<div class="bar"><button id="prev">上一条</button><select id="case"></select><button id="next">下一条</button><span id="progress"></span></div>
<p id="warning"></p><h2 id="title"></h2><p id="targets"></p><div class="frames" id="frames"></div>
<p><button data-label="PASS">PASS</button><button data-label="FAIL">FAIL</button><button data-label="UNCERTAIN">UNCERTAIN</button><button data-label="">清空</button><b id="label"></b></p>
<textarea id="reason" placeholder="可选原因：目标不可辨认、严重遮挡、过小、错帧等"></textarea>
<script id="data" type="application/json">__DATA__</script>
<script>
'use strict';
const cases=JSON.parse(document.getElementById('data').textContent), key='r1_fov_blinded_review_105_v1';
const el=id=>document.getElementById(id);let saved={};
try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){el('warning').textContent='浏览器本地保存不可用，请定期导出 JSON 备份。'}
let index=0;const store=()=>{try{localStorage.setItem(key,JSON.stringify(saved))}catch(e){el('warning').textContent='请立即导出 JSON 备份。'}};
for(const [i,c] of cases.entries()){const o=document.createElement('option');o.value=i;o.textContent=c.case_id;el('case').append(o)}
function draw(){const c=cases[index],a=saved[c.case_id]||{};el('case').value=index;el('title').textContent=c.case_id;
el('targets').textContent='Scene: '+c.scene_id+'；目标：'+c.target_labels.join(' / ');el('label').textContent=a.verdict||'尚未标注';
el('reason').value=a.optional_reason||'';el('frames').replaceChildren();
for(const [i,f] of c.frames.entries()){const fig=document.createElement('figure'),im=document.createElement('img'),cap=document.createElement('figcaption');im.src=f;im.alt=c.case_id+' frame '+i;cap.textContent=i===0?'Initial':i===c.frames.length-1?'Terminal':'Frame '+i;fig.append(im,cap);el('frames').append(fig)}
el('progress').textContent=cases.filter(c=>saved[c.case_id]?.verdict).length+' / '+cases.length+' 已标注';}
el('case').onchange=()=>{index=Number(el('case').value);draw()};el('prev').onclick=()=>{index=Math.max(0,index-1);draw()};el('next').onclick=()=>{index=Math.min(cases.length-1,index+1);draw()};
document.querySelectorAll('[data-label]').forEach(b=>b.onclick=()=>{if(!el('reviewer').value.trim()){el('warning').textContent='请先填写人工 Reviewer ID。';return}
saved[cases[index].case_id]={case_id:cases[index].case_id,verdict:b.dataset.label,optional_reason:el('reason').value,reviewer_id:el('reviewer').value.trim(),reviewed_utc:new Date().toISOString()};el('warning').textContent='';store();draw()});
el('reason').onchange=()=>{const a=saved[cases[index].case_id];if(a){a.optional_reason=el('reason').value;store()}};
function download(name,text,type){const url=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
el('export').onclick=()=>{const cols=['case_id','verdict','optional_reason','reviewer_id','reviewed_utc'],quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';const rows=[cols.join(','),...cases.map(c=>cols.map(k=>quote((saved[c.case_id]||{case_id:c.case_id})[k])).join(','))];download('annotations_human.csv',rows.join('\r\n')+'\r\n','text/csv;charset=utf-8')};
el('jsonExport').onclick=()=>download('annotations_backup.json',JSON.stringify(saved,null,2),'application/json');
el('import').onchange=async()=>{const f=el('import').files[0];if(!f)return;try{const x=JSON.parse(await f.text());for(const [k,v] of Object.entries(x)){if(!cases.some(c=>c.case_id===k)||!['PASS','FAIL','UNCERTAIN',''].includes(v.verdict))throw Error('invalid annotation')}saved=x;store();draw()}catch(e){el('warning').textContent='备份无效：'+e.message}};
draw();</script></html>'''


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pack', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    blinded = rows(a.pack / 'review_manifest_blinded.jsonl')
    internal = {r['case_id']: r for r in rows(a.pack / 'case_metadata_internal.jsonl')}
    assert len(blinded) == len(internal) == 105
    a.output.mkdir(parents=True, exist_ok=False)
    cases, provenance = [], []
    for case in blinded:
        case_id = case['case_id']
        meta = internal[case_id]
        image_root = Path(meta['image_root'])
        manifest = image_root.parent.parent / 'observability_manifest.jsonl'
        if manifest.is_file():
            match = [r for r in rows(manifest) if r['task_id'] == meta['task_id']]
            assert len(match) == 1
            labels = [o['label'] for o in match[0]['frames'][0]['objects']]
        else:
            pilot = a.pack.parent / 'final_pilot_min4_visible_runtime_v3/fov_v2/r1_fov_repair_pilot_h1_v2_layout_gated.jsonl'
            match = [r for r in rows(pilot) if r['task_id'] == meta['task_id']]
            assert len(match) == 1
            labels = [o['label'] for o in match[0]['target_object']['objects']]
        assert len(labels) == 2
        paths = []
        for i, original in enumerate(case['review_frames']):
            source = Path(original)
            relative = Path('images') / case_id / f'{i:02d}.png'
            target = a.output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            assert sha(source) == sha(target)
            paths.append(str(relative))
            provenance.append({'case_id': case_id, 'frame_index': i, 'source': str(source), 'review_image': str(relative), 'sha256': sha(source)})
        cases.append({'case_id': case_id, 'scene_id': meta['scene_id'], 'target_labels': labels, 'frames': paths})
    data = json.dumps(cases, ensure_ascii=False).replace('<', '\\u003c')
    (a.output/'index.html').write_text(PAGE.replace('__DATA__', data))
    (a.output/'review_manifest_blinded.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases))
    shutil.copyfile(a.pack/'annotations_template.csv', a.output/'annotations_template.csv')
    # Keep evidence paths outside reviewer data. No predictions or labels even here.
    (a.output/'image_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    status={'cases':105,'frames':len(provenance),'human_labels_present':False,'gate_v3_frozen':False,
            'source_manifest_sha256':sha(a.pack/'review_manifest_blinded.jsonl'),
            'required_action':'Human reviewer opens index.html, labels 105 cases, exports annotations_human.csv.',
            'blind_fields':['no model prediction','no gate scores','no failure reasons','no historical labels'],'images':'byte-identical official RGB'}
    (a.output/'annotation_status.json').write_text(json.dumps(status,indent=2)+'\n')
    (a.output/'SHA256SUMS').write_text(''.join(f'{sha(f)}  {f.relative_to(a.output)}\n' for f in sorted(a.output.rglob('*')) if f.is_file() and f.name!='SHA256SUMS'))
    print(json.dumps(status,indent=2))


if __name__=='__main__':
    main()
