import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from PIL import Image
import pytest

from data_gen.interiorgs_qa.capability.review import build_site, question_sha256


def fixture(tmp_path):
    paths=[]
    for name,color in [("first","red"),("second","green"),("overlay","blue")]:
        path=tmp_path/(name+".png");Image.new("RGB",(10,10),color).save(path);paths.append(str(path))
    rows=[{"qid":"q1","group":"遮挡判断","task_type":"occlusion_alignment","scene_id":"scene_a",
           "parent_task_id":"formal_task_1","question":"In the second image, is the chair hidden?",
           "options":["fully hidden","partly hidden","visible","outside the image"],"answer":"A",
           "answer_key":"fully_hidden","images":paths[:2],"overlays":[None,paths[2]],
           "evidence_zh":["第二张图中，椅子仍在画面范围内。"],"source":{"formal":True},"human_review":"pending"},
          {"qid":"q2","group":"距离判断","task_type":"absolute_positioning","scene_id":"scene_a",
           "parent_task_id":"formal_task_2","question":"About how far is the chair? </script><script>bad()</script>",
           "options":["1 metre","2 metres"],"answer":"B","answer_key":"2",
           "images":[paths[0]],"overlays":[None],"evidence_zh":["相机到椅子中心的水平距离为2米。"],
           "source":{"formal":True},"human_review":"pending"}]
    path=tmp_path/"questions.jsonl";path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    return path,rows


def test_site_preserves_originals_and_binds_full_question(tmp_path):
    path,rows=fixture(tmp_path);originals={p:Path(p).read_bytes() for q in rows for p in q["images"]}
    out=tmp_path/"site";summary=build_site(path,out);data=json.loads((out/"data.json").read_text())
    assert summary["questions"]==2 and summary["image_inputs"]==3 and summary["independent_images"]==2
    assert summary["human_pending"]==2 and summary["published"] is False
    assert data["training_release_count"]==0 and data["model_results_included"] is False
    for q,row in zip(data["items"],rows):
        assert q["question_sha256"]==question_sha256(row)
        for old,new,digest in zip(row["images"],q["images"],q["image_sha256"]):
            assert (out/new).read_bytes()==originals[old]==Path(old).read_bytes()
            assert hashlib.sha256(originals[old]).hexdigest()==digest
        assert all(p.startswith("assets/image_") for p in q["images"])
        assert all(p is None or p.startswith("assets/overlay_") for p in q["overlays"])
    html=(out/"index.html").read_text();assert "</script><script>bad()" not in html
    assert "GT" not in data["items"][0]["question"]
    changed=dict(rows[0],options=list(reversed(rows[0]["options"])))
    assert question_sha256(changed)!=question_sha256(rows[0])
    with pytest.raises(ValueError):build_site(path,out)


def test_rejects_duplicate_ids_and_preexisting_human_approval(tmp_path):
    path,rows=fixture(tmp_path);path.write_text(json.dumps(rows[0])+"\n"+json.dumps(rows[0])+"\n")
    with pytest.raises(ValueError):build_site(path,tmp_path/"dup")
    rows[0]["human_review"]="accepted";path.write_text(json.dumps(rows[0])+"\n")
    with pytest.raises(ValueError):build_site(path,tmp_path/"approved")


def test_svg_overlay_stays_separate_from_original(tmp_path):
    path,rows=fixture(tmp_path)
    overlay=tmp_path/"box.svg"
    overlay.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect x="1" y="1" width="8" height="8" fill="none" stroke="red"/></svg>')
    rows[0]["overlays"][1]=str(overlay)
    path.write_text(json.dumps(rows[0])+"\n")
    out=tmp_path/"svg-site";build_site(path,out)
    row=json.loads((out/"data.json").read_text())["items"][0]
    assert row["overlays"][1].endswith('.svg') and row["overlays"][1] not in row["images"]
    assert (out/row["overlays"][1]).read_bytes()==overlay.read_bytes()


def test_review_ui_two_images_notes_and_strict_import(tmp_path):
    node=shutil.which("node") or "/tmp/qa-sites-node-runtime/package/bin/node"
    if not Path(node).exists():pytest.skip("Node unavailable; Python asset checks still run")
    path,_=fixture(tmp_path);out=tmp_path/"site";build_site(path,out)
    html=(out/"index.html").read_text();script=re.search(r"<script>(.*?)</script>",html,re.S).group(1)
    js=tmp_path/"page.js";js.write_text(script)
    harness=tmp_path/"check.cjs";harness.write_text(r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
class E{constructor(tag='div'){this.tagName=tag.toUpperCase();this.children=[];this.value='';this.checked=false;this.hidden=false;this.disabled=false;this.files=[];this.textContent='';this.classes=new Set();this.classList={toggle:(k,v)=>{if(v)this.classes.add(k);else this.classes.delete(k)}};}append(...xs){this.children.push(...xs)}replaceChildren(...xs){this.children=xs}setAttribute(){}focus(){}click(){this.onclick?.()}}
const ids={};for(const name of ['reviewer','group','status','show-answer','show-boxes','export','import','import-file','prev','next','progress','overall','empty','workspace','meta','images','question','options','answer-panel','answer','evidence','source','review-buttons','note','saved','reset','notice','intro'])ids[name]=new E();
const cache=new Map(),ctx={document:{getElementById:id=>ids[id],createElement:tag=>new E(tag),addEventListener(){}},window:{},localStorage:{getItem:k=>cache.get(k)||null,setItem:(k,v)=>cache.set(k,v)},setTimeout:()=>0,clearTimeout(){},Blob,URL:{createObjectURL(){return 'download'},revokeObjectURL(){}},Date,JSON,Map,Set};vm.createContext(ctx);vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),ctx);const api=ctx.window.CAPABILITY_REVIEW_API;
assert.equal(ids.images.children.length,2);assert.equal(ids['answer-panel'].hidden,true);assert.equal(api.exportBody().records[0].status,'pending');assert.equal(api.exportBody().records[0].reviewed_at,null);
api.saveVerdict('correct');assert.equal(api.exportBody().records[0].status,'pending');ids.reviewer.value='真实测试仅在内存';ids.reviewer.oninput();api.saveVerdict('correct');assert.equal(api.exportBody().records[0].status,'correct');assert(api.exportBody().records[0].reviewed_at);assert.equal(api.exportBody().analysis_queue.length,0);
ids.note.value='答案正确，但请核查椅子指代。';ids.note.oninput();assert.equal(api.exportBody().analysis_queue.length,1);assert.equal(api.exportBody().analysis_queue[0].status,'correct');
ids['show-boxes'].checked=true;ids['show-boxes'].onchange();assert.equal(ids.images.children[1].children[0].children.length,2);ids['show-answer'].checked=true;ids['show-answer'].onchange();assert.equal(ids['answer-panel'].hidden,false);assert(ids.answer.textContent.includes('A'));
ids.next.click();assert.equal(api.current().qid,'q2');ids.note.value='只有备注，尚未确定答案。';ids.note.oninput();assert.equal(api.exportBody().analysis_queue.length,2);assert.equal(api.exportBody().records[1].status,'pending');assert.equal(api.exportBody().records[1].reviewed_at,null);
const body=JSON.parse(JSON.stringify(api.exportBody()));assert.equal(Object.keys(api.validateImport(body)).length,2);
for(const field of ['manifest_sha256']){const bad=JSON.parse(JSON.stringify(body));bad[field]='wrong';assert.throws(()=>api.validateImport(bad));}
for(const field of ['question_sha256','reviewer_kind','reviewed_at']){const bad=JSON.parse(JSON.stringify(body));bad.records[0][field]=null;assert.throws(()=>api.validateImport(bad));}
const reversed=JSON.parse(JSON.stringify(body));reversed.records[0].image_sha256.reverse();assert.throws(()=>api.validateImport(reversed));const dup=JSON.parse(JSON.stringify(body));dup.records[1]=dup.records[0];assert.throws(()=>api.validateImport(dup));
const cacheOnly=JSON.parse(cache.get('interiorgs-capability-human-review:'+api.data.manifest_sha256));assert.equal(cacheOnly.records.length,2);console.log('two-image UI, hidden GT, optional notes queue, and bound imports passed; simulated DOM only');
''')
    result=subprocess.run([node,str(harness),str(js)],text=True,capture_output=True)
    assert result.returncode==0,result.stderr
