"""Bind actual assistant image reviews to this production version.

Assistant review permits candidate-format inspection only. Human approval is
never inferred and qa_sft_allowed remains false in this command.
"""
import argparse, json, hashlib
from collections import Counter
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--reviews',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    rows=[json.loads(x) for x in Path(a.manifest).read_text().splitlines() if x.strip()]; reviews={}
    for path in a.reviews:
        d=json.loads(Path(path).read_text()); d=d.get('reviews',[]) if isinstance(d,dict) else d
        for r in d:
            if r['sample_id'] in reviews: raise ValueError('duplicate review ID')
            if r.get('reviewer_kind')!='assistant' or not r.get('reviewer') or not r.get('reason'): raise ValueError('review requires assistant identity and reason')
            if r.get('status') not in ('passed','rejected'): raise ValueError('review must be an actual decision')
            if r.get('status')=='rejected' and r.get('category') not in {'坏图','目标不明确','题意不明确','标签错误','证据不足'}: raise ValueError('unknown rejection category')
            reviews[r['sample_id']]=r
    unknown=set(reviews)-{r['sample_id'] for r in rows}
    if unknown: raise ValueError('reviews reference unknown samples')
    counts=Counter(); reasons=Counter()
    for row in rows:
        r=reviews.get(row['sample_id'])
        if r:
            if sha(row['public_observation']['image_path'])!=r['image_sha256']: raise ValueError('review image no longer matches')
            if r.get('question_sha256')!=hashlib.sha256(row['question'].encode()).hexdigest(): raise ValueError('reviewed question changed')
            state_answer=row.get('automatic_state_predicate_answer',row.get('private_answer'))
            if r['status']=='passed' and r.get('independent_visual_answer')!=state_answer: raise ValueError('visual answer disagrees with state label')
            if r['status']=='passed' and row.get('visibility_admission',{}).get('ok') is not True: raise ValueError('visibility gate not passed')
            row['assistant_review']=r; row['candidate_qa_allowed']=r['status']=='passed'
            if r['status']=='rejected':
                # A simulator predicate is not a valid QA answer when the
                # observation or question fails review. Preserve it only as
                # diagnostic evidence, never as a supervised Yes/No label.
                row['automatic_state_predicate_answer']=state_answer
                row['private_answer']=None
                row['label_validity']='rejected_after_visual_review'
                row['observability_validity']='invalid_or_insufficient'
                row['failure_reason']=r['category']
            else:
                row['private_answer']=state_answer
                row['label_validity']='valid'
                row['observability_validity']='valid'
                row['observability_evidence']='actual_assistant_image_review; human review still pending'
        else: row['candidate_qa_allowed']=False
        row['qa_sft_allowed']=False
        row['semantic_review']={'status':'pending','reviewer':None,'reason':None}
        counts[row.get('assistant_review',{}).get('status','pending')]+=1
        if r and r['status']=='rejected': reasons[r['category']]+=1
        # Preserve exactly the image/question pair that was actually reviewed.
    dest=Path(a.output)
    if dest.exists():raise ValueError('versioned reviewed manifest already exists')
    dest.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
    summary={'manifest_rows':len(rows),'independent_inputs':len({sha(row['public_observation']['image_path']) for row in rows}),'assistant_review_counts':dict(counts),'rejection_categories':dict(reasons),'human_approved':0,'training_released':0,'qa_sft_allowed':False,'candidate_scope':'read-only format and supervision validation; human review pending','source_manifest_sha256':sha(a.manifest),'reviews':{x:sha(x) for x in a.reviews}}
    dest.with_suffix('.review_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
