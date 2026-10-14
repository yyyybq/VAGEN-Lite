"""Read-only CPU budget audit of existing r5 task/frames manifests; no records or images emitted."""
import collections
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path('/mnt/umm/users/yinbaiqiao/VAGEN-Lite')
RUN = ROOT / 'exps/vagen_active_spatial/interiorgs_v10_gen_20261014'
SNAP = RUN / 'job/src_shard1'
TOK = ROOT / 'exps/vagen_active_spatial/r1_h1_aoss_repair_20260905/r1_canonical_dev_eval32_20260915/model_restore/qwen25vl7b_pretrained_cc594898/tokenizer.json'
sys.path.insert(0, str(SNAP))
from tokenizers import Tokenizer
from data_gen.interiorgs_v10.admission import load_policy, chunk_certificate, turn_boundaries
from data_gen.interiorgs_v10.qa_states import certificate, certificate_poses
from data_gen.interiorgs_v10.release import protocol_system_prompt, _user_turn

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

tokenizer = Tokenizer.from_file(str(TOK))
policy = load_policy()
exp = policy['sft']['export']
cache = {}
def n(text):
    if text not in cache:
        cache[text] = len(tokenizer.encode(text, add_special_tokens=False).ids)
    return cache[text]

rows = [json.loads(x) for x in (RUN/'sft/rows.jsonl').open() if x.strip()]
counts = collections.defaultdict(collections.Counter)
lengths = collections.defaultdict(list)
image_token = '<|vision_start|>' + '<|image_pad|>' * 81 + '<|vision_end|>'
for row in rows:
    s = row['split']; c = counts[s]
    cert = certificate(row)
    chunks = chunk_certificate(cert, exp['chunking'], exp['max_actions_per_turn'])
    bounds = turn_boundaries(row, policy)
    poses = certificate_poses(row)
    c['tasks'] += 1; c['primitive_actions'] += len(cert)
    c['planned_render_frames'] += len(bounds)
    c['assistant_turns'] += len(chunks) + int(exp['explicit_done'])
    c['training_image_occurrences'] += len(chunks) + int(exp['explicit_done'])
    system = protocol_system_prompt(row['task_type'], policy)
    total = n('<|im_start|>system\n' + system + '<|im_end|>\n')
    content_total = n(system)
    pairs = [(i, chunk, bounds[i]) for i, chunk in enumerate(chunks)]
    if exp['explicit_done']:
        pairs.append((len(chunks), ['done'], len(cert)))
    for i, chunk, b in pairs:
        user = _user_turn(row['task_description'], i == 0, policy, poses[b])
        answer = '<action>' + ''.join(a + '|' for a in chunk) + '</action>'
        c['assistant_content_tokens'] += n(answer)
        c['estimated_loss_tokens_with_im_end_newline'] += n(answer + '<|im_end|>\n')
        content_total += n(user.replace('<image>', '')) + n(answer)
        total += n('<|im_start|>user\n' + user.replace('<image>', image_token) + '<|im_end|>\n<|im_start|>assistant\n')
        total += n(answer + '<|im_end|>\n')
    c['text_content_tokens_excluding_images_template'] += content_total
    c['estimated_total_tokens_chatml_visual81'] += total
    lengths[s].append(total)

output = {
    'source_run': str(RUN), 'source_snapshot': str(SNAP),
    'source_commit': (RUN/'code_commit.txt').read_text().strip(),
    'tokenizer': str(TOK), 'tokenizer_sha256': sha(TOK),
    'policy': exp,
    'assumptions': ['No GPU, no render, no dataset export, no trained-model inference.',
                    'Counts use existing certificates and frozen r5 prompt functions in memory.',
                    '256x256 RGB assumed smart-resized to 252x252; patch 14, merge 2 gives 81 LLM visual tokens.',
                    'ChatML qwen2_vl template; train_on_prompt false, mask_history false, no packing/truncation simulated.',
                    'Estimated loss includes assistant content + im_end + newline; actual labels must be audited after final processing.',
                    'Terminal frames are rendered for validation but not fed into action-only SFT when explicit_done=false.'],
    'counts': {s: dict(v) for s,v in counts.items()},
    'sequence_lengths': {s: {'min':min(v), 'median':sorted(v)[len(v)//2], 'max':max(v), 'over_16384':sum(x>16384 for x in v)} for s,v in lengths.items()},
    'source_sha256': {str(p.relative_to(ROOT)):sha(p) for p in [RUN/'sft/rows.jsonl', RUN/'sft/frames.jsonl', RUN/'split_manifest.json', SNAP/'data_gen/interiorgs_v10/admission.py', SNAP/'data_gen/interiorgs_v10/release.py', SNAP/'data_gen/interiorgs_v10/qa_states.py', SNAP/'data_gen/interiorgs_v10/configs/admission_v2.json', TOK.parent/'tokenizer_config.json', TOK.parent/'preprocessor_config.json', ROOT/'third_party/LLaMA-Factory/src/llamafactory/data/template.py']}
}
Path('/tmp/interiorgs_qa_budget_audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k:output[k] for k in ['source_commit','counts','sequence_lengths']}, indent=2))
