"""Public task context must survive observation windows and token budgeting."""
import hashlib
import re
import unicodedata

TASK_CONTEXT_VERSION = "persistent_public_task_v1"


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text))).strip()


def assert_task_text(text, task):
    if not task or normalize("Task: " + task) not in normalize(text):
        raise ValueError("policy input lost its public task context")


def check_policy_tokens(tokenizer, prompt_ids, task, prompt_limit):
    """Check exactly the IDs sent to inference, not an earlier template."""
    if len(prompt_ids) > prompt_limit:
        raise ValueError("policy prompt exceeds budget; never truncate task/image tokens")
    decoded = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    assert_task_text(decoded, task)
    return {"version": TASK_CONTEXT_VERSION, "task_present": True,
            "task_sha256": hashlib.sha256(task.encode()).hexdigest(),
            "prompt_tokens": len(prompt_ids)}
