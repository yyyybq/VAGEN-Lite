"""Build a separate static human review page for capability questions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any
import xml.etree.ElementTree as ET

from PIL import Image

REVIEW_LABELS = {
    "correct": "答案对", "incorrect": "答案错", "undecidable": "从图里看不出来",
    "bad_image": "画面质量差",
}
TEMPLATE = Path(__file__).with_name("review_template.html")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def question_sha256(row: dict[str, Any]) -> str:
    """Bind the full question and the displayed option order, not only its stem."""
    text = json.dumps({"question": row["question"], "options": row["options"]},
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _digest(text.encode())


def build_site(questions_path: Path, out: Path) -> dict[str, Any]:
    questions_path, out = Path(questions_path), Path(out)
    raw = questions_path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows or len({r["qid"] for r in rows}) != len(rows):
        raise ValueError("题目必须非空，且每个题号唯一")
    if out.exists() and any(out.iterdir()):
        raise ValueError("请用新的空目录，避免覆盖已有网页和审阅记录")
    out.mkdir(parents=True, exist_ok=True)
    assets = out / "assets"
    assets.mkdir(exist_ok=True)
    copies: dict[str, str] = {}

    def copy_asset(value: str, *, original: bool) -> tuple[str, str]:
        path = Path(value)
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"图像必须是存在的绝对路径：{value}")
        content = path.read_bytes()
        digest = _digest(content)
        if not original and path.suffix.lower() == ".svg":
            tree = ET.fromstring(content)
            if not tree.tag.endswith("svg") or any(e.tag.endswith(("script", "foreignObject")) for e in tree.iter()):
                raise ValueError("检查用SVG包含无效内容")
        elif path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("图像或检查叠加图必须为栅格图片")
        else:
            with Image.open(path) as im:
                im.load()
        # Separate paths keep inspection overlays out of model-image lists.
        key = ("image_" if original else "overlay_") + digest
        if key not in copies:
            rel = "assets/" + key + path.suffix.lower()
            shutil.copyfile(path, out / rel)
            copies[key] = rel
        return copies[key], digest

    items = []
    for row in rows:
        options = row["options"]
        if not isinstance(options, list) or not 2 <= len(options) <= 8 or not all(isinstance(x, str) for x in options):
            raise ValueError(f"选项无效：{row['qid']}")
        if len(row["answer"]) != 1 or row["answer"] not in "ABCDEFGH"[:len(options)]:
            raise ValueError(f"答案字母与选项不匹配：{row['qid']}")
        if row.get("human_review", "pending") != "pending":
            raise ValueError("新样题页不能静默重置已有人工结论")
        images = row["images"]
        overlays = row.get("overlays") or [None] * len(images)
        if not images or len(overlays) != len(images):
            raise ValueError("检查叠加图必须与原图逐张对应，缺省项用null")
        copied = [copy_asset(x, original=True) for x in images]
        copied_overlays = [copy_asset(x, original=False)[0] if x else None for x in overlays]
        items.append({k: row[k] for k in ("qid", "group", "task_type", "scene_id", "parent_task_id", "question",
                     "options", "answer", "answer_key", "evidence_zh", "source")} | {
            "images": [x[0] for x in copied], "image_sha256": [x[1] for x in copied],
            "overlays": copied_overlays, "question_sha256": question_sha256(row), "human_review": "pending",
        })
    data = {"schema_version": "interiorgs_capability_review_v1", "manifest_sha256": _digest(raw),
            "review_labels": REVIEW_LABELS, "items": items, "human_review": "pending",
            "model_results_included": False, "training_release_count": 0}
    blob = json.dumps(data, ensure_ascii=False, allow_nan=False)
    (out / "data.json").write_text(blob + "\n", encoding="utf-8")
    # Embed data too, so the same page opens from disk without fetch/CORS.
    safe = blob.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__CAPABILITY_DATA__*/null", safe)
    (out / "index.html").write_text(html, encoding="utf-8")
    summary = {"questions": len(items), "image_inputs": sum(len(x["images"]) for x in items),
               "independent_images": len({h for x in items for h in x["image_sha256"]}),
               "scenes": len({x["scene_id"] for x in items}), "human_pending": len(items),
               "manifest_sha256": data["manifest_sha256"], "assets": len(copies), "published": False}
    (out / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_site(args.questions, args.out), ensure_ascii=False))


if __name__ == "__main__":
    main()
