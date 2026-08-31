# Active Spatial OOD Splits — Summary

## Training Data Info
- Train items: 11157
- Train scenes: 92
- Train object labels: 454
- Train task types: absolute_positioning, apparent_size_ordering, centering, equidistance, fov_inclusion, occlusion_alignment, projective_relations

## OOD Split Statistics

| Split | Axis | N items | Scenes | Task distribution |
|-------|------|---------|--------|-------------------|
| ood_scene | — | 316 | 8 | absolute_positioning:138, equidistance:35, fov_inclusion:32, occlusion_alignment:38, projective_relations:73 |
| ood_instance | — | 200 | 8 | absolute_positioning:129, equidistance:13, fov_inclusion:13, occlusion_alignment:16, projective_relations:29 |
| ood_category | — | 338 | 54 | absolute_positioning:49, centering:25, equidistance:66, fov_inclusion:66, occlusion_alignment:66, projective_relations:66 |
| ood_template | — | 355 | 80 | absolute_positioning:66, centering:25, equidistance:66, fov_inclusion:66, occlusion_alignment:66, projective_relations:66 |
| ood_geometry | — | 335 | 73 | absolute_positioning:66, centering:5, equidistance:66, fov_inclusion:66, occlusion_alignment:66, projective_relations:66 |

## Split Descriptions

### Split-1: OOD Scene (`ood_scene.jsonl`)
- Items from the **8 held-out scenes** not present in training.
- Held-out scene IDs: 0229_840306, 0240_840881, 0266_840789, 0275_840778, 0276_840780, 0351_840366, 0367_840260, 0374_840227
- Tests: Can the model navigate spatially in completely unseen room layouts?

### Split-2: OOD Instance (`ood_instance.jsonl`)
- Items from OOD scenes where `object_label` **was seen in training**.
- Same category of object (e.g., "wardrobe") but in a different room.
- Subset of Split-1; isolates instance-level vs. scene-level generalization.
- Tests: Does the model generalize to new instances of familiar object types?

### Split-3: OOD Category (`ood_category.jsonl`)
- Items where `object_label` is **entirely absent from the training set**.
- Unique new labels: 89
- Tests: Can the model navigate to object types it has never seen during training?

### Split-4: OOD Template (`ood_template.jsonl`)
- Same task content, but `task_description` rewritten with **alternative phrasing**.
- Rewrites per task type:
  - `absolute_positioning`: "Move to any position {d}m from X" → "Navigate to a location {d} meters away from X"
  - `equidistance`: "… equidistant from A and B" → "… equally far from both A and B"
  - `projective_relations`: "A appears to the left of B" → "B appears to the right of A" (semantically equivalent)
  - `occlusion_alignment`: "A is hidden behind B" → "B fully blocks your view of A"
  - `fov_inclusion`: "both A and B are visible" → "you can see both A and B at the same time"
  - `centering`: "A is centered between B and C" → "A appears midway between B and C"
  - `apparent_size_ordering`: "A appears larger than B" → "A looks bigger than B"
- Original description stored in `task_description_original` field.
- Tests: Does the model parse task instructions by template matching or semantic understanding?

### Split-5: OOD Geometry (`ood_geometry.jsonl`)
- Items with `distance` **outside [p10, p90]** of training distribution per task type.
- Thresholds (computed from train_100scenes_7types.jsonl):
  - `absolute_positioning`: p10=1.50m, p90=2.58m
  - `centering`: p10=5.63m, p90=10.50m
  - `equidistance`: p10=3.25m, p90=9.32m
  - `fov_inclusion`: p10=1.35m, p90=3.45m
  - `occlusion_alignment`: p10=2.13m, p90=5.95m
  - `projective_relations`: p10=3.19m, p90=6.17m
- Tests: Can the model accurately execute spatial tasks at unusual distances/scales?

## Usage

```bash
# Run evaluation on a specific OOD split
# (point your eval config to the split JSONL instead of the main val JSONL)
python3 scripts/gen_ood_splits.py  # regenerate splits

# Analyze OOD results after running evaluation
# python3 scripts/analyze_experiments.py --exps <exp_name> --ood_eval
```
