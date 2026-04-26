# Label Policy

## Principles

- Prefer deterministic labels from source annotations when the task is directly supported.
- Add manual review labels when a VLM question requires judgment beyond the source annotation.
- Track `label_version`, `label_source`, and `is_hard_case` for every task.
- Keep raw annotations, derived labels, and human review corrections separate.

## Task-Specific Rules

### Safety Helmet Check

Use SHWD `hat` annotations to derive `answer=true`. Use visible head/person annotations without a matching helmet to derive `answer=false`. Mark crowded or occluded cases as hard cases.

### Retail Shelf Check

Use SKU-110K boxes to derive density and crowding metadata. Use manual review for `missing_stock`, because dense product boxes alone do not prove an empty shelf slot.

### Industrial Safety Check

Use MVTec AD normal/anomaly folders to derive `anomaly_present`. Use defect type and mask size to derive a coarse `risk_level`, then review a sample manually.

