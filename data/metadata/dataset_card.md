# Dataset Card

This benchmark uses public datasets as source material and converts their annotations into VLM evaluation tasks.

## Source Datasets

- SKU-110K for retail shelf and dense object scenes.
- MVTec AD for industrial anomaly and quality inspection scenes.
- SHWD for safety helmet compliance scenes.

## Derived Tasks

- `retail_shelf_check`: asks whether visibly empty shelf slots or stockout-like areas are present.
- `industrial_safety_check`: asks whether an anomaly or unsafe condition is visible.
- `safety_helmet_check`: asks whether the target person/head region shows a safety helmet.

## Label Policy

Labels derived directly from source annotations are stored separately from manually reviewed labels.

Object-detection boxes are not treated as complete VLM labels. Ambiguous questions, such as empty shelf slots or safety risk level, require a manual review layer before they are used in final reports.

## Intended Use

Portfolio, research, and evaluation engineering demonstration.

## Non-Intended Use

Do not use this dataset bundle as a commercial monitoring product without reviewing upstream dataset licenses and collecting production-appropriate labels.

