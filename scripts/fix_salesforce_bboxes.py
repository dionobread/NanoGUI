#!/usr/bin/env python3
"""
Fix Salesforce Grounding dataset bounding boxes.
Converts pixel coordinates [x, y, w, h] to normalized [x1, y1, x2, y2].
"""

import json
from pathlib import Path

def fix_bboxes():
    root = Path("datasets/salesforce_grounding")
    for anno_file in root.rglob("*_annotations.json"):
        print(f"Fixing {anno_file}")
        with open(anno_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        fixed = 0
        for item in data:
            if "bbox" not in item or "image_size" not in item:
                continue

            bbox = item["bbox"]
            img_size = item["image_size"]

            # Check if bbox is in pixel format (values > 1.0)
            if any(v > 1.5 for v in bbox):
                x, y, w, h = bbox
                iw, ih = img_size

                # Convert to normalized [x1, y1, x2, y2]
                x1 = max(0.0, min(1.0, x / iw))
                y1 = max(0.0, min(1.0, y / ih))
                x2 = max(0.0, min(1.0, (x + w) / iw))
                y2 = max(0.0, min(1.0, (y + h) / ih))

                item["bbox"] = [x1, y1, x2, y2]
                fixed += 1

        # Save back
        with open(anno_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"  Fixed {fixed} bboxes")

if __name__ == "__main__":
    fix_bboxes()
