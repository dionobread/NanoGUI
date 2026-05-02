"""Fix Salesforce bbox normalization from pixel [x,y,w,h] to normalized [x1,y1,x2,y2]."""
from pathlib import Path
import json
from PIL import Image

root = Path('/scratch/gz27/nanogui/datasets/salesforce_grounding')
images_dir = root / 'images'
annotations_dir = root / 'annotations'

fixed_count = 0
for anno_file in annotations_dir.glob('*_annotations.json'):
    with open(anno_file) as f:
        data = json.load(f)

    for item in data:
        bbox = item.get('bbox')
        if bbox is None:
            continue
        # Check if bbox is in pixel format [x, y, w, h]
        if any(v > 1.5 for v in bbox):
            img_path = item.get('image_path', '')
            if img_path:
                try:
                    img = Image.open(img_path)
                    iw, ih = img.size
                except Exception:
                    iw, ih = 1000, 1000
            else:
                iw, ih = 1000, 1000

            x, y, w, h = bbox
            x1 = max(0.0, min(1.0, x / iw))
            y1 = max(0.0, min(1.0, y / ih))
            x2 = max(0.0, min(1.0, (x + w) / iw))
            y2 = max(0.0, min(1.0, (y + h) / ih))
            item['bbox'] = [x1, y1, x2, y2]
            item['bbox_original'] = [x, y, w, h]
            fixed_count += 1

    with open(anno_file, 'w') as f:
        json.dump(data, f, indent=2)

print(f'Fixed {fixed_count} Salesforce bboxes from pixel to normalized [x1,y1,x2,y2]')
