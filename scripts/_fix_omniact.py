"""Re-download OmniAct images."""
from datasets import load_dataset
from pathlib import Path
import json
from PIL import Image

out_dir = Path('/scratch/gz27/nanogui/datasets/omniact')
images_dir = out_dir / 'images'
images_dir.mkdir(parents=True, exist_ok=True)

print('Loading OmniAct...')
ds = load_dataset('Writer/OmniAct')

for split_name, split_data in ds.items():
    print(f'  Processing {split_name}: {len(split_data)} samples')
    annotations = []
    for idx, sample in enumerate(split_data):
        annotation = {
            'id': f'{split_name}_{idx:05d}',
            'instruction': sample.get('task', ''),
            'data_type': sample.get('data_type', 'unknown'),
        }
        if 'box' in sample:
            annotation['bbox'] = sample['box']

        img = sample.get('image')
        if img is not None:
            if not isinstance(img, Image.Image):
                try:
                    img = Image.fromarray(img)
                except Exception:
                    img = None
            if img is not None:
                img_path = images_dir / f'{split_name}_{idx:05d}.png'
                img.save(img_path)
                annotation['image_path'] = str(img_path)
                annotation['image_size'] = list(img.size)

        annotations.append(annotation)

    with open(out_dir / 'annotations' / f'{split_name}_annotations.json', 'w') as f:
        json.dump(annotations, f, indent=2)
    print(f'  Saved {len(annotations)} annotations')

count = len(list(images_dir.glob('*.png')))
print(f'Total OmniAct images: {count}')
