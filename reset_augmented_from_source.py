from pathlib import Path
import shutil

src = Path('data/merged_RFMID')
dst = Path('data/merged_RFMID_augmented')
tmp = Path('data/merged_RFMID_augmented_reset_tmp')

if not src.exists():
    raise FileNotFoundError(f'Source not found: {src}')

if tmp.exists():
    shutil.rmtree(tmp)

shutil.copytree(src, tmp)

if dst.exists():
    shutil.rmtree(dst)

tmp.rename(dst)

class_count = len([p for p in dst.iterdir() if p.is_dir()])
file_count = len([p for p in dst.rglob('*') if p.is_file()])
print(f'AUG_CLASSES={class_count}')
print(f'FILES_TOTAL={file_count}')
