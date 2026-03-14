from pathlib import Path
import shutil
import cv2

SRC = Path('data/merged_RFMID')
DST = Path('data/merged_RFMID_augmented')
VALID = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def images(folder: Path):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID]


def count_readable(files):
    readable = []
    bad = []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            bad.append(p)
        else:
            readable.append(p)
    return readable, bad


def main():
    if not SRC.exists() or not DST.exists():
        raise FileNotFoundError('Source or destination dataset folder missing')

    src_classes = sorted([p.name for p in SRC.iterdir() if p.is_dir()])

    removed = 0
    restored = 0

    for cname in src_classes:
        sdir = SRC / cname
        ddir = DST / cname
        ddir.mkdir(parents=True, exist_ok=True)

        d_files = images(ddir)
        d_readable, d_bad = count_readable(d_files)

        for b in d_bad:
            try:
                b.unlink(missing_ok=True)
                removed += 1
            except Exception:
                pass

        if len(d_readable) == 0:
            s_files = images(sdir)
            s_readable, _ = count_readable(s_files)
            for sp in s_readable:
                shutil.copy2(sp, ddir / sp.name)
                restored += 1

    # final stats
    class_dirs = sorted([p for p in DST.iterdir() if p.is_dir()])
    nonzero = 0
    total = 0
    for cdir in class_dirs:
        r, _ = count_readable(images(cdir))
        if len(r) > 0:
            nonzero += 1
        total += len(r)

    print(f'CLASSES_TOTAL={len(class_dirs)}')
    print(f'CLASSES_NONZERO_READABLE={nonzero}')
    print(f'READABLE_FILES_TOTAL={total}')
    print(f'REMOVED_BAD_FILES={removed}')
    print(f'RESTORED_FROM_SOURCE={restored}')


if __name__ == '__main__':
    main()
