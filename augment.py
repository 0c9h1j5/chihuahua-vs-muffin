"""학습/검증/테스트 이미지 전처리 + 오프라인 증강.

원본(archive/train, archive/test)은 그대로 두고, 모두 224x224로 줄여서
새 폴더에 저장한다.
- train -> train_224: 원본의 85%를 줄인 것 + 회전 / 샤프닝 / 노이즈 버전 (4배)
- train -> val_224  : 원본의 나머지 15%, 줄이기만 함 (설정 고르기용)
- test  -> test_224 : 줄이기만 함 (최종 평가용)

train/val 은 원본 이미지 단위로 나누므로, 같은 원본에서 나온 증강 사진이
val 쪽에 섞여 들어가지 않는다. val/test 는 증강하지 않는다.

사용법:
    uv run --no-sync python augment.py
    uv run --no-sync python augment.py --val-ratio 0.2 --size 224 --suffix _224
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def rotate(img: Image.Image, rng: random.Random) -> Image.Image:
    """±25도 회전 후 검은 모서리가 안 보이게 가운데를 잘라 원래 크기로 되돌린다."""
    angle = rng.uniform(-25, 25)
    w, h = img.size
    rotated = img.rotate(angle, resample=Image.Resampling.BICUBIC)
    rad = math.radians(abs(angle))
    scale = 1 / (math.cos(rad) + math.sin(rad) * max(w / h, h / w))
    cw, ch = int(w * scale), int(h * scale)
    left, top = (w - cw) // 2, (h - ch) // 2
    cropped = rotated.crop((left, top, left + cw, top + ch))
    return cropped.resize((w, h), Image.Resampling.BICUBIC)


def sharpen(img: Image.Image, rng: random.Random) -> Image.Image:
    """선명도를 2~3배로 높인다."""
    return ImageEnhance.Sharpness(img).enhance(rng.uniform(2.0, 3.0))


def add_noise(img: Image.Image, rng: random.Random) -> Image.Image:
    """가우시안 노이즈 (표준편차 8~20)를 더한다."""
    arr = np.asarray(img, dtype=np.float32)
    np_rng = np.random.default_rng(rng.randrange(2**32))
    noise = np_rng.normal(0, rng.uniform(8, 20), arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


AUGMENTATIONS = {"rot": rotate, "sharp": sharpen, "noise": add_noise}


def list_images(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def save_images(
    files: list[Path], out_dir: Path, size: int, augment: bool, rng: random.Random
) -> None:
    """files를 size x size로 줄여 out_dir에 저장한다. augment면 증강본도 저장."""
    out_dir.mkdir(parents=True)
    for path in files:
        with Image.open(path) as img:
            # 학습 코드의 transforms.Resize((224, 224))와 같은 방식(비율 무시)
            small = img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        small.save(out_dir / f"{path.stem}.jpg", quality=95)
        if augment:
            for name, fn in AUGMENTATIONS.items():
                fn(small, rng).save(out_dir / f"{path.stem}_{name}.jpg", quality=95)
    n = len(list(out_dir.iterdir()))
    print(f"{out_dir.parent.name}/{out_dir.name}: 원본 {len(files)} -> {n}장")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("archive"))
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--suffix", default="_224")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_dst = args.root / f"train{args.suffix}"
    val_dst = args.root / f"val{args.suffix}"
    test_dst = args.root / f"test{args.suffix}"
    for dst in (train_dst, val_dst, test_dst):
        if dst.exists():
            raise SystemExit(f"{dst} 가 이미 있습니다. 지우고 다시 실행하세요.")

    split_rng = random.Random(args.seed)  # train/val 나누기 전용
    aug_rng = random.Random(args.seed + 1)  # 증강 전용

    for class_dir in sorted(p for p in (args.root / "train").iterdir() if p.is_dir()):
        files = list_images(class_dir)
        split_rng.shuffle(files)
        n_val = round(len(files) * args.val_ratio)
        val_files, train_files = sorted(files[:n_val]), sorted(files[n_val:])
        save_images(train_files, train_dst / class_dir.name, args.size, True, aug_rng)
        save_images(val_files, val_dst / class_dir.name, args.size, False, aug_rng)

    for class_dir in sorted(p for p in (args.root / "test").iterdir() if p.is_dir()):
        files = list_images(class_dir)
        save_images(files, test_dst / class_dir.name, args.size, False, aug_rng)


if __name__ == "__main__":
    main()
