"""학습/테스트 이미지 전처리 + 오프라인 증강.

원본(archive/train, archive/test)은 그대로 두고, 모두 224x224로 줄여서
새 폴더에 저장한다.
- train -> train_224: 줄인 원본 + 회전 / 샤프닝 / 노이즈 버전 (데이터셋 4배)
- test  -> test_224 : 줄이기만 함 (평가용이므로 증강하지 않음)

사용법:
    uv run --no-sync python augment.py
    uv run --no-sync python augment.py --size 224 --suffix _224
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


def process_split(
    src: Path, dst: Path, size: int, augment: bool, rng: random.Random
) -> None:
    """src의 클래스 폴더별 이미지를 size x size로 줄여 dst에 저장한다."""
    for class_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        out_dir = dst / class_dir.name
        out_dir.mkdir(parents=True)
        files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        for path in files:
            with Image.open(path) as img:
                # 학습 코드의 transforms.Resize((224, 224))와 같은 방식(비율 무시)
                small = img.convert("RGB").resize(
                    (size, size), Image.Resampling.BILINEAR
                )
            small.save(out_dir / f"{path.stem}.jpg", quality=95)
            if augment:
                for name, fn in AUGMENTATIONS.items():
                    fn(small, rng).save(out_dir / f"{path.stem}_{name}.jpg", quality=95)
        n = len(list(out_dir.iterdir()))
        print(f"{dst.name}/{class_dir.name}: 원본 {len(files)} -> {n}장")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("archive"))
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--suffix", default="_224")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits = {"train": True, "test": False}  # split 이름: 증강 여부
    dsts = {name: args.root / f"{name}{args.suffix}" for name in splits}
    for dst in dsts.values():
        if dst.exists():
            raise SystemExit(f"{dst} 가 이미 있습니다. 지우고 다시 실행하세요.")

    rng = random.Random(args.seed)
    for name, augment in splits.items():
        process_split(args.root / name, dsts[name], args.size, augment, rng)


if __name__ == "__main__":
    main()
