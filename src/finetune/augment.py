"""
只对训练集做离线增强：每张原图保留原图 + 生成 N 个视觉变体（默认 25 → 每张 26 条）。

关键：先按源图切分 train/val/test，再只对 train 做增强，
这样同一张源图的变体不可能漏进验证/测试集（防数据泄漏）。

增强只改视角和光照，不改标签：
  - 视角：随机旋转、透视变换、随机裁剪缩放（保守参数，避免把棋子切掉）、水平翻转
  - 光照：亮度 / 对比度 / 饱和度 / 色调抖动

用法：
  python src/finetune/augment.py --manifest data/train.json --image-root data \
      --out-dir data/train_aug --n 25
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image
from torchvision import transforms as T

from common import SEED


def build_transform(size: int) -> T.Compose:
    return T.Compose([
        T.RandomApply([T.RandomRotation(12, fill=255)], p=0.6),
        T.RandomPerspective(distortion_scale=0.15, p=0.4, fill=255),
        T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),  # 棋子左右对称，翻转不改变类别
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.03),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="训练集 manifest（仅 train！）")
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=25, help="每张原图生成的变体数")
    ap.add_argument("--size", type=int, default=448)
    args = ap.parse_args()

    random.seed(SEED)
    import torch
    torch.manual_seed(SEED)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tf = build_transform(args.size)
    records = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    new = []
    for r in records:
        img = Image.open(Path(args.image_root) / r["image"]).convert("RGB")
        stem = Path(r["image"]).stem
        # 原图本身也保留
        orig = out / f"{stem}_orig.jpg"
        img.save(orig, quality=95)
        new.append({**r, "id": f"{r['id']}_orig", "image": str(orig), "source_id": r["id"]})
        for k in range(args.n):
            p = out / f"{stem}_aug{k:02d}.jpg"
            tf(img).save(p, quality=95)
            new.append({**r, "id": f"{r['id']}_aug{k:02d}", "image": str(p), "source_id": r["id"]})

    (out / "train_aug.json").write_text(json.dumps(new, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(records)} 张源图 -> {len(new)} 条训练记录，写入 {out / 'train_aug.json'}")


if __name__ == "__main__":
    main()
