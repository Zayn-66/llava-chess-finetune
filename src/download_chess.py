"""
下载 Trelis/chess_pieces 数据集，导出为本地图片 + 一份清单表格。
运行前先装依赖：pip install datasets
运行：python download_chess.py
"""
import os
import re
import csv

from datasets import load_dataset

# ---- 配置 ----
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
IMG_DIR = os.path.join(OUT_DIR, "chess_images")
MANIFEST = os.path.join(OUT_DIR, "manifest.csv")

# 6 种棋子，用来从 caption 里自动识别“核心棋子名”
PIECE_NAMES = ["queen", "king", "knight", "bishop", "rook", "pawn"]


def extract_piece(caption: str) -> str:
    """从 caption 句子里找出是哪种棋子；找不到就返回 unknown。"""
    text = caption.lower()
    for name in PIECE_NAMES:
        if re.search(rf"\b{name}s?\b", text):  # 兼容复数，如 pawns
            return name
    return "unknown"


def extract_color(caption: str) -> str:
    """顺便提取颜色（white / black），有些实验可能用得上。"""
    text = caption.lower()
    if "white" in text:
        return "white"
    if "black" in text:
        return "black"
    return "unknown"


def main():
    os.makedirs(IMG_DIR, exist_ok=True)

    print("正在下载 Trelis/chess_pieces ...（首次会联网下载，稍等）")
    ds = load_dataset("Trelis/chess_pieces", split="train")
    print(f"共 {len(ds)} 条数据\n")

    rows = []
    for i, row in enumerate(ds):
        fname = f"{i:03d}.jpg"
        row["image"].convert("RGB").save(os.path.join(IMG_DIR, fname))

        caption = row["caption"]
        piece = extract_piece(caption)
        color = extract_color(caption)
        rows.append({"file": fname, "caption": caption, "piece": piece, "color": color})
        print(f"{fname}  | {piece:8s} | {color:6s} | {caption}")

    # 写清单
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "caption", "piece", "color"])
        writer.writeheader()
        writer.writerows(rows)

    # 简单统计一下每种棋子有几张，方便你了解数据分布
    from collections import Counter
    counts = Counter(r["piece"] for r in rows)
    print("\n各棋子数量分布：", dict(counts))

    print(f"\n完成 ✅")
    print(f"图片已存到: {IMG_DIR}")
    print(f"清单已存到: {MANIFEST}")
    print("现在可以在访达里打开 data/chess_images 文件夹翻图看了。")


if __name__ == "__main__":
    main()
