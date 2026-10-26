"""
从 manifest.csv 生成 zero-shot 测试用的“考卷” test_set.json。
只用单棋子特写图（默认前 36 张，000-035），只考棋子种类。
运行：python make_testset.py
"""
import os
import csv
import json

BASE = os.path.dirname(__file__)
MANIFEST = os.path.join(BASE, "data", "manifest.csv")
OUT = os.path.join(BASE, "data", "test_set.json")

# 统一的问题（三组实验都用这一句）
QUESTION = "What chess piece is shown in this image? Answer with one word."

# 只用单棋子图：文件名 000.jpg ~ 035.jpg
SINGLE_PIECE_MAX_INDEX = 35


def main():
    items = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = int(row["file"].split(".")[0])  # "000.jpg" -> 0
            if idx > SINGLE_PIECE_MAX_INDEX:
                continue  # 跳过多棋子同框的图
            items.append({
                "file": row["file"],
                "question": QUESTION,
                "answer_piece": row["piece"],   # 标准答案：棋子种类
            })

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    # 打印分布，确认考卷合理
    from collections import Counter
    dist = Counter(it["answer_piece"] for it in items)
    print(f"生成 {len(items)} 道题")
    print("各棋子数量：", dict(dist))
    print(f"考卷已存到: {OUT}")


if __name__ == "__main__":
    main()
