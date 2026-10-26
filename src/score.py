"""
步骤5：评分。读 baseline_results.json，用关键词匹配判对错，
算总准确率 + 每类棋子准确率 + 混淆情况。
用法：python score.py [结果文件]  （默认 data/baseline_results.json）
"""
import os
import sys
import json
from collections import Counter, defaultdict

BASE = os.path.dirname(__file__)
PIECES = ["queen", "king", "knight", "bishop", "rook", "pawn"]


def predicted_piece(answer: str) -> str:
    """从模型的自由回答里找出它说的是哪种棋子；找不到返回 none。"""
    a = answer.lower()
    for p in PIECES:
        if p in a:
            return p
    return "none"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "data", "baseline_results.json")
    with open(path, encoding="utf-8") as f:
        results = json.load(f)

    correct = 0
    per_class_total = Counter()
    per_class_correct = Counter()
    confusion = defaultdict(Counter)  # 标答 -> 模型预测 计数

    for r in results:
        gold = r["answer_piece"]
        pred = predicted_piece(r["model_answer"])
        per_class_total[gold] += 1
        confusion[gold][pred] += 1
        if pred == gold:
            correct += 1
            per_class_correct[gold] += 1

    total = len(results)
    print(f"文件: {path}\n")
    print(f"总准确率: {correct}/{total} = {correct/total*100:.1f}%\n")

    print("各棋子准确率:")
    for p in PIECES:
        t = per_class_total[p]
        if t:
            print(f"  {p:8s}: {per_class_correct[p]}/{t} = {per_class_correct[p]/t*100:.0f}%")

    print("\n模型都把它们认成了什么（标答 -> 预测分布）:")
    for p in PIECES:
        if per_class_total[p]:
            dist = ", ".join(f"{k}×{v}" for k, v in confusion[p].most_common())
            print(f"  真实 {p:8s} -> {dist}")


if __name__ == "__main__":
    main()
