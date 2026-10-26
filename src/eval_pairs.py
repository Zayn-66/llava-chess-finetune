"""
配对级评测器：把模型的自由文本回答解析成〈颜色 + 棋子〉集合，与标准答案比对。

相比"回答里含关键词就算对"，它能抓出两类错误：
  - 颜色认错（white rook 说成 black rook）
  - 幻觉（编出图里不存在的棋子）

指标：
  Micro Precision / Recall / F1、整图全对率（exact match）、幻觉率 = FP / (TP + FP)

用法：
  python src/eval_pairs.py results.json [--ref-key reference] [--pred-key prediction]
results.json 是一个列表，每条至少包含参考答案和模型回答两个字段。
"""
import argparse
import json
import re

COLORS = r"(?:white|black|brown)"
PIECES = r"(?:pawn|knight|bishop|rook|queen|king)"


def extract_chess_pairs(text: str) -> set:
    """'A white king and a black queen' -> {'white king', 'black queen'}"""
    text = text.lower()
    pairs = set()
    # 形如 "white rook" / "white chess rook"
    for c, p in re.findall(rf"\b({COLORS})\s+(?:chess\s+)?({PIECES})s?\b", text):
        pairs.add(f"{c} {p}")
    # 形如 "a white chess piece, namely, a queen"
    for c, p in re.findall(
        rf"\b({COLORS})\s+(?:chess\s+)?piece,?\s+(?:namely,?\s*-?\s*|being\s+)?(?:a|an)\s+({PIECES})\b",
        text,
    ):
        pairs.add(f"{c} {p}")
    return pairs


def evaluate(records, ref_key="reference", pred_key="prediction"):
    tp_all = fp_all = fn_all = exact = 0
    per_sample = []
    for r in records:
        gold = extract_chess_pairs(r[ref_key])
        pred = extract_chess_pairs(r[pred_key])
        tp, fp, fn = len(gold & pred), len(pred - gold), len(gold - pred)
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn
        exact += gold == pred
        per_sample.append({"id": r.get("id"), "gold": sorted(gold), "pred": sorted(pred),
                           "tp": tp, "fp": fp, "fn": fn})

    p = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0.0
    rc = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    return {
        "n": len(records), "tp": tp_all, "fp": fp_all, "fn": fn_all,
        "micro_precision": p, "micro_recall": rc, "micro_f1": f1,
        "exact_match_rate": exact / len(records) if records else 0.0,
        "hallucination_rate": fp_all / (tp_all + fp_all) if tp_all + fp_all else 0.0,
        "per_sample": per_sample,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--ref-key", default="reference")
    ap.add_argument("--pred-key", default="prediction")
    args = ap.parse_args()

    with open(args.results, encoding="utf-8") as f:
        records = json.load(f)
    m = evaluate(records, args.ref_key, args.pred_key)
    for k in ["micro_precision", "micro_recall", "micro_f1", "exact_match_rate", "hallucination_rate"]:
        print(f"{k:20s}: {m[k]:.3f}")
    print(f"TP/FP/FN           : {m['tp']}/{m['fp']}/{m['fn']}  (n={m['n']})")


if __name__ == "__main__":
    main()
