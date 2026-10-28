"""
训练 / 推理共用的配置与工具函数。

标注格式（manifest JSON，列表）：
  [{"id": "t001", "image": "train/t001.jpg", "labels": "white pawn, black queen"}, ...]
  - labels：逗号分隔的规范标签，12 类 = {white, black} × {pawn, knight, bishop, rook, queen, king}
  - 残缺（只露出一部分）的棋子不写进 labels
"""
from collections import Counter
import re

COLOURS = ("white", "black")
PIECES = ("pawn", "knight", "bishop", "rook", "queen", "king")
CLASSES = [f"{c} {p}" for c in COLOURS for p in PIECES]

# 训练 / 验证 / 测试统一使用同一条指令，保证三边口径一致
INSTRUCTION = (
    "Identify every clearly visible chess piece in this image. "
    "Ignore pieces that are only partially visible. "
    "Answer only with a comma-separated list of labels in the form "
    "'<colour> <piece>', e.g. 'white pawn, black queen'."
)

SEED = 9444

# LoRA 只挂在语言模型的注意力投影上（q/k/v/o），不动视觉塔和投影层。
# 注意：CLIP 视觉塔里也有 q_proj/k_proj/v_proj，所以必须用正则限定在 language_model 内，
# 否则可训练参数会多出一截（7B 正确值 = 8,388,608；13B = 13,107,200）。
LORA_TARGET_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


def build_prompt(answer: str | None = None) -> str:
    """llava-hf 1.5 的对话模板：USER: <image>\\n{指令} ASSISTANT: {回答}"""
    prompt = f"USER: <image>\n{INSTRUCTION} ASSISTANT:"
    return prompt if answer is None else f"{prompt} {answer}"


_LABEL_RE = re.compile(
    rf"\b({'|'.join(COLOURS)})\s+(?:chess\s+)?({'|'.join(PIECES)})s?\b"
)


def parse_labels(text: str) -> Counter:
    """把回答规范化成多重集合 Counter。用 Counter 而不是 set，两个白兵能数成 2。"""
    return Counter(f"{c} {p}" for c, p in _LABEL_RE.findall(text.lower()))


def score(golds: list[Counter], preds: list[Counter]) -> dict:
    """Micro P/R/F1、整图全对率、幻觉率 = FP / (TP + FP)，以及每类 F1。"""
    tp = fp = fn = exact = 0
    per_cls = {c: Counter() for c in CLASSES}
    for g, p in zip(golds, preds):
        inter = g & p
        tp += sum(inter.values())
        fp += sum((p - g).values())
        fn += sum((g - p).values())
        exact += g == p
        for c in CLASSES:
            per_cls[c]["tp"] += inter[c]
            per_cls[c]["fp"] += (p - g)[c]
            per_cls[c]["fn"] += (g - p)[c]

    def f1(t, f_p, f_n):
        pr = t / (t + f_p) if t + f_p else 0.0
        rc = t / (t + f_n) if t + f_n else 0.0
        return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)

    p, r, f = f1(tp, fp, fn)
    return {
        "n": len(golds), "tp": tp, "fp": fp, "fn": fn,
        "micro_precision": p, "micro_recall": r, "micro_f1": f,
        "exact_match": exact / len(golds) if golds else 0.0,
        "hallucination_rate": fp / (tp + fp) if tp + fp else 0.0,
        "per_class_f1": {c: f1(v["tp"], v["fp"], v["fn"])[2] for c, v in per_cls.items()},
    }
