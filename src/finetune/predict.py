"""
用底座（零样本）或底座 + LoRA adapter 在测试集上推理并评分。
贪心解码（do_sample=False, num_beams=1, max_new_tokens=64），三组实验完全同一口径。

用法：
  # 零样本 baseline（BF16）
  python src/finetune/predict.py --model llava-hf/llava-1.5-13b-hf --test data/test.json --out runs/base_13b
  # 微调后
  python src/finetune/predict.py --model llava-hf/llava-1.5-13b-hf --adapter runs/qlora_13b/best_adapter \
      --load-4bit --test data/test.json --out runs/qlora_13b
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

from common import build_prompt, parse_labels, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA adapter 用 4-bit 底座推理")
    ap.add_argument("--test", required=True)
    ap.add_argument("--image-root", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    kw = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if args.load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = LlavaForConditionalGeneration.from_pretrained(args.model, **kw)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)

    records = json.loads(Path(args.test).read_text(encoding="utf-8"))
    rows, golds, preds = [], [], []
    for r in records:
        p = Path(r["image"])
        img = Image.open(p if p.is_absolute() else Path(args.image_root) / p).convert("RGB")
        enc = processor(images=img, text=build_prompt(), return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, do_sample=False, num_beams=1, max_new_tokens=64)
        answer = processor.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        g, pr = parse_labels(r["labels"]), parse_labels(answer)
        golds.append(g)
        preds.append(pr)
        rows.append({"id": r["id"], "reference": r["labels"], "prediction": answer,
                     "gold": dict(g), "pred": dict(pr), "exact": g == pr})
        print(f"{r['id']:>8} | {'✓' if g == pr else '✗'} | ref: {r['labels']} | pred: {answer}")

    metrics = score(golds, preds)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for k in ["micro_precision", "micro_recall", "micro_f1", "exact_match", "hallucination_rate"]:
        print(f"{k:20s}: {metrics[k]:.3f}")


if __name__ == "__main__":
    main()
