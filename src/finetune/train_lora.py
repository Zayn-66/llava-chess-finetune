"""
LLaVA-1.5 LoRA / QLoRA 微调（Hugging Face transformers + PEFT）。

对照实验纪律：LoRA 和 QLoRA 只有 --mode 不同（底座 BF16 vs 4-bit NF4），
数据、指令、LoRA 配置、超参数、种子、选点规则完全一致。

用法（Colab A100）：
  python src/finetune/train_lora.py --model llava-hf/llava-1.5-13b-hf --mode qlora \
      --train data/train_aug/train_aug.json --val data/val.json --image-root . \
      --out runs/qlora_13b

输出：
  runs/<name>/best_adapter/   按验证集 loss 选出的最优 adapter（~36MB / ~54MB）
  runs/<name>/run_info.json   训练时长、显存峰值、最优 step、可训练参数量
"""
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration,
                          Trainer, TrainingArguments, set_seed)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from common import (LORA_ALPHA, LORA_DROPOUT, LORA_R, LORA_TARGET_REGEX, SEED, build_prompt)


# ---------------------------------------------------------------- 数据
class ChessSFTDataset(Dataset):
    def __init__(self, manifest: str, image_root: str):
        self.records = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.root = Path(image_root)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img_path = Path(r["image"])
        if not img_path.is_absolute():
            img_path = self.root / img_path
        return Image.open(img_path).convert("RGB"), r["labels"]


class AnswerOnlyCollator:
    """
    监督微调的关键：只对 assistant 的回答算 loss。
    图像 token、system/user 指令、padding 一律置为 -100，梯度不经过它们。
    做法：分别编码"完整对话"和"只有提问部分"，提问部分的长度就是要 mask 的前缀。
    """

    def __init__(self, processor):
        self.processor = processor
        self.eos = processor.tokenizer.eos_token

    def __call__(self, batch):
        images, answers = zip(*batch)
        full_texts = [build_prompt(a) + self.eos for a in answers]
        prompt_texts = [build_prompt() for _ in answers]

        enc = self.processor(images=list(images), text=full_texts,
                             return_tensors="pt", padding=True)
        prompt_enc = self.processor(images=list(images), text=prompt_texts,
                                    return_tensors="pt", padding=True)

        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        pad_left = self.processor.tokenizer.padding_side == "left"
        for i in range(labels.size(0)):
            plen = int(prompt_enc["attention_mask"][i].sum())
            start = int((enc["attention_mask"][i] == 0).sum()) if pad_left else 0
            labels[i, start:start + plen] = -100  # mask 掉 图像 + 指令
        enc["labels"] = labels
        return enc


# ---------------------------------------------------------------- 模型
def load_model(model_id: str, mode: str):
    if mode == "qlora":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # 4-bit NormalFloat
            bnb_4bit_use_double_quant=True,     # 双重量化，再省一点显存
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="auto")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:  # lora: BF16 底座
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                      target_modules=LORA_TARGET_REGEX, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    return model


# ---------------------------------------------------------------- 训练
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-hf/llava-1.5-13b-hf")
    ap.add_argument("--mode", choices=["lora", "qlora"], required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--image-root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    set_seed(SEED)
    out = Path(args.out)
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "right"

    model = load_model(args.model, args.mode)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {trainable:,}")

    train_ds = ChessSFTDataset(args.train, args.image_root)
    val_ds = ChessSFTDataset(args.val, args.image_root)

    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,         # 等效 batch = 4
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",                 # 每个 epoch 在 val 上评一次
        save_strategy="epoch",
        load_best_model_at_end=True,           # 按 val loss 选点，而不是盲用最后一个 epoch
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=args.epochs,
        logging_steps=20,
        remove_unused_columns=False,           # 保留 PIL 图像给 collator
        dataloader_num_workers=2,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
    )

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=AnswerOnlyCollator(processor))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60

    best_dir = out / "best_adapter"
    trainer.model.save_pretrained(best_dir)   # 只存 LoRA 权重
    processor.save_pretrained(best_dir)

    info = {
        "model": args.model, "mode": args.mode,
        "trainable_params": trainable,
        "train_minutes": round(minutes, 1),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_val_loss": trainer.state.best_metric,
        "log_history": trainer.state.log_history,
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "log_history"}, indent=2))


if __name__ == "__main__":
    main()
