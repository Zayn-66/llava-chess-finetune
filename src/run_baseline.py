"""
步骤4：批量跑 zero-shot baseline。
读 data/test_set.json（考卷），让原版 LLaVA 逐题回答，
结果存到 data/baseline_results.json。
模型只加载一次，然后循环推理 36 张图。

用法：python run_baseline.py
"""
import os
import json
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from PIL import Image

BASE = os.path.dirname(__file__)
IMG_DIR = os.path.join(BASE, "data", "chess_images")
TEST_SET = os.path.join(BASE, "data", "test_set.json")
OUT = os.path.join(BASE, "data", "baseline_results.json")

MODEL_PATH = "liuhaotian/llava-v1.5-7b"
DEVICE = "mps"


def infer_one(model, tokenizer, image_processor, image_path, query):
    qs = DEFAULT_IMAGE_TOKEN + "\n" + query
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config).to(
        model.device, dtype=torch.float16
    )
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=False,        # 贪心解码，结果可复现
            max_new_tokens=64,
            use_cache=True,
        )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def main():
    with open(TEST_SET, encoding="utf-8") as f:
        questions = json.load(f)

    disable_torch_init()
    model_name = get_model_name_from_path(MODEL_PATH)
    print("加载模型中...")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_PATH, None, model_name, device=DEVICE
    )

    results = []
    for i, q in enumerate(questions):
        img_path = os.path.join(IMG_DIR, q["file"])
        answer = infer_one(model, tokenizer, image_processor, img_path, q["question"])
        results.append({
            "file": q["file"],
            "question": q["question"],
            "answer_piece": q["answer_piece"],   # 标准答案
            "model_answer": answer,               # 模型回答
        })
        print(f"[{i+1}/{len(questions)}] {q['file']} | 标答: {q['answer_piece']:8s} | 模型: {answer}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n完成 ✅ 结果已存到: {OUT}")


if __name__ == "__main__":
    main()
