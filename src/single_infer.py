"""
Mac(MPS)版单图推理：拿一张棋子图问 LLaVA，确认整条链路能跑通。
用法：
    python single_infer.py data/chess_images/000.jpg "What chess piece is shown in this image? Answer with one word."
首次运行会自动下载 llava-v1.5-7b 模型（约 14GB），请耐心等待。
"""
import sys
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from PIL import Image

MODEL_PATH = "liuhaotian/llava-v1.5-7b"
DEVICE = "mps"


def main():
    image_file = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else "What is shown in this image?"

    disable_torch_init()
    model_name = get_model_name_from_path(MODEL_PATH)

    print("加载模型中（首次会下载约14GB）...")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_PATH, None, model_name, device=DEVICE
    )

    # 拼接提示词：图片占位符 + 问题
    qs = DEFAULT_IMAGE_TOKEN + "\n" + query
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # 处理图片
    image = Image.open(image_file).convert("RGB")
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
            do_sample=False,          # 贪心解码，结果可复现
            max_new_tokens=64,
            use_cache=True,
        )

    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print("\n===== 结果 =====")
    print("图片 :", image_file)
    print("问题 :", query)
    print("回答 :", answer)


if __name__ == "__main__":
    main()
