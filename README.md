# LLaVA 棋子识别领域微调（LoRA / QLoRA）

[English](README_EN.md) | 中文

对开源视觉-语言模型 **LLaVA-v1.5（7B / 13B）** 做参数高效微调（PEFT），让它能准确说出图片里**每个清晰可见棋子的颜色和类型**。这是一个细粒度视觉问答（VQA）的领域适配任务。

> UNSW COMP9444（Neural Networks and Deep Learning）26T2 小组项目，Group ReLU，5 人。
> 本人负责：数据划分、零样本基线、评测指标，并参与 LoRA / QLoRA 微调实验。
> `src/finetune/` 是本人按最终实验配置整理的独立训练和评测脚本，课程提交的团队 notebook 不在本仓库公开。

## 结果一览（held-out test，n = 30）

| 配置 | Precision | Recall | **F1** | 整图全对率 | 幻觉率 |
|---|---|---|---|---|---|
| 7B 零样本 | .247 | .365 | .295 | 6.7% | 75.3% |
| 7B QLoRA | .818 | .865 | .841 | 76.7% | 18.2% |
| 7B LoRA | .870 | .904 | .887 | 80.0% | 13.0% |
| 13B 零样本 | .314 | .519 | .391 | 23.3% | 68.6% |
| **13B QLoRA** | .906 | .923 | **.914** | **83.3%** | **9.4%** |
| 13B LoRA | .906 | .923 | .914 | 83.3% | 9.4% |

| 方法 | 训练时长 | 显存峰值 | Adapter 大小 |
|---|---|---|---|
| 7B QLoRA | 81.6 min | 5.32 GB | 36 MB |
| 7B LoRA | 53.5 min | 13.81 GB | 36 MB |
| 13B QLoRA | 98.9 min | **9.04 GB** | 54 MB |
| 13B LoRA | 65.7 min | 25.76 GB | 54 MB |

**结论**
1. **微调远比堆模型规模有效**：零样本从 7B 换到 13B，F1 只涨 0.10；微调能涨 0.52 – 0.59。
2. **13B 下 QLoRA 不掉点**：和 16-bit LoRA 精度完全持平，显存省了约 65%，代价是训练慢约 50%。
3. **总分相同 ≠ 错误相同**：两个 13B adapter 在各类别上的强弱不一样（QLoRA 车、马更强，LoRA 象、王更强），选型要看混淆结构，不能只看 Micro F1。

## 方法

- **底座**：LLaVA-1.5-7B / 13B，冻结视觉编码器、投影层和语言模型主体
- **Adapter**：LoRA 挂在 `q/k/v/o` 投影上，r = 8，α = 16，dropout = 0.05
- **QLoRA**：底座用 4-bit NF4 加双重量化，adapter 计算用 BF16
- **受控对照**：LoRA 和 QLoRA 之间只改底座精度这一个变量，数据、prompt、超参、种子（9444）、解码策略全部一致
- **损失**：只对 assistant 回答部分算 loss，图像、system、user 的 token 用 -100 mask 掉
- **解码**：贪心解码（`do_sample=False, num_beams=1`），保证可复现

## 本人负责的部分

| 模块 | 文件 | 说明 |
|---|---|---|
| 数据下载与清单 | [`src/download_chess.py`](src/download_chess.py) | 拉取 `Trelis/chess_pieces`，从 caption 自动抽取棋子类型和颜色 |
| 防泄漏数据划分 | [`src/make_split.py`](src/make_split.py) | 按"棋子 × 颜色"分层，**硬编码**切分保证全组用同一份考卷；近似重复的图强制放在同一侧 |
| 零样本基线（本地） | [`src/single_infer.py`](src/single_infer.py)、[`src/run_baseline.py`](src/run_baseline.py) | Apple Silicon（MPS）上 fp16 跑 LLaVA-7B |
| 零样本基线（云端） | [`notebooks/baseline_llava13b_colab.ipynb`](notebooks/baseline_llava13b_colab.ipynb) | 4-bit 加载 13B，T4 上约 9 GB 显存；自动定位共享云盘，组员开箱即跑 |
| 评测器 | [`src/eval_pairs.py`](src/eval_pairs.py)、[`src/score.py`](src/score.py) | 把自由文本解析成〈颜色-棋子〉集合，算 P/R/F1、整图全对率、幻觉率 |
| 训练集增强 | [`src/finetune/augment.py`](src/finetune/augment.py) | 只增强 train：每张原图加 25 个视角或光照变体，标签不变 |
| LoRA / QLoRA 微调 | [`src/finetune/train_lora.py`](src/finetune/train_lora.py) | `--mode lora / qlora` 一个开关切换底座精度，其余配置完全一致；只对回答算 loss；按 val loss 选最优 checkpoint；记录训练时长和显存峰值 |
| 微调后评测 | [`src/finetune/predict.py`](src/finetune/predict.py)、[`src/finetune/common.py`](src/finetune/common.py) | 零样本和微调后共用同一套推理和评分；用 Counter 计数，两个白兵能算成 2 |

## 微调流程

```
147 张源图 ──按源图划分──▶ train 90 / val 18 / test 30 / external 9
                │
        train 90 ──augment.py──▶ 2,340 条（原图 + 25 个变体）
                │
        train_lora.py --mode {lora | qlora}
          · 冻结视觉塔、投影层和 LLM，只在 LLM 的 q/k/v/o 上挂 LoRA（r=8，α=16）
          · 可训练参数：7B 838 万 / 13B 1,311 万（约占全模型 0.1%）
          · 3 epoch，lr 2e-4，等效 batch 4，cosine 调度，3% warmup，梯度检查点
          · 每个 epoch 在 val 上评一次，取 val loss 最低的 checkpoint
                │
        predict.py ──▶ test 30 上贪心解码 ──▶ P / R / F1 / 整图全对率 / 幻觉率
```

几个容易踩的坑：
- **LoRA 的目标模块要限定在 `language_model` 里**。CLIP 视觉塔里也有 `q_proj / k_proj / v_proj`，不加限定会连视觉塔一起挂上 LoRA，可训练参数量就对不上了。
- **只对回答算 loss**。图像的 576 个 token 加上指令占了序列的绝大部分，不 mask 掉的话，模型会把大部分算力花在"复述提问"上。
- **选点看 val，不盲用最后一个 epoch**。实验里 7B QLoRA 在第 1 个 epoch 就到了最优，13B QLoRA 在第 3 个 epoch 才到，过拟合的节奏不一样。

### 早期发现：原版模型几乎不认识棋子

在第一阶段的 36 张单棋子图上，原版 LLaVA-7B 准确率只有 **13.9%**（6 类随机猜约 17%），回答几乎只有 `king` 和 `pawn`，8 个车里有 7 个被认成王。详见 [`results/baseline_7b_zero_shot.md`](results/baseline_7b_zero_shot.md)。正是这个结果让团队确定了"必须做领域微调"的方向。

## 数据

- 种子数据：[Trelis/chess_pieces](https://huggingface.co/datasets/Trelis/chess_pieces)（48 张）
- 团队另外自采 99 张，统一重标成 12 类（2 种颜色 × 6 种棋子），合计 **147 张源图**
- 划分：train 90 / val 18 / test 30 / external 9，**先按源图划分再做增强**，杜绝增强副本泄漏进测试集
- 训练集每张原图生成 25 个视觉变体，共 2,340 条训练记录
- 仓库里不包含图片，请从 Hugging Face 下载

## 快速开始

```bash
git clone https://github.com/haotian-liu/LLaVA.git && pip install -e LLaVA
pip install -r requirements.txt

python src/download_chess.py                         # 下载数据 → data/
python src/make_split.py                             # 生成固定划分
python src/single_infer.py data/chess_images/000.jpg "What chess piece is shown?"
python src/run_baseline.py && python src/score.py    # 跑零样本基线并评分
python src/eval_pairs.py results.json                # 配对级评测

# 微调（CUDA GPU，13B LoRA 需约 26 GB 显存，QLoRA 约 9 GB）
cd src/finetune
python augment.py --manifest ../../data/train.json --image-root ../../data --out-dir ../../data/train_aug
python train_lora.py --model llava-hf/llava-1.5-13b-hf --mode qlora \
    --train ../../data/train_aug/train_aug.json --val ../../data/val.json \
    --image-root ../../data --out ../../runs/qlora_13b
python predict.py --model llava-hf/llava-1.5-13b-hf --adapter ../../runs/qlora_13b/best_adapter \
    --load-4bit --test ../../data/test.json --image-root ../../data --out ../../runs/qlora_13b
```

manifest 格式：`[{"id": "t001", "image": "train/t001.jpg", "labels": "white pawn, black queen"}]`

## 局限

测试集小（30 张）、只用了一个种子和一种划分，没有置信区间；外部测试只有 9 张（F1 降到 0.839），说明模型遇到域偏移时仍会混淆车、马和后。任务只识别"有哪些棋子"，不做定位，也不还原整张棋盘。

## 参考

- Liu et al., *Visual Instruction Tuning*, NeurIPS 2023
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023
