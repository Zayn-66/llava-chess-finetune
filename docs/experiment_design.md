# LLaVA 棋子领域微调 —— 完整实验设计

> 目标：对官方 LLaVA-v1.5-13B 在国际象棋棋子数据上做 LoRA / QLoRA 微调，
> 对比"零样本基线 vs 微调后"以及"LoRA vs QLoRA"的效果。

---

## 评测问题（baseline / LoRA / QLoRA 三边都用同一句）

```
What chess pieces are visible in this image? Describe their colours and piece types in one sentence.
```

中文：这张图里有哪些棋子？用一句话说清它们的颜色和类型。

- 评分沿用 notebook 里的 `extract_chess_pairs`：把回答解析成 `〈颜色+棋子〉` 集合，
  和标准答案集合比对，算 **精确率 / 召回率 / F1** + **整图全对率**。
  比"含关键词就算对"严谨，能抓出"颜色认错"。

---

## 0. 固定项（全程不变，保证可比）

| 项目 | 取值 |
|---|---|
| 底座模型 | `liuhaotian/llava-v1.5-13b` |
| 数据划分 | train 30 / val 6 / test 12（见 `data/dataset/split_summary.md`，冻结） |
| 评测问题 | 上面那句，**baseline 和微调后完全相同** |
| 解码 | 贪心（`do_sample=False, num_beams=1`），保证可复现 |
| 评分 | `extract_chess_pairs` → 配对级 P/R/F1 + 整图全对率 |
| 随机种子 | 9444（train / data seed 都固定） |
| 标准答案 | 从 caption 自动生成，**残缺棋子（portion / half）不计入** |

> **对照实验纪律**：LoRA 与 QLoRA 之间，**唯一允许不同的变量是"底座 4bit(QLoRA) vs 16bit(LoRA)"**。
> 数据、epoch、学习率、LoRA rank/alpha/dropout、target modules、种子、评测口径必须完全一致，
> 否则无法判断差距来自量化还是别的因素。

---

## 1. 数据准备（Colab 上做一次）

- 下载 `Trelis/chess_pieces`，按 `split_summary.md` 的编号切成 train 30 / val 6 / test 12。
- 转成官方 LLaVA 训练格式 JSON：
  ```json
  {"id": "...", "image": "xxx.jpg",
   "conversations": [
     {"from": "human", "value": "<image>\n{问题}"},
     {"from": "gpt",   "value": "{caption}"}
   ]}
  ```
  - **训练集**：可用多种问法轮换做轻度增强，让模型不死记一种问法。
  - **val / test**：**只用那一句评测问题**，保证口径统一。
- 产出 `train.json / val.json / test.json` + 每张图的标准答案（残缺棋子不计入）。

## 2. Baseline（13B 零样本）

- 加载官方原版 13B，不训练，直接在 **test 12 张**上跑评测问题 → `baseline_results.json`。
- 算 P/R/F1、整图全对率 → 这是**共同基准线**，LoRA 和 QLoRA 都跟它比。

## 3. 训练（两条线并行，除量化外全同）

对每条线（**Exp-A: QLoRA** / **Exp-B: LoRA**）：

1. 载入底座（QLoRA = `--bits 4`；LoRA = 默认 16bit），挂同一套 LoRA 配置
   （同 rank / alpha / dropout / target modules）。
2. 在 **train 30** 上训练，`num_train_epochs=5`，每个 epoch 存一次 checkpoint。
3. **每个 epoch 在 val 6 上评一次**，用 val 表现（loss 或配对 F1）挑最优存档点、防过拟合。
4. 保存 best adapter，并用 `merge_lora_weights.py` 合并。

> **val 的作用**：数据只有 30 张，几乎必然过拟合。val 替你决定"停在第几个 epoch 的存档"。
> test 全程不碰，只在第 4 步最后测一次。

## 4. 复测（同一份考卷）

- 分别加载 Baseline / Exp-A(QLoRA) / Exp-B(LoRA)，在**同一份 test 12** 上跑同一问题、同一评分。
- 产出三份 results + 一张对比表。

## 5. 结果对比

| 模型 | Precision | Recall | F1 | 整图全对率 | 显存峰值 | 训练耗时 |
|---|---|---|---|---|---|---|
| Baseline(13B) | | | | | — | — |
| QLoRA | | | | | | |
| LoRA | | | | | | |

关注两点：
1. **微调 vs baseline** 提升多少；
2. **QLoRA vs LoRA** 精度差多少、各自省多少显存 / 时间。

## 6. 后续（数据补充后）

数据量上来后再做**消融**：
- 纯 chess 数据 vs chess + 通用数据混训（测灾难性遗忘）；
- 或换问题模板，测鲁棒性。

---

## 三份数据分工（一句话）

**train 30 → 学；val 6 → 挑最优存档点、防过拟合；test 12 → 最后一次性判分。**
