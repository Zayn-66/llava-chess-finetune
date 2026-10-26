# 第一阶段：Zero-shot Baseline 总结

## 实验设置

| 项目 | 内容 |
|------|------|
| 模型 | `liuhaotian/llava-v1.5-7b`（原版，未微调） |
| 运行环境 | MacBook Pro M4 Pro 24GB，MPS（Apple GPU），float16 推理 |
| 解码方式 | 贪心解码（do_sample=False），结果可复现 |
| 测试集 | chess_pieces 前 36 张**单棋子特写**图（000–035.jpg） |
| 任务 | 只考棋子种类（6 类：queen/king/knight/bishop/rook/pawn） |
| 统一问题 | "What chess piece is shown in this image? Answer with one word." |
| 评分规则 | 关键词匹配：模型回答里含正确棋子名即算对 |

## 结果

**总准确率：5 / 36 = 13.9%**

### 各棋子准确率

| 棋子 | 准确率 |
|------|--------|
| king | 4/6 = 67% |
| pawn | 1/3 = 33% |
| queen | 0/7 = 0% |
| knight | 0/6 = 0% |
| bishop | 0/6 = 0% |
| rook | 0/8 = 0% |

### 模型的混淆情况（真实 → 预测）

| 真实棋子 | 模型认成了 |
|----------|-----------|
| queen | king×4, pawn×3 |
| king | king×4, pawn×2 |
| knight | pawn×3, rook×2, king×1 |
| bishop | pawn×4, king×2 |
| rook | king×7, pawn×1 |
| pawn | king×2, pawn×1 |

## 关键发现

1. **原版模型基本不会认棋子**：13.9% 接近随机水平（6 类随机 ≈ 17%）。
2. **模型有强烈偏见**：回答几乎只出现 `king` 和 `pawn`，完全不认识 knight/bishop/queen/rook。
3. **只有 king 相对可识别**（67%），推测因其造型（顶部十字）辨识度最高。
4. **结论**：通用 LLaVA 对棋子这一细分领域几乎无能，为后续领域微调留下巨大提升空间。

## 产出文件

- `data/test_set.json` — 考卷（36 题），三组实验共用，**已冻结不再改动**
- `data/baseline_results.json` — 模型逐题回答
- `data/baseline_summary.md` — 本文件
- 脚本：`download_chess.py` / `make_testset.py` / `run_baseline.py` / `single_infer.py` / `score.py`

## 下一步（需上云 GPU）

- 实验 2：LoRA/QLoRA 微调（chess_pieces + 部分 llava-instruct-mix），用**同一份考卷**复测，对比准确率提升。
- 实验 3：消融（纯 chess vs 混合数据），验证混入通用数据是否防止“灾难性遗忘”。
- 复测时直接复用 `run_baseline.py`（改模型路径）+ `score.py`。
