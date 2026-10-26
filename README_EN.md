# Domain Fine-Tuning LLaVA for Chess Piece Recognition (LoRA / QLoRA)

English | [中文](README.md)

Parameter-efficient fine-tuning of **LLaVA-v1.5 (7B / 13B)** to name the **colour and type of every clearly visible chess piece** in an image: a fine-grained VQA domain-adaptation task.

> Group project (Group ReLU, 5 members) for UNSW COMP9444 *Neural Networks and Deep Learning*, 26T2.
> This repo contains **only my own contributions**: data splitting, zero-shot baselines and the evaluation metrics. The full training code lives in the course submission and is not published here.

## Results (held-out test, n = 30)

| Config | Precision | Recall | **F1** | Exact match | Hallucination |
|---|---|---|---|---|---|
| 7B zero-shot | .247 | .365 | .295 | 6.7% | 75.3% |
| 7B QLoRA | .818 | .865 | .841 | 76.7% | 18.2% |
| 7B LoRA | .870 | .904 | .887 | 80.0% | 13.0% |
| 13B zero-shot | .314 | .519 | .391 | 23.3% | 68.6% |
| **13B QLoRA** | .906 | .923 | **.914** | **83.3%** | **9.4%** |
| 13B LoRA | .906 | .923 | .914 | 83.3% | 9.4% |

| Method | Train time | Peak memory | Adapter |
|---|---|---|---|
| 7B QLoRA | 81.6 min | 5.32 GB | 36 MB |
| 7B LoRA | 53.5 min | 13.81 GB | 36 MB |
| 13B QLoRA | 98.9 min | **9.04 GB** | 54 MB |
| 13B LoRA | 65.7 min | 25.76 GB | 54 MB |

**Takeaways**
1. **Adaptation beats scale.** Going from 7B to 13B zero-shot adds only 0.10 F1; fine-tuning adds 0.52 to 0.59.
2. **QLoRA is lossless at 13B.** It matches BF16 LoRA exactly while using about 65% less peak memory, and trains about 50% slower.
3. **Equal Micro F1 ≠ equal errors.** The two 13B adapters have different per-class strengths: QLoRA is better on rooks and knights, LoRA on bishops and kings.

## Method

- Frozen vision tower, projector and LLM; LoRA on `q/k/v/o` (r = 8, α = 16, dropout = 0.05)
- QLoRA: 4-bit NF4 backbone with double quantisation, BF16 adapter compute
- Controlled comparison: base-weight precision is the only variable between LoRA and QLoRA (same data, prompt, hyper-parameters, seed 9444 and decoding)
- Loss computed on assistant tokens only (image, system and user tokens masked with -100); greedy decoding for reproducibility

## My contributions

| Module | Files | Notes |
|---|---|---|
| Data download & manifest | [`src/download_chess.py`](src/download_chess.py) | Pulls `Trelis/chess_pieces`, parses piece type and colour from captions |
| Leakage-safe split | [`src/make_split.py`](src/make_split.py) | Stratified by piece × colour, hard-coded so the whole team shares one test set; near-duplicates kept on the same side |
| Zero-shot baseline (local) | [`src/single_infer.py`](src/single_infer.py), [`src/run_baseline.py`](src/run_baseline.py) | LLaVA-7B in fp16 on Apple Silicon (MPS) |
| Zero-shot baseline (cloud) | [`notebooks/baseline_llava13b_colab.ipynb`](notebooks/baseline_llava13b_colab.ipynb) | 13B in 4-bit (about 9 GB on a T4); auto-locates the shared Drive folder for teammates |
| Evaluator | [`src/eval_pairs.py`](src/eval_pairs.py), [`src/score.py`](src/score.py) | Parses free text into (colour, piece) pairs and computes P/R/F1, exact match and hallucination rate |

Early finding: on 36 single-piece images, vanilla LLaVA-7B scored **13.9%**, below the 6-class chance level of about 17%. It answered almost only `king` or `pawn` (7 of 8 rooks were called kings). This finding set the team's fine-tuning direction. See [`results/baseline_7b_zero_shot.md`](results/baseline_7b_zero_shot.md).

## Data

The seed set is [Trelis/chess_pieces](https://huggingface.co/datasets/Trelis/chess_pieces) (48 images). The team collected 99 more and re-labelled everything into 12 classes, for **147 source images**. Splits are train 90 / val 18 / test 30 / external 9, made at the source level **before** augmentation; each training image gets 25 variants, giving 2,340 records. Images are not included in this repo.

## Quick start

```bash
git clone https://github.com/haotian-liu/LLaVA.git && pip install -e LLaVA
pip install -r requirements.txt
python src/download_chess.py && python src/make_split.py
python src/run_baseline.py && python src/score.py
python src/eval_pairs.py results.json
```

## Limitations

The test set is small (n = 30), with a single seed and split and no confidence intervals. On a 9-image external set, F1 drops to 0.839, so rooks, knights and queens are still confused under domain shift. The model recognises which pieces are present, with no localisation and no full-board reconstruction.
