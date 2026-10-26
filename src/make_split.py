"""
Split the 48 chess_pieces images into train / val / test folders.

Design notes
------------
- Task framing = "describe ALL pieces and their colours" (multi-label),
  scored later by F1 over (colour, piece) pairs. So multi-piece images
  (036-047) are kept and are valid samples in every split.
- Stratified by piece type AND colour: every split covers the 6 piece
  types and both colours as evenly as the tiny dataset allows. All 12
  (colour, piece) combos appear in TRAIN so the model can learn them,
  and TEST covers all 12 combos so the exam is complete.
- Near-identical shots are kept on the SAME side to avoid data leakage:
    036 / 037 / 038  (identical "white rook, white pawn, black queen, top view")
    046 / 047        (near-identical "white pawn, black queen, white knight")
    018 / 019        (near-identical "black pawn + black rook")
  -> all placed in TRAIN.
- The split is HARD-CODED (not random) so it is 100% reproducible and
  everyone on the team uses the exact same exam.

Output
------
  data/dataset/
    train/  *.jpg  + train.json
    val/    *.jpg  + val.json
    test/   *.jpg  + test.json
    split_summary.md
Each *.json is in LLaVA conversation format:
  {"id","image","conversations":[{human:"<image>\n<Q>"},{gpt:"<caption>"}]}
"""

import csv
import json
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC_IMAGES = BASE / "data" / "chess_images"
MANIFEST = BASE / "data" / "manifest.csv"
OUT = BASE / "data" / "dataset"

# Fixed question used for every sample (task = describe all pieces).
QUESTION = "What chess pieces are visible in this image? Describe their colours and piece types."

TEST_IDS = [2, 4, 8, 13, 22, 28, 31, 35, 39, 41, 43, 44]
VAL_IDS = [9, 16, 17, 23, 40, 42]
# TRAIN = everything not in TEST or VAL (computed below)


def load_captions():
    caps = {}
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            num = int(Path(row["file"]).stem)  # "000.jpg" -> 0
            caps[num] = row["caption"].strip()
    return caps


def build():
    captions = load_captions()
    all_ids = sorted(captions)  # 0..47
    test = set(TEST_IDS)
    val = set(VAL_IDS)
    train = [i for i in all_ids if i not in test and i not in val]

    assert len(test) == 12, len(test)
    assert len(val) == 6, len(val)
    assert len(train) == 30, len(train)
    assert not (test & val), "test/val overlap!"

    splits = {"train": train, "val": sorted(val), "test": sorted(test)}

    if OUT.exists():
        shutil.rmtree(OUT)

    summary_lines = ["# Split summary\n"]
    for name, ids in splits.items():
        split_dir = OUT / name
        split_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for i in ids:
            fname = f"{i:03d}.jpg"
            shutil.copy2(SRC_IMAGES / fname, split_dir / fname)
            records.append({
                "id": f"{i:03d}",
                "image": f"{name}/{fname}",
                "conversations": [
                    {"from": "human", "value": f"<image>\n{QUESTION}"},
                    {"from": "gpt", "value": captions[i]},
                ],
            })
        with open(split_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        multi = [i for i in ids if i >= 36]
        single = [i for i in ids if i < 36]
        summary_lines.append(
            f"\n## {name}  ({len(ids)} images: {len(single)} single + {len(multi)} multi)\n"
        )
        for i in ids:
            tag = "multi " if i >= 36 else "single"
            summary_lines.append(f"- {i:03d} [{tag}] {captions[i]}")

    with open(OUT / "split_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print("Done. Wrote:", OUT)
    for name, ids in splits.items():
        print(f"  {name:5s}: {len(ids)} images -> {OUT / name}")


if __name__ == "__main__":
    build()
