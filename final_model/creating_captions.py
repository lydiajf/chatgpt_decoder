from datasets import load_dataset
from pathlib import Path

ds = load_dataset("nlphuji/flickr30k")
text = ""

for split in ds:
    row_texts = [string for row in ds[split]["caption"] for string in row]
    text += "\n".join(row_texts)

with open(
    Path(__file__).parent.parent / "final_model/all_text.txt", "w+", encoding="utf-8"
) as f:
    f.write(text)