
from pathlib import Path
import re

folder = Path("output/punctuated")

for file_path in sorted(folder.glob("*.txt")):
    text = file_path.read_text(encoding="utf-8")

    # Count sentences using predicted punctuation
    sentences = re.split(r"(?<=[।?!])\s+", text.strip())
    sentences = [s for s in sentences if s]

    print(f"\n{'=' * 60}")
    print("FILE:", file_path.name)
    print("Sentence count:", len(sentences))
    print("NULL count:", text.count("NULL"))

    # Show sentences containing NULL
    for sentence in sentences:
        if "NULL" in sentence:
            print("NULL sentence:", sentence)

    # Show first 5 predicted sentences
    print("\nFirst 5 sentences:")
    for sentence in sentences[:5]:
        print("-", sentence)
