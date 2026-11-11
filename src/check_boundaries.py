
from pathlib import Path
import re

for file_path in sorted(Path("output/sentences").glob("*.txt")):
    lines = file_path.read_text(encoding="utf-8").splitlines()

    print(f"\n{file_path.name}")

    for i, sentence in enumerate(lines, 1):
        ends_with_punctuation = re.search(r"[।?!]$", sentence.strip())

        if len(sentence.split()) <= 3 or not ends_with_punctuation:
            print(f"{i}: {sentence}")
