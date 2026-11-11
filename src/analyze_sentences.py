
from pathlib import Path
from collections import Counter

folder = Path("output/sentences")

for file_path in sorted(folder.glob("*.txt")):
    sentences = [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    null_sentences = sum("NULL" in s for s in sentences)

    print(f"\n{file_path.name}")
    print("Total sentences:", len(sentences))
    print("Sentences containing NULL:", null_sentences)

    lengths = [len(s.split()) for s in sentences]
    print("Average words per sentence:", round(sum(lengths) / len(lengths), 2))

    print("First 3 sentences:")
    for sentence in sentences[:3]:
        print("-", sentence)
