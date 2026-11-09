
from pathlib import Path
import unicodedata
import re

data_folder = Path("data")
output_folder = Path("output/cleaned")

output_folder.mkdir(parents=True, exist_ok=True)

for file_path in sorted(data_folder.glob("*.txt")):

    # Read original file
    text = file_path.read_text(encoding="utf-8")

    # Normalize Unicode characters
    text = unicodedata.normalize("NFC", text)

    # Replace multiple spaces/newlines with a single space
    text = re.sub(r"\s+", " ", text).strip()

    # Save cleaned text separately
    output_path = output_folder / file_path.name
    output_path.write_text(text, encoding="utf-8")

    print(f"\nProcessed: {file_path.name}")
    print("Characters:", len(text))
    print("NULL entries:", text.count("NULL"))

    # Display context around each NULL
    for match in re.finditer(r"\bNULL\b", text):
        start = max(0, match.start() - 45)
        end = min(len(text), match.end() + 45)
        print("NULL context:", text[start:end])

print("\nPreprocessing completed!")
