
from pathlib import Path
import re

input_folder = Path("output/null_checked")
output_folder = Path("output/sentences")
output_folder.mkdir(parents=True, exist_ok=True)

for file_path in sorted(input_folder.glob("*.txt")):
    text = file_path.read_text(encoding="utf-8")

    # Fix the missing boundary before the Ganga Mela heading
    text = text.replace(
        "चलें गढ़ मुक्तेश्वर का गंगा मेला लक्सर से",
        "चलें गढ़ मुक्तेश्वर का गंगा मेला। लक्सर से"
    )

    # Split at sentence-ending punctuation
    sentences = re.split(r"(?<=[।?!])\s+", text.strip())

    sentences = [s.strip() for s in sentences if s.strip()]

    output_path = output_folder / file_path.name
    output_path.write_text("\n".join(sentences), encoding="utf-8")

    print(f"{file_path.name}: {len(sentences)} sentences extracted")

print("\nSentence extraction completed!")
