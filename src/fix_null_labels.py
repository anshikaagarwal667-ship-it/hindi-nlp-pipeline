
from pathlib import Path
import re

input_folder = Path("output/punctuated")
output_folder = Path("output/null_checked")

output_folder.mkdir(parents=True, exist_ok=True)

for file_path in sorted(input_folder.glob("*.txt")):

    text = file_path.read_text(encoding="utf-8")

    # Remove only the colon incorrectly attached to NULL
    text = re.sub(r"\bNULL\s*:", "NULL", text)

    output_path = output_folder / file_path.name
    output_path.write_text(text, encoding="utf-8")

    print("Processed:", file_path.name)
    print("Remaining NULL:", text.count("NULL"))

print("\nNULL label correction completed!")
