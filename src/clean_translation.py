
from pathlib import Path

INPUT_DIR = Path("output/translated")
OUTPUT_DIR = Path("output/translated_cleaned")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

replacements = {
    "Luxor": "Laksar",
    "fallen Pawni Ganga": "sacred River Ganga",
    "fallen Pawni Ganges": "sacred River Ganga",
    "fallen Pawani Ganga": "sacred River Ganga",
    "fallen Pawani Ganges": "sacred River Ganga",
    "Maharaj Shivi": "Maharaja Shivi",
    "seven shrines": "seven sacred cities",
}

for file in sorted(INPUT_DIR.glob("*.txt")):
    text = file.read_text(encoding="utf-8")

    # Remove NULL artifacts
    text = text.replace("NULL", "")

    # Apply contextual translation corrections
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Clean extra spaces
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    output_file = OUTPUT_DIR / file.name
    output_file.write_text(text, encoding="utf-8")

    print(f"Cleaned: {file.name}")

print("Translation cleanup completed!")
