from pathlib import Path

data_folder = Path("data")

for file_path in sorted(data_folder.glob("*.txt")):
    text = file_path.read_text(encoding="utf-8")

    print("\nFile:", file_path.name)
    print("Characters:", len(text))
    print("NULL entries:", text.count("NULL"))
    print("Words:", len(text.split()))
    print("Preview:", text[:200])
    print("-" * 50)
