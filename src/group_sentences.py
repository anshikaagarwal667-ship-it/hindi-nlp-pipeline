from pathlib import Path

input_folder = Path("output/hybrid_ordered")
output_folder = Path("output/grouped")
output_folder.mkdir(parents=True, exist_ok=True)

for file_path in sorted(input_folder.glob("*.txt")):
    sentences = [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    groups = [
        sentences[i:i + 5]
        for i in range(0, len(sentences), 5)
    ]

    output = ""

    for i, group in enumerate(groups, 1):
        output += f"\nGROUP {i}\n"
        output += "\n".join(group)
        output += "\n"

    output_path = output_folder / file_path.name
    output_path.write_text(output, encoding="utf-8")

    print(f"{file_path.name}: {len(groups)} groups")

print("\nGrouping completed!")