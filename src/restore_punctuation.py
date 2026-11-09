
from pathlib import Path
from huggingface_hub import hf_hub_download
import importlib.util

# Load the punctuation restoration helper
model_name = "Horizon-Labs/punctuation-restoration-small"

helper_path = hf_hub_download(
    repo_id=model_name,
    filename="punctuate.py"
)

spec = importlib.util.spec_from_file_location(
    "punctuate", helper_path
)

punctuate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(punctuate)

# Load model once, then use it for all files
model = punctuate.Punctuator(model_name)

input_folder = Path("output/cleaned")
output_folder = Path("output/punctuated")
output_folder.mkdir(parents=True, exist_ok=True)

for file_path in sorted(input_folder.glob("*.txt")):

    print(f"\nProcessing {file_path.name}...")

    text = file_path.read_text(encoding="utf-8")

    # Restore punctuation
    result = model(text)

    # Save result
    output_path = output_folder / file_path.name
    output_path.write_text(result, encoding="utf-8")

    print("Original:", text[:300])
    print("Punctuated:", result[:300])
    print("Saved:", output_path)

print("\nPunctuation restoration completed!")
