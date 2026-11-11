
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit import IndicProcessor

MODEL = "ai4bharat/indictrans2-indic-en-dist-200M"
INPUT_DIR = Path("output/grouped")
OUTPUT_DIR = Path("output/translated")
BATCH_SIZE = 8

print("Loading IndicTrans2...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL, trust_remote_code=True
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL, trust_remote_code=True
)
model.eval()

processor = IndicProcessor(inference=True)


def translate_batch(texts):
    if not texts:
        return []

    prepared = processor.preprocess_batch(
        texts,
        src_lang="hin_Deva",
        tgt_lang="eng_Latn"
    )

    inputs = tokenizer(
        prepared,
        truncation=True,
        padding="longest",
        return_tensors="pt"
    )

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            num_beams=5,
            max_length=256
        )

    decoded = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    return processor.postprocess_batch(
        decoded,
        lang="eng_Latn"
    )


def translate_file(input_path, output_path):
    lines = input_path.read_text(encoding="utf-8").splitlines()

    # Store each line's parts and collect Hindi text for translation.
    line_parts = []
    all_texts = []

    for line in lines:
        if not line.strip() or line.strip().startswith("GROUP "):
            line_parts.append([("keep", line)])
            continue

        # Preserve NULL markers while translating the remaining text.
        parts = re.split(r"(\[?NULL\]?)", line)
        current_parts = []

        for part in parts:
            if not part:
                continue

            if re.fullmatch(r"\[?NULL\]?", part):
                current_parts.append(("keep", part))
            elif part.strip():
                index = len(all_texts)
                all_texts.append(part.strip())
                current_parts.append(("translate", index))
            else:
                current_parts.append(("keep", part))

        line_parts.append(current_parts)

    # Translate in small batches.
    translations = []

    for start in range(0, len(all_texts), BATCH_SIZE):
        batch = all_texts[start:start + BATCH_SIZE]
        translations.extend(translate_batch(batch))

        print(
            f"Translated {min(start + BATCH_SIZE, len(all_texts))}"
            f"/{len(all_texts)} sentences"
        )

    # Reconstruct the original group structure.
    output_lines = []

    for parts in line_parts:
        translated_parts = []

        for kind, value in parts:
            if kind == "keep":
                translated_parts.append(value)
            else:
                translated_parts.append(translations[value])

        output_lines.append(" ".join(
            part.strip() for part in translated_parts if part.strip()
        ))

    output_path.write_text(
        "\n".join(output_lines) + "\n",
        encoding="utf-8"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob("sample-*.txt"))

    if not files:
        print("No input files found!")
        return

    for input_path in files:
        output_path = OUTPUT_DIR / input_path.name

        print(f"\nTranslating {input_path.name}...")
        translate_file(input_path, output_path)
        print(f"Saved: {output_path}")

    print("\nAll translations completed!")


if __name__ == "__main__":
    main()
