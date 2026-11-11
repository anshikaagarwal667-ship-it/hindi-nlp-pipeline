
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, BertForNextSentencePrediction

MODEL_NAME = "google-bert/bert-base-multilingual-cased"

INPUT_FILE = Path("output/sentences/sample-1.txt")
OUTPUT_FOLDER = Path("output/nsp_ordered")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
BEAM_WIDTH = 20

print("Loading multilingual BERT NSP model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = BertForNextSentencePrediction.from_pretrained(MODEL_NAME)
model.eval()

sentences = [
    line.strip()
    for line in INPUT_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

n = len(sentences)
print(f"Total sentences: {n}")

# Calculate directional scores for every pair
scores = np.full((n, n), -np.inf)

pairs = [
    (i, j)
    for i in range(n)
    for j in range(n)
    if i != j
]

for start in range(0, len(pairs), BATCH_SIZE):
    batch_pairs = pairs[start:start + BATCH_SIZE]

    first = [sentences[i] for i, j in batch_pairs]
    second = [sentences[j] for i, j in batch_pairs]

    inputs = tokenizer(
        first,
        second,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)

    for k, (i, j) in enumerate(batch_pairs):
        # Log odds of the pair being consecutive
        p = probabilities[k, 0].item()
        p = np.clip(p, 1e-7, 1 - 1e-7)
        scores[i, j] = np.log(p / (1 - p))

    print(f"Processed {min(start + BATCH_SIZE, len(pairs))}/{len(pairs)} pairs")

# Beam search for a high-scoring sequence
beam = [
    (0.0, [start], {start})
    for start in range(n)
]

for step in range(n - 1):
    candidates = []

    for total_score, path, visited in beam:
        current = path[-1]

        for nxt in range(n):
            if nxt not in visited:
                new_score = total_score + scores[current, nxt]
                candidates.append(
                    (new_score, path + [nxt], visited | {nxt})
                )

    candidates.sort(key=lambda x: x[0], reverse=True)
    beam = candidates[:BEAM_WIDTH]

    print(f"Ordering step: {step + 1}/{n - 1}")

best_path = beam[0][1]

ordered = [sentences[i] for i in best_path]

output_file = OUTPUT_FOLDER / INPUT_FILE.name
output_file.write_text("\n".join(ordered), encoding="utf-8")

print("\nNSP ordering completed!")
print("Saved to:", output_file)
