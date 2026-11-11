
from pathlib import Path
import re
import torch
import numpy as np

from transformers import AutoTokenizer, BertForNextSentencePrediction
from sentence_transformers import SentenceTransformer

INPUT_FILE = Path("output/sentences/sample-1.txt")
BATCH_SIZE = 16

print("Loading models...")

semantic_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu"
)

tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-multilingual-cased"
)

nsp_model = BertForNextSentencePrediction.from_pretrained(
    "google-bert/bert-base-multilingual-cased"
)
nsp_model.eval()

stopwords = {
    "है", "हैं", "था", "थे", "की", "का", "के",
    "और", "से", "में", "पर", "को", "यह", "वह",
    "भी", "तो", "ही", "एक", "कि", "ने", "हो", "कर", "लिए"
}

def get_words(sentence):
    words = set(re.findall(r"[\u0900-\u097F]+", sentence))
    return words - stopwords - {"NULL"}

def overlap_score(a, b):
    wa = get_words(a)
    wb = get_words(b)

    if not wa or not wb:
        return 0.0

    return len(wa & wb) / len(wa | wb)


sentences = [
    line.strip()
    for line in INPUT_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

print("Total sentences:", len(sentences))

print("Calculating embeddings...")
embeddings = semantic_model.encode(
    sentences,
    normalize_embeddings=True
)

anchors = [
    "भगवान राम के पूर्वज",
    "चेंगलपट स्टेशन से अरकोनम जाने वाली लाइन",
    "गढ़मुक्तेश्वर हिंदुओं"
]

for keyword in anchors:

    matches = [
        i for i, sentence in enumerate(sentences)
        if keyword in sentence
    ]

    if not matches:
        print(f"\nAnchor not found: {keyword}")
        continue

    i = matches[0]
    anchor = sentences[i]

    candidates = [
        j for j in range(len(sentences)) if j != i
    ]

    first = [anchor] * len(candidates)
    second = [sentences[j] for j in candidates]

    nsp_values = []

    for start in range(0, len(candidates), BATCH_SIZE):
        batch_first = first[start:start + BATCH_SIZE]
        batch_second = second[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch_first,
            batch_second,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )

        with torch.no_grad():
            logits = nsp_model(**inputs).logits
            probabilities = torch.softmax(logits, dim=1)

        nsp_values.extend(probabilities[:, 0].tolist())

    results = []

    for k, j in enumerate(candidates):
        semantic = (
            np.dot(embeddings[i], embeddings[j]) + 1
        ) / 2

        overlap = overlap_score(sentences[i], sentences[j])
        nsp = nsp_values[k]

        combined = (
            0.50 * nsp
            + 0.30 * semantic
            + 0.20 * overlap
        )

        results.append(
            (combined, nsp, semantic, overlap, j)
        )

    results.sort(reverse=True)

    print("\n" + "=" * 60)
    print("ANCHOR:", anchor)
    print("\nTop 5 possible next sentences:")

    for score, nsp, semantic, overlap, j in results[:5]:
        print(f"\nSentence index: {j}")
        print(f"Combined: {score:.3f} | NSP: {nsp:.3f} | "
              f"Semantic: {semantic:.3f} | Overlap: {overlap:.3f}")
        print(sentences[j])

print("\nDiagnostic completed!")
