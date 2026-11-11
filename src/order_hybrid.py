
from pathlib import Path
import re
import torch
import numpy as np

from transformers import (
    AutoTokenizer,
    BertForNextSentencePrediction
)
from sentence_transformers import SentenceTransformer

INPUT_FOLDER = Path("output/sentences")
OUTPUT_FOLDER = Path("output/hybrid_ordered")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
BEAM_WIDTH = 30

print("Loading models...")

# Model 1: Semantic similarity
semantic_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu"
)

# Model 2: Next Sentence Prediction
tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-multilingual-cased"
)

nsp_model = BertForNextSentencePrediction.from_pretrained(
    "google-bert/bert-base-multilingual-cased"
)

nsp_model.eval()

# Words that provide clues about sentence continuation
connectors = [
    "इसलिए", "अतः", "तब", "लेकिन",
    "इसके बाद", "इसके अलावा", "फिर",
    "हालांकि", "इस प्रकार", "उधर"
]

stopwords = {
    "है", "हैं", "था", "थे", "की", "का",
    "के", "और", "से", "में", "पर", "को",
    "यह", "वह", "भी", "तो", "ही", "एक",
    "कि", "ने", "हो", "कर", "लिए"
}


def get_words(sentence):
    words = set(re.findall(r"[\u0900-\u097F]+", sentence))
    return words - stopwords - {"NULL"}


def overlap_score(a, b):
    words_a = get_words(a)
    words_b = get_words(b)

    if not words_a or not words_b:
        return 0.0

    return len(words_a & words_b) / len(words_a | words_b)


def has_connector(sentence):
    sentence = sentence.strip()

    return any(
        sentence.startswith(word)
        for word in connectors
    )


def calculate_nsp_scores(sentences):
    n = len(sentences)
    scores = np.zeros((n, n))

    pairs = [
        (i, j)
        for i in range(n)
        for j in range(n)
        if i != j
    ]

    for start in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[start:start + BATCH_SIZE]

        first = [sentences[i] for i, j in batch]
        second = [sentences[j] for i, j in batch]

        inputs = tokenizer(
            first,
            second,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )

        with torch.no_grad():
            logits = nsp_model(**inputs).logits
            probabilities = torch.softmax(logits, dim=1)

        for k, (i, j) in enumerate(batch):
            scores[i, j] = probabilities[k, 0].item()

        print(
            f"NSP: {min(start + BATCH_SIZE, len(pairs))}"
            f"/{len(pairs)} pairs"
        )

    return scores


def build_combined_scores(sentences):
    print("Calculating semantic embeddings...")

    embeddings = semantic_model.encode(
        sentences,
        normalize_embeddings=True
    )

    semantic_scores = np.dot(embeddings, embeddings.T)

    print("Calculating NSP scores...")

    nsp_scores = calculate_nsp_scores(sentences)

    n = len(sentences)
    combined = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            semantic = (semantic_scores[i, j] + 1) / 2
            nsp = nsp_scores[i, j]
            overlap = overlap_score(sentences[i], sentences[j])

            # A sentence beginning with a connector
            # is more likely to follow another sentence.
            combined[i, j] = (
                0.50 * nsp
                + 0.30 * semantic
                + 0.20 * overlap
            )

    return combined


def find_best_order(scores):
    n = len(scores)

    beam = [
        (0.0, [start], {start})
        for start in range(n)
    ]

    for step in range(n - 1):
        candidates = []

        for total, path, visited in beam:
            current = path[-1]

            for nxt in range(n):
                if nxt not in visited:
                    new_score = total + scores[current, nxt]

                    candidates.append((
                        new_score,
                        path + [nxt],
                        visited | {nxt}
                    ))

        candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        beam = candidates[:BEAM_WIDTH]

        print(f"Ordering: {step + 1}/{n - 1}")

    return beam[0][1]


for file_path in sorted(INPUT_FOLDER.glob("*.txt")):

    print(f"\nProcessing {file_path.name}")

    sentences = [
        line.strip()
        for line in file_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    scores = build_combined_scores(sentences)

    order = find_best_order(scores)

    ordered_sentences = [
        sentences[i] for i in order
    ]

    output_file = OUTPUT_FOLDER / file_path.name

    output_file.write_text(
        "\n".join(ordered_sentences),
        encoding="utf-8"
    )

    print("Saved:", output_file)

print("\nHybrid ordering completed!")
