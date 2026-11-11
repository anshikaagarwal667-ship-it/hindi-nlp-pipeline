from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

input_folder = Path("output/sentences")
output_folder = Path("output/ordered")
output_folder.mkdir(parents=True, exist_ok=True)

print("Loading Hindi-compatible sentence model...")
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu"
)

for file_path in sorted(input_folder.glob("*.txt")):
    sentences = [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    embeddings = model.encode(
        sentences,
        normalize_embeddings=True
    )

    similarity = np.dot(embeddings, embeddings.T)

    # Try different starting sentences
    best_order = []
    best_score = float("-inf")

    for start in range(len(sentences)):
        visited = {start}
        order = [start]
        score = 0
        current = start

        while len(visited) < len(sentences):
            candidates = [
                i for i in range(len(sentences))
                if i not in visited
            ]

            next_sentence = max(
                candidates,
                key=lambda i: similarity[current][i]
            )

            score += similarity[current][next_sentence]
            order.append(next_sentence)
            visited.add(next_sentence)
            current = next_sentence

        if score > best_score:
            best_score = score
            best_order = order

    ordered_sentences = [sentences[i] for i in best_order]

    output_path = output_folder / file_path.name
    output_path.write_text(
        "\n".join(ordered_sentences),
        encoding="utf-8"
    )

    print(f"{file_path.name}: {len(ordered_sentences)} sentences ordered")

print("\nSentence ordering completed!")