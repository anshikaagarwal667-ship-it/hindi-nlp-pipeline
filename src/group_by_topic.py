
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

INPUT_FOLDER = Path("output/sentences")
OUTPUT_FOLDER = Path("output/topic_groups")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Similarity threshold: higher means stricter topic grouping.
DISTANCE_THRESHOLD = 0.45

print("Loading semantic model...")
model = SentenceTransformer(MODEL_NAME, device="cpu")

for file_path in sorted(INPUT_FOLDER.glob("*.txt")):
    print(f"\nProcessing {file_path.name}")

    sentences = [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(sentences) < 2:
        continue

    embeddings = model.encode(
        sentences,
        normalize_embeddings=True
    )

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=DISTANCE_THRESHOLD
    )

    labels = clustering.fit_predict(embeddings)

    groups = {}

    for sentence, label in zip(sentences, labels):
        groups.setdefault(label, []).append(sentence)

    output_file = OUTPUT_FOLDER / file_path.name

    with output_file.open("w", encoding="utf-8") as f:
        for group_number, (label, group) in enumerate(
            sorted(groups.items()), start=1
        ):
            f.write(f"TOPIC {group_number}\n")
            for sentence in group:
                f.write(sentence + "\n")
            f.write("\n")

    print("Total sentences:", len(sentences))
    print("Topics found:", len(groups))
    print("Saved:", output_file)

print("\nTopic grouping completed!")

