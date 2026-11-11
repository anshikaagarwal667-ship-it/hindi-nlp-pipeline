
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

INPUT_FOLDER = Path("output/sentences")
OUTPUT_FOLDER = Path("output/clustered")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu"
)

for file_path in sorted(INPUT_FOLDER.glob("*.txt")):
    sentences = [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    embeddings = model.encode(
        sentences,
        normalize_embeddings=True
    )

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.40,
        metric="cosine",
        linkage="average"
    )

    labels = clustering.fit_predict(embeddings)

    groups = {}

    for sentence, label in zip(sentences, labels):
        groups.setdefault(label, []).append(sentence)

    output = ""

    for i, group in enumerate(groups.values(), 1):
        output += f"\nTOPIC {i}\n"
        output += "\n".join(group) + "\n"

    (OUTPUT_FOLDER / file_path.name).write_text(
        output,
        encoding="utf-8"
    )

    print(file_path.name, "Topics:", len(groups))

print("Topic clustering completed!")
