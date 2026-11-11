from pathlib import Path
import re
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel, BertForNextSentencePrediction
from sklearn.cluster import AgglomerativeClustering

INPUT = Path("output/sentences")
ORDERED = Path("output/topic_ordered")
CLUSTERS = Path("output/topic_clusters")
GROUPED = Path("output/topic_grouped")

for folder in (ORDERED, CLUSTERS, GROUPED):
    folder.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"
BATCH = 16
BEAM_WIDTH = 100
CLUSTER_DISTANCE = 0.42

SEMANTIC_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
NSP_MODEL = "google-bert/bert-base-multilingual-cased"

print("Loading existing models...")

emb_tokenizer = AutoTokenizer.from_pretrained(SEMANTIC_MODEL)
emb_model = AutoModel.from_pretrained(SEMANTIC_MODEL).to(DEVICE)
emb_model.eval()

nsp_tokenizer = AutoTokenizer.from_pretrained(NSP_MODEL)
nsp_model = BertForNextSentencePrediction.from_pretrained(NSP_MODEL).to(DEVICE)
nsp_model.eval()


def get_embeddings(sentences):
    vectors = []

    for start in range(0, len(sentences), BATCH):
        batch = sentences[start:start + BATCH]
        inputs = emb_tokenizer(
            batch, padding=True, truncation=True,
            max_length=256, return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            output = emb_model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (output * mask).sum(1) / mask.sum(1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        vectors.extend(pooled.cpu().numpy())

    return np.asarray(vectors)


def get_nsp_scores(sentences):
    n = len(sentences)
    scores = np.zeros((n, n), dtype=float)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

    for start in range(0, len(pairs), BATCH):
        batch = pairs[start:start + BATCH]
        first = [sentences[i] for i, j in batch]
        second = [sentences[j] for i, j in batch]

        inputs = nsp_tokenizer(
            first, second, padding=True, truncation=True,
            max_length=256, return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            logits = nsp_model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[:, 0]

        for k, (i, j) in enumerate(batch):
            scores[i, j] = probs[k].item()

    return scores


def order_cluster(indices, sentences, embeddings, nsp):
    if len(indices) <= 2:
        return indices

    local_sem = embeddings[indices] @ embeddings[indices].T
    local_nsp = nsp[np.ix_(indices, indices)]
    m = len(indices)

    # NSP and semantic compatibility are considered together.
    scores = 0.65 * local_nsp + 0.35 * ((local_sem + 1) / 2)

    beam = [(0.0, [i], {i}) for i in range(m)]

    for _ in range(m - 1):
        candidates = []

        for total, path, visited in beam:
            current = path[-1]

            for nxt in range(m):
                if nxt not in visited:
                    candidates.append((
                        total + scores[current, nxt],
                        path + [nxt],
                        visited | {nxt}
                    ))

        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = candidates[:BEAM_WIDTH]

    return [indices[i] for i in beam[0][1]]


def write_groups(sentences, filename):
    output = []
    for start in range(0, len(sentences), 5):
        output.append(f"GROUP {start // 5 + 1}")
        output.extend(sentences[start:start + 5])
        output.append("")

    (GROUPED / filename).write_text(
        "\n".join(output), encoding="utf-8"
    )


for file in sorted(INPUT.glob("*.txt")):
    print(f"\nProcessing {file.name}")

    sentences = [
        line.strip()
        for line in file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not sentences:
        print("No sentences found; skipping.")
        continue

    embeddings = get_embeddings(sentences)

    if len(sentences) == 1:
        labels = np.array([0])
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=CLUSTER_DISTANCE,
            metric="cosine",
            linkage="average"
        )
        labels = clustering.fit_predict(embeddings)

    nsp = get_nsp_scores(sentences)

    clusters = {}
    for index, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(index)

    # Keep topic clusters in order of their first occurrence.
    topic_lists = sorted(clusters.values(), key=lambda x: min(x))

    final_indices = []
    cluster_report = []

    for number, indices in enumerate(topic_lists, 1):
        ordered_indices = order_cluster(
            indices, sentences, embeddings, nsp
        )
        final_indices.extend(ordered_indices)

        cluster_report.append(
            f"TOPIC {number} ({len(indices)} sentences)"
        )
        for index in ordered_indices:
            cluster_report.append(f"[{index}] {sentences[index]}")
        cluster_report.append("")

    final_sentences = [sentences[i] for i in final_indices]

    # Verify that no sentence was lost or duplicated.
    assert len(final_indices) == len(sentences)
    assert sorted(final_indices) == list(range(len(sentences)))

    (ORDERED / file.name).write_text(
        "\n".join(final_sentences), encoding="utf-8"
    )

    (CLUSTERS / file.name).write_text(
        "\n".join(cluster_report), encoding="utf-8"
    )

    write_groups(final_sentences, file.name)

    print(f"Sentences: {len(sentences)}")
    print(f"Detected topics: {len(topic_lists)}")
    print(f"Saved ordered sentences: {ORDERED / file.name}")
    print(f"Saved topic report: {CLUSTERS / file.name}")
    print(f"Saved groups of five: {GROUPED / file.name}")

print("\nTopic-based ordering completed.")
