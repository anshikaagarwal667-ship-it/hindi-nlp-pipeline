
from pathlib import Path
import spacy

nlp = spacy.load("xx_ent_wiki_sm")

input_file = Path("output/sentences/sample-1.txt")

sentences = [
    line.strip()
    for line in input_file.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

for i, sentence in enumerate(sentences, 1):
    doc = nlp(sentence)

    entities = [
        (ent.text, ent.label_)
        for ent in doc.ents
    ]

    if entities:
        print(f"\nSentence {i}: {sentence}")
        print("Entities:", entities)

