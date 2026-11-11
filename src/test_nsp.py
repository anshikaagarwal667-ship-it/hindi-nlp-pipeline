
import torch
from transformers import AutoTokenizer, BertForNextSentencePrediction

MODEL_NAME = "google-bert/bert-base-multilingual-cased"

print("Loading multilingual BERT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = BertForNextSentencePrediction.from_pretrained(MODEL_NAME)
model.eval()

def get_score(sentence_a, sentence_b):
    inputs = tokenizer(
        sentence_a,
        sentence_b,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)

    return probabilities[0][0].item()

pairs = [
    (
        "हिंदू धर्म में तीर्थ का बड़ा महत्व है।",
        "गढ़मुक्तेश्वर हिंदुओं का पावन तीर्थ है।"
    ),
    (
        "अयोध्या भगवान श्री राम के अवतार की पवित्र भूमि है।",
        "अयोध्या पवित्र नदी सरयू के तट पर स्थित है।"
    )
]

for a, b in pairs:
    forward = get_score(a, b)
    backward = get_score(b, a)

    print("\nSentence A:", a)
    print("Sentence B:", b)
    print(f"A -> B: {forward:.4f}")
    print(f"B -> A: {backward:.4f}")

print("\nNSP test completed!")
