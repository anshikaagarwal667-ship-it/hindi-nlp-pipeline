# Hindi NLP Pipeline

A Python-based Natural Language Processing pipeline for processing unpunctuated and shuffled Hindi text.

## Project Objectives

1. Restore punctuation in Hindi text.
2. Extract individual sentences.
3. Reconstruct the most likely original sentence order.
4. Group consecutive sentences into groups of five.
5. Translate the groups into English.

## Project Structure

- `data/` – Input Hindi text files.
- `src/` – Python implementation.
- `output/` – Generated results.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Hindi punctuation restoration model

## Model

Model: Horizon-Labs/punctuation-restoration-small

Model weights will be provided separately through Google Drive.

**Model download:** ## Model Files

The Hindi NLP pipeline uses the following pretrained models:

1. **Punctuation Restoration:** `Horizon-Labs/punctuation-restoration-small`
2. **Sentence Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
3. **Sentence Ordering:** `google-bert/bert-base-multilingual-cased`
4. **Hindi-to-English Translation:** `ai4bharat/indictrans2-indic-en-dist-200M`

### Google Drive
https://drive.google.com/drive/folders/13sbj37AD_YSRmP3zyd16Uyr55Gy2Ogip?usp=drive_link


