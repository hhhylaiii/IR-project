# Information Retrieval — Project 1

## Project Structure

```
Project 1/
├── main.py
├── VectorSpace.py
├── Parser.py
├── ParserZH.py
├── PorterStemmer.py
├── Eval.py
├── util.py
├── english.stop
├── chinese.stop
├── dict.txt.big
├── EnglishNews/
├── ChineseNews/
└── smaller_dataset/
    ├── collections/
    ├── queries/
    └── rel.tsv
```

## Environment & Installation

**Requirements**
- Python 3.8+
- OS: Windows (WSL/Ubuntu) or Linux
- All file paths are relative to the project root.

**Install dependencies**
```bash
pip install numpy jieba nltk
```

**NLTK data (for Task 3 only)**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
# For newer NLTK versions:
# python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
```

## Command-line Interface

- `--en_query "<text>"` : English retrieval
- `--ch_query "<text>"` : Chinese retrieval
- `--feedback` : Pseudo relevance feedback (requires `--en_query`)
- `--eval` : Run evaluation (MRR@10, MAP@10, Recall@10)

> If none of `--en_query`, `--ch_query`, or `--eval` is provided, the program exits with an error.

## Tasks, Usage & Outputs

### Task 1 — English Retrieval (TF / TF-IDF; Cosine & Euclidean)

**Run**
```bash
python main.py --en_query "airline fuel price"
```

**Output**
- Top-10 by: TF Cosine, TF-IDF Cosine, TF Euclidean (similarity `1/(1+dist)`), TF-IDF Euclidean (distance; lower is better)

**Imports**
- `numpy`
- Local: `VectorSpace.py`, `Parser.py`, `PorterStemmer.py`, `util.py`, `english.stop`

### Task 2 — Chinese Retrieval (Jieba)

**Run**
```bash
python main.py --ch_query "音樂 科技"
```

**Output**
- Top-10 by Cosine (TF and TF-IDF)

**Imports**
- `jieba`
- Local: `ParserZH.py`, `VectorSpace.py`, `util.py`, `chinese.stop`, `dict.txt.big`

### Task 3 — Pseudo Relevance Feedback (English)

**Run**
```bash
python main.py --en_query "airline fuel price" --feedback
```

**Output**
- Re-ranked Top-10 by TF-IDF Cosine
- Formula: `q_new = x * q_orig + y * q_fb` (default `x=1.0`, `y=0.5`)

**Imports**
- `nltk`, `numpy`
- Local: `VectorSpace.py`, `Parser.py`, `PorterStemmer.py`, `util.py`, `english.stop`

### Task 4 — Evaluation (MRR@10, MAP@10, Recall@10)

**Run**
```bash
python main.py --eval
```

**Output**
- Prints MRR@10, MAP@10, Recall@10 using `smaller_dataset/`

**Imports**
- `numpy`
- Local: `Eval.py`, `VectorSpace.py`, `Parser.py`, `PorterStemmer.py`, `util.py`, `english.stop`

## Implementation Details

- **Indexing**
  - English: `Parser` (lowercasing, stopword removal, Porter stemming)
  - Chinese: `ParserZH` + Jieba (`dict.txt.big`, stopword removal)
  - Vocabulary and DF built from the collection

- **Weighting**
  - TF vectors
  - IDF: `idf = log(N / df)`
  - TF-IDF when `use_tfidf=True`

- **Scoring**
  - Cosine similarity (TF or TF-IDF)
  - Euclidean: `1/(1+dist)` similarity or raw distance
  - Results sorted and Top-10 printed

- **Relevance Feedback (Task 3)**
  - Take top-1 document, extract nouns/verbs (NLTK POS tagging)
  - Build feedback vector and combine with original query

- **Evaluation (Task 4)**
  - **MRR@10**: reciprocal rank of first relevant hit
  - **AP@10**: precision at relevant hits in top-10, normalized by `min(10, |Rel(q)|)`
  - **MAP@10**: mean of AP@10
  - **Recall@10**: `|top10 ∩ Rel(q)| / |Rel(q)|`

## Execution Times Summary

| Task  | Example Command | Time (s) |
|-------|-----------------|----------|
| Task 1 | `python main.py --en_query "..."` | `<TBD>` |
| Task 2 | `python main.py --ch_query "..."` | `<TBD>` |
| Task 3 | `python main.py --en_query "..." --feedback` | `<TBD>` |
| Task 4 | `python main.py --eval` | `<TBD>` |
