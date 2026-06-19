# NLP Project 2026 – Opinion vs Fact Classification

A two-stage NLP pipeline for classifying sentences as **opinion** or **fact**.


### Install required packages

```bash
pip install -r requirements.txt
```


## Usage

### Run Stage 1

Stage 1 uses traditional NLP techniques and machine learning models.

```bash
python -m stage1.stage1_pipeline --train data/dataset.csv --test data/validationset.csv
```

**Output:** `group55_classifications_1.csv`

---

### Run Stage 2

Stage 2 uses `facebook/bart-large-mnli` (zero-shot baseline) and a fine-tuned
`facebook/bart-base` (main method) for LLM-based classification.

```bash
python -m stage2.stage2_pipeline --train data/dataset.csv --test data/validationset.csv
```

**Output:** `group55_classifications_2.csv`, plus confusion matrices, a
baseline-vs-main comparison chart, and a fine-tuning learning curve in `results/`.

---

### Run Full Pipeline

Run both stages sequentially:

```bash
python run_all.py --train data/dataset.csv --test data/validationset.csv
```

This generates both classification files.

---
