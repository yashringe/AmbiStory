# AmbiStory Plausibility Modeling

A lightweight NLP regression pipeline for estimating how well a candidate word meaning fits within an ambiguous short narrative. The project combines lexical text representations, calibrated regression, threshold tuning, and group-aware post-processing to produce stable plausibility predictions on a 1–5 scale.

The implementation is designed to be easy to run locally, reproducible, and transparent enough to inspect or extend.

\---

## Project Overview

The model receives short narrative samples containing an ambiguous word and a candidate meaning. It builds separate representations for the story context and the candidate sense, extracts lexical and similarity-based features, predicts a continuous plausibility score, and converts it into an integer rating from 1 to 5.

The final prediction flow is:

```text
Input JSON
  → Text construction
  → Feature extraction
  → Regression model
  → Calibration and thresholding
  → Group-aware post-processing
  → JSONL predictions
```

\---

## Repository Structure

```text
.
├── predict.py                     # Main prediction entry point
├── train.py                       # Training and tuning script
├── model.py                       # Feature engineering, model classes, utilities
├── model.joblib                   # Trained model used by predict.py
├── base\_model.joblib              # Preserved baseline model
├── requirements.txt               # Core Python dependencies
├── requirements-transformer.txt   # Optional transformer dependencies
├── train\_transformer\_optional.py  # Optional transformer training script
├── metrics.json                   # Development-set metrics from the latest run
├── dev\_predictions.jsonl          # Example prediction output
├── report\_notes.md                # Notes useful for reporting/evaluation
└── README.md                      # Project documentation
```

Recommended data placement:

```text
.
├── train.json
├── dev.json
└── test.json
```

The scripts also support full file paths, so the data does not have to be stored inside the repository folder.

\---

## Environment Setup

### 1\. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2\. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

\---

## Running Predictions

Use `predict.py` with an input JSON file and an output JSONL path:

```bash
python predict.py <input\_json> <output\_jsonl>
```

Example using the development file:

```bash
python predict.py dev.json dev\_predictions.jsonl
```

Example with full Windows paths:

```bash
python predict.py "R:\\NLP\\AmbiStory\\dev.json" "R:\\NLP\\AmbiStory\\dev\_predictions.jsonl"
```

The output file is written in JSONL format:

```json
{"id": "0", "prediction": 3}
{"id": "1", "prediction": 4}
{"id": "2", "prediction": 2}
```

Each line contains:

|Field|Description|
|-|-|
|`id`|Sample id from the input JSON|
|`prediction`|Integer plausibility score from 1 to 5|

\---

## Retraining the Model

To retrain the model with the default configuration:

```bash
python train.py --train\_json train.json --dev\_json dev.json
```

To train with additional candidate models enabled:

```bash
python train.py --train\_json train.json --dev\_json dev.json --train\_extra\_models
```

To manually specify output paths:

```bash
python train.py ^
  --train\_json train.json ^
  --dev\_json dev.json ^
  --output model.joblib ^
  --metrics\_output metrics.json ^
  --predictions\_output dev\_predictions.jsonl
```

On macOS/Linux, use backslashes instead of `^` for line continuation:

```bash
python train.py \\
  --train\_json train.json \\
  --dev\_json dev.json \\
  --output model.joblib \\
  --metrics\_output metrics.json \\
  --predictions\_output dev\_predictions.jsonl
```

Training updates or creates:

```text
model.joblib
metrics.json
dev\_predictions.jsonl
```

\---

## Input Format

The input file should be a JSON object where each key is a sample id and each value is a sample object.

Example:

```json
{
  "0": {
    "homonym": "track",
    "judged\_meaning": "a pair of parallel rails providing a runway for wheels",
    "precontext": "The detectives arrived at the abandoned train station. They were looking for signs of the missing artifact. A faint trail caught their attention.",
    "sentence": "They followed the track.",
    "ending": "They began to run along the abandoned railway line.",
    "example\_sentence": "The train glided smoothly along the track."
  }
}
```

For training, the sample should also contain the gold fields used for supervision, such as:

```text
average
stdev
choices
```

For prediction, the model mainly uses the text fields and sample ids.

\---

## Model Details

The implemented system uses a calibrated regression approach.

Main components:

|Component|Purpose|
|-|-|
|Word-level TF-IDF|Captures important words and short phrases|
|Character-level TF-IDF|Handles morphology, spelling variation, and unseen forms|
|Lexical overlap features|Measures direct overlap between story and sense text|
|Similarity-style features|Captures relation between context, ending, meaning, and example|
|Ridge-style regression|Predicts a continuous plausibility score|
|Threshold tuning|Converts continuous scores into integer labels|
|Group-aware post-processing|Adjusts competing sense predictions within the same story context|

The model is intentionally lightweight and local-first. It does not require external API calls.

\---

## Optional Transformer Extension

An optional transformer training script is included for experimentation on GPU-enabled environments.

Install optional dependencies:

```bash
pip install -r requirements-transformer.txt
```

Example run:

```bash
python train\_transformer\_optional.py \\
  --train\_json train.json \\
  --dev\_json dev.json \\
  --model\_name roberta-base \\
  --epochs 6 \\
  --batch\_size 8 \\
  --learning\_rate 2e-5 \\
  --max\_length 320
```

This path is intended for further experimentation. The default local pipeline remains the classical calibrated regression model stored in `model.joblib`.

\---

## Common Commands

### Run prediction

```bash
python predict.py dev.json dev\_predictions.jsonl
```

### Retrain model

```bash
python train.py --train\_json train.json --dev\_json dev.json
```

### Retrain with extra models

```bash
python train.py --train\_json train.json --dev\_json dev.json --train\_extra\_models
```

### Inspect metrics

Windows:

```bash
type metrics.json
```

macOS/Linux:

```bash
cat metrics.json
```

\---

## Reproducibility Checklist

Before running the project, verify that the repository contains:

```text
predict.py
train.py
model.py
model.joblib
requirements.txt
```

For retraining, also verify that the data files are available:

```text
train.json
dev.json
```

After prediction, verify that the output file exists and contains one JSON object per line:

```json
{"id": "sample\_id", "prediction": 3}
```

\---

## Limitations and Future Work

The current model is strong for a lightweight lexical-regression pipeline, but it has limitations. It may struggle when the intended meaning is implied indirectly rather than through visible word overlap. It can also be conservative and predict middle scores for highly ambiguous examples.

Future improvements could include:

* fine-tuning a transformer cross-encoder,
* training on individual annotator ratings instead of only the mean score,
* adding pairwise ranking loss for competing senses,
* combining transformer predictions with the existing calibrated regression model.

