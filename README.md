# AmbiStory SemEval 2026 Task 5 — Upgraded Model

This repository is a submission-safe AmbiStory plausibility predictor.

The evaluator calls:

```bash
python predict.py <input_json> <output_jsonl>
```

`predict.py` writes one JSON object per line:

```json
{"id": "42", "prediction": 3}
```

where `prediction` is an integer from 1 to 5.

## What changed from the starter baseline

The previous starter was a single sense-aware TF-IDF + Ridge-style regressor. This version keeps that baseline and adds the six planned upgrades:

1. **Baseline preserved** — the original local TF-IDF model is kept as `base_model.joblib` and is wrapped by the final model.
2. **Hyperparameter/calibration tuning** — `train.py` tunes group contrast and integer thresholds against the dev labels. It can also train extra Ridge/SVR/SGD candidates with `--train_extra_models`.
3. **Threshold tuning** — final integer outputs use tuned cut-points instead of naive rounding.
4. **Group-aware ranking/contrast** — samples with the same `precontext + sentence + ending` are treated as competing candidate senses. Their group mean is preserved, but the within-group difference is amplified.
5. **Ensemble-ready architecture** — `AmbiStoryEnsemble` can combine several local models with weights. The bundled final model currently selects the strongest calibrated baseline because the extra local candidates did not beat it on dev.
6. **Optional transformer path** — `train_transformer_optional.py` gives a local HuggingFace cross-encoder training route. It is not required for `predict.py` and is kept out of the main dependencies so the submitted model remains lightweight and runnable without external APIs.

## Current dev metrics

From the bundled `metrics.json`:

| Version | Spearman continuous | Spearman integer | Accuracy within stdev | MAE |
|---|---:|---:|---:|---:|
| Starter baseline | 0.1961 | 0.1741 | 0.6344 | 1.0000 |
| Upgraded calibrated model | 0.2255 | 0.2800 | 0.6633 | 1.0084 |

The MAE is slightly worse, but the task prioritizes rank correlation and accuracy-within-standard-deviation, so this is a better submission strategy.

## Files

```text
predict.py                    Required evaluation entry point
model.py                      Feature extraction, base model wrapper, ensemble, calibrators
train.py                      Rebuild/tune final model.joblib
model.joblib                  Final calibrated model used by predict.py
base_model.joblib             Original baseline kept for reproducibility
metrics.json                  Dev evaluation for final model
requirements.txt              Main dependencies
train_transformer_optional.py Optional local transformer training script
requirements-transformer.txt  Optional transformer-only dependencies
```

## Run prediction

```bash
python predict.py dev.json dev_predictions.jsonl
```

## Rebuild the final model

Fast default, using the bundled baseline and tuning the post-processing:

```bash
python train.py --train_json train.json --dev_json dev.json --output model.joblib
```

Optional extra local candidates:

```bash
python train.py --train_json train.json --dev_json dev.json --output model.joblib --train_extra_models
```

## Optional transformer experiment

This is not needed for the submitted zip. Use it only on a machine with the optional dependencies installed:

```bash
pip install -r requirements-transformer.txt
python train_transformer_optional.py \
  --train_json train.json \
  --dev_json dev.json \
  --model_name distilroberta-base \
  --output_dir transformer_checkpoint
```

The transformer should be treated as an additional ensemble candidate, not as a replacement for the calibrated classical model.
