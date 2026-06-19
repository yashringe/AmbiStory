"""Prediction entry point required by the AmbiStory evaluator.

"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib


def load_input(input_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = list(data.keys())
    samples = [data[k] for k in ids]
    return ids, samples


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python predict.py <input_json> <output_jsonl>")

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    model_path = Path(__file__).resolve().parent / "model.joblib"

    ids, samples = load_input(input_path)

    if model_path.exists():
        model = joblib.load(model_path)
        if hasattr(model, "predict_int"):
            predictions = model.predict_int(samples)
        else:
            predictions = model.predict(samples)
    else:
        # Last-resort fallback. Real submission should include model.joblib.
        predictions = [3 for _ in samples]

    with open(output_path, "w", encoding="utf-8") as f:
        for sample_id, pred in zip(ids, predictions):
            pred = int(max(1, min(5, round(float(pred)))))
            f.write(json.dumps({"id": str(sample_id), "prediction": pred}) + "\n")


if __name__ == "__main__":
    main()
