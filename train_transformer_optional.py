"""Optional local transformer cross-encoder training script.

This file is not required by predict.py. It is included so the project has a
clean upgrade path from the bundled TF-IDF ensemble to a DeBERTa/RoBERTa-style
cross-encoder when a machine with HuggingFace + GPU/CPU budget is available.

Example:
    pip install -r requirements-transformer.txt
    python train_transformer_optional.py --train_json train.json --dev_json dev.json \
        --model_name distilroberta-base --output_dir transformer_checkpoint

The resulting checkpoint can be used to produce dev/test scores and then be
ensembled with the classical model through train.py after adding those scores as
an extra base predictor. It uses no external LLM API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_input(sample: dict) -> str:
    ending = sample.get("ending") or "[NO_ENDING]"
    return (
        f"Story: {sample.get('precontext','')} {sample.get('sentence','')} {ending}\n"
        f"Target word: {sample.get('homonym','')}\n"
        f"Candidate sense: {sample.get('judged_meaning','')}\n"
        f"Example: {sample.get('example_sentence','')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", default="train.json")
    parser.add_argument("--dev_json", default="dev.json")
    parser.add_argument("--model_name", default="distilroberta-base")
    parser.add_argument("--output_dir", default="transformer_checkpoint")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        from torch.utils.data import Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise SystemExit(
            "Optional transformer dependencies are missing. Install with: "
            "pip install -r requirements-transformer.txt"
        ) from exc

    def load(path: str):
        data = json.load(open(path, encoding="utf-8"))
        return list(data.values())

    train_samples = load(args.train_json)
    dev_samples = load(args.dev_json)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    class AmbiDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples
            self.texts = [build_input(s) for s in samples]
            self.labels = [float(s["average"]) for s in samples]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            enc = tokenizer(
                self.texts[idx],
                max_length=args.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
            return item

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1, problem_type="regression")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.clip(preds.reshape(-1), 1, 5)
        labels = labels.reshape(-1)
        try:
            from scipy.stats import spearmanr
            rho = spearmanr(labels, preds).correlation
        except Exception:
            rho = 0.0
        return {"spearman": float(0.0 if np.isnan(rho) else rho), "mae": float(np.mean(np.abs(preds - labels)))}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="spearman",
        greater_is_better=True,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=AmbiDataset(train_samples),
        eval_dataset=AmbiDataset(dev_samples),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    Path(args.output_dir, "README.txt").write_text(
        "Local transformer checkpoint for AmbiStory regression. Use as an optional ensemble member.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
