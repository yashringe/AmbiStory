"""Train, tune and package the upgraded AmbiStory model.

Usage:
    python train.py --train_json train.json --dev_json dev.json --output model.joblib

The final stack keeps the TF-IDF baseline, tunes hyperparameters, applies
within-story contrast calibration, tunes integer thresholds, and saves one
``model.joblib`` consumed by ``predict.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.svm import LinearSVR

from model import AmbiStoryEnsemble, AmbiStoryVectorizer, ClippedRoundedRegressor, GroupContrastCalibrator, ThresholdCalibrator


def load_json_samples(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = list(data.keys())
    samples = [data[k] for k in ids]
    return ids, samples


def labels(samples: Sequence[Dict[str, object]]) -> np.ndarray:
    return np.asarray([float(sample["average"]) for sample in samples], dtype=np.float32)


def stdevs(samples: Sequence[Dict[str, object]]) -> np.ndarray:
    return np.asarray([float(sample.get("stdev", 1.0)) for sample in samples], dtype=np.float32)


def accuracy_within_stdev(y_true: np.ndarray, pred_int: np.ndarray, sd: np.ndarray) -> float:
    return float(np.mean(np.abs(pred_int - y_true) <= np.maximum(sd, 1.0)))


def evaluate(y_true: np.ndarray, continuous: np.ndarray, sd: np.ndarray, pred_int: np.ndarray | None = None) -> Dict[str, float]:
    continuous = np.clip(np.asarray(continuous, dtype=np.float32), 1.0, 5.0)
    if pred_int is None:
        pred_int = np.rint(continuous).clip(1, 5).astype(int)
    rho_cont = spearmanr(y_true, continuous).correlation
    rho_int = spearmanr(y_true, pred_int).correlation
    return {
        "spearman_continuous": float(0.0 if np.isnan(rho_cont) else rho_cont),
        "spearman_integer": float(0.0 if np.isnan(rho_int) else rho_int),
        "accuracy_within_stdev": accuracy_within_stdev(y_true, pred_int, sd),
        "mae": float(np.mean(np.abs(continuous - y_true))),
    }


def selection_score(metrics: Dict[str, float]) -> float:
    # Spearman is the main task metric; accuracy-within-stdev is secondary.
    return metrics["spearman_integer"] + 0.15 * metrics["accuracy_within_stdev"]


def make_model(config: Dict[str, object]) -> ClippedRoundedRegressor:
    vectorizer = AmbiStoryVectorizer(
        word_ngram_max=int(config["word_ngram_max"]),
        word_max_features=int(config["word_max_features"]),
        char_max_features=int(config["char_max_features"]),
        ending_max_features=int(config["ending_max_features"]),
        pair_max_features=int(config["pair_max_features"]),
        min_df=int(config["min_df"]),
        use_char=bool(config["use_char"]),
        use_pair_sims=bool(config["use_pair_sims"]),
    )

    kind = str(config["regressor"])
    if kind == "ridge":
        regressor = Ridge(alpha=float(config["alpha"]), solver="lsqr", max_iter=500, tol=1e-2)
    elif kind == "svr":
        regressor = LinearSVR(
            C=float(config["C"]),
            epsilon=float(config.get("epsilon", 0.1)),
            dual="auto",
            max_iter=2500,
            random_state=42,
        )
    else:
        regressor = SGDRegressor(
            loss=str(config.get("loss", "huber")),
            alpha=float(config["alpha"]),
            epsilon=float(config.get("epsilon", 0.3)),
            penalty="elasticnet",
            l1_ratio=float(config.get("l1_ratio", 0.05)),
            max_iter=4000,
            tol=1e-4,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        )
    return ClippedRoundedRegressor(vectorizer=vectorizer, regressor=regressor)


def candidate_configs() -> List[Dict[str, object]]:
    return [
        {
            "name": "ridge_medium_alpha10",
            "regressor": "ridge",
            "alpha": 10.0,
            "word_ngram_max": 2,
            "word_max_features": 12000,
            "char_max_features": 7000,
            "ending_max_features": 4000,
            "pair_max_features": 9000,
            "min_df": 2,
            "use_char": True,
            "use_pair_sims": True,
        },
        {
            "name": "ridge_conservative_alpha40",
            "regressor": "ridge",
            "alpha": 40.0,
            "word_ngram_max": 2,
            "word_max_features": 12000,
            "char_max_features": 7000,
            "ending_max_features": 4000,
            "pair_max_features": 9000,
            "min_df": 2,
            "use_char": True,
            "use_pair_sims": True,
        },
        {
            "name": "linear_svr_margin",
            "regressor": "svr",
            "C": 0.12,
            "epsilon": 0.1,
            "word_ngram_max": 2,
            "word_max_features": 9000,
            "char_max_features": 5000,
            "ending_max_features": 3000,
            "pair_max_features": 7000,
            "min_df": 2,
            "use_char": True,
            "use_pair_sims": True,
        },
        {
            "name": "sgd_huber_robust",
            "regressor": "sgd",
            "alpha": 0.0002,
            "loss": "huber",
            "epsilon": 0.35,
            "l1_ratio": 0.05,
            "word_ngram_max": 2,
            "word_max_features": 9000,
            "char_max_features": 5000,
            "ending_max_features": 3000,
            "pair_max_features": 6000,
            "min_df": 2,
            "use_char": True,
            "use_pair_sims": True,
        },
    ]


def tune_thresholds(y_true: np.ndarray, scores: np.ndarray, sd: np.ndarray) -> Tuple[Tuple[float, float, float, float], Dict[str, float]]:
    """Tune a small, stable set of integer cut-points.

    A full 4D threshold grid is unnecessary for this small dataset and can make
    retraining slow. These candidates include normal rounding, the best coarse
    families found during development, and score-quantile thresholds.
    """
    quantiles = tuple(float(x) for x in np.quantile(scores, [0.2, 0.4, 0.6, 0.8]))
    candidate_thresholds = [
        (1.5, 2.5, 3.5, 4.5),
        (1.5, 2.5, 3.75, 4.0),
        (1.5, 2.75, 3.5, 3.75),
        (1.75, 2.5, 2.75, 3.75),
        (1.75, 2.5, 3.0, 4.0),
        (2.0, 2.5, 3.0, 4.0),
        (2.0, 2.75, 3.25, 4.0),
        quantiles,
    ]

    best_thresholds = (1.5, 2.5, 3.5, 4.5)
    best_metrics = evaluate(y_true, scores, sd)
    best_score = selection_score(best_metrics)

    for thresholds in candidate_thresholds:
        if len(set(thresholds)) != 4 or any(thresholds[i] >= thresholds[i + 1] for i in range(3)):
            continue
        pred_int = ThresholdCalibrator(thresholds).predict_int(scores)
        metrics = evaluate(y_true, scores, sd, pred_int=pred_int)
        score = selection_score(metrics)
        if score > best_score:
            best_score = score
            best_thresholds = tuple(float(x) for x in thresholds)
            best_metrics = metrics
    return best_thresholds, best_metrics


def tune_group_and_thresholds(dev_samples, y_dev, raw_scores, sd_dev):
    best = None
    for scale in np.arange(0.9, 2.51, 0.2):
        contrasted = GroupContrastCalibrator(float(scale)).transform(dev_samples, raw_scores)
        thresholds, metrics = tune_thresholds(y_dev, contrasted, sd_dev)
        score = selection_score(metrics)
        record = {
            "group_scale": float(scale),
            "thresholds": thresholds,
            "metrics": metrics,
            "selection_score": float(score),
        }
        if best is None or score > best["selection_score"]:
            best = record
    return best


def tune_subset_ensemble(models, dev_predictions, dev_samples, y_dev, sd_dev):
    best = None
    n = len(models)

    # Keep the search intentionally small so training stays reproducible on a
    # normal laptop. We test every single model, the full equal-weight ensemble,
    # and every pair containing the first model, which is normally the strong
    # baseline loaded from base_model.joblib.
    candidate_subsets = {(i,) for i in range(n)}
    if n > 1:
        candidate_subsets.add(tuple(range(n)))
        for i in range(1, n):
            candidate_subsets.add((0, i))

    for idxs in sorted(candidate_subsets, key=lambda x: (len(x), x)):
        raw = np.mean([dev_predictions[i] for i in idxs], axis=0)
        calibration = tune_group_and_thresholds(dev_samples, y_dev, raw, sd_dev)
        score = calibration["selection_score"]
        record = {"indices": idxs, "calibration": calibration, "raw_metrics": evaluate(y_dev, raw, sd_dev)}
        if best is None or score > best["calibration"]["selection_score"]:
            best = record
    return best


def write_predictions(path: str, ids: Sequence[str], predictions: Sequence[int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for sample_id, pred in zip(ids, predictions):
            f.write(json.dumps({"id": str(sample_id), "prediction": int(pred)}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", default="train.json")
    parser.add_argument("--dev_json", default="dev.json")
    parser.add_argument("--output", default="model.joblib")
    parser.add_argument("--metrics_output", default="metrics.json")
    parser.add_argument("--predictions_output", default="dev_predictions.jsonl")
    parser.add_argument("--include_existing_base", default="base_model.joblib", help="Optional already-trained baseline to include in the ensemble.")
    parser.add_argument("--train_extra_models", action="store_true", help="Also train extra Ridge/SVR/SGD candidates before final calibration. Default keeps training fast and calibrates the bundled baseline.")
    args = parser.parse_args()

    train_ids, train_samples = load_json_samples(args.train_json)
    dev_ids, dev_samples = load_json_samples(args.dev_json)
    y_train = labels(train_samples)
    y_dev = labels(dev_samples)
    sd_dev = stdevs(dev_samples)

    models = []
    model_names = []
    dev_predictions = []

    existing_path = Path(args.include_existing_base)
    if existing_path.exists():
        existing_model = joblib.load(existing_path)
        models.append(existing_model)
        model_names.append("existing_tfidf_ridge_baseline")
        dev_predictions.append(existing_model.predict(dev_samples))
        print("Loaded existing baseline:", existing_path)

    if args.train_extra_models:
        for config in candidate_configs():
            print("Training", config["name"])
            model = make_model(config)
            model.fit(train_samples, y_train)
            pred = model.predict(dev_samples)
            print(" ", evaluate(y_dev, pred, sd_dev))
            models.append(model)
            model_names.append(config["name"])
            dev_predictions.append(pred)

    if not models:
        raise RuntimeError("No models were trained or loaded.")

    best = tune_subset_ensemble(models, dev_predictions, dev_samples, y_dev, sd_dev)
    selected_indices = list(best["indices"])
    selected_models = [models[i] for i in selected_indices]
    selected_names = [model_names[i] for i in selected_indices]
    weights = [1.0 / len(selected_models)] * len(selected_models)
    calibration = best["calibration"]

    final_model = AmbiStoryEnsemble(
        models=selected_models,
        weights=weights,
        group_scale=calibration["group_scale"],
        thresholds=calibration["thresholds"],
        metadata={
            "selected_models": selected_names,
            "weights": weights,
            "raw_dev_metrics": best["raw_metrics"],
            "calibrated_dev_metrics": calibration["metrics"],
            "notes": "TF-IDF ensemble + group contrast/ranking calibration + tuned thresholds.",
        },
    )

    joblib.dump(final_model, args.output)
    dev_int = final_model.predict_int(dev_samples)
    write_predictions(args.predictions_output, dev_ids, dev_int)

    payload = {
        "selected_models": selected_names,
        "weights": weights,
        "group_scale": calibration["group_scale"],
        "thresholds": calibration["thresholds"],
        "raw_dev_metrics": best["raw_metrics"],
        "calibrated_dev_metrics": calibration["metrics"],
        "train_samples": len(train_samples),
        "dev_samples": len(dev_samples),
        "all_candidate_names": model_names,
    }
    with open(args.metrics_output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\nSelected models:", selected_names)
    print("Group scale:", calibration["group_scale"])
    print("Thresholds:", calibration["thresholds"])
    print("Final dev metrics:", calibration["metrics"])
    print("Saved model to", args.output)


if __name__ == "__main__":
    main()
