"""AmbiStory model utilities.

This module contains the complete prediction stack used by ``predict.py``:

1. a sense-aware TF-IDF feature extractor;
2. classical regressors used as local, no-API baseline models;
3. group-aware contrast post-processing for the two competing senses of the
   same story/ending;
4. tuned integer thresholds instead of naive rounding;
5. an ensemble wrapper that can combine several local models.

Only the text fields available at test time are used as features. Gold labels
such as ``average``, ``choices``, ``stdev`` and ``nonsensical`` are never used
inside feature extraction.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

TOKEN_RE = re.compile(r"[A-Za-z']+")

TEXT_FIELDS = [
    "homonym",
    "judged_meaning",
    "precontext",
    "sentence",
    "ending",
    "example_sentence",
]


def safe_text(value: object) -> str:
    """Return a clean string for a possibly missing JSON value."""
    if value is None:
        return ""
    return str(value).strip()


def tokenize(text: str) -> List[str]:
    """Simple deterministic tokenizer used for lexical overlap features."""
    return TOKEN_RE.findall(safe_text(text).lower())


def prepare_sample(sample: Dict[str, object]) -> Dict[str, str]:
    """Create derived text fields from a raw AmbiStory sample.

    The target variables are intentionally excluded to avoid label leakage.
    """
    row = {field: safe_text(sample.get(field, "")) for field in TEXT_FIELDS}
    row["story"] = " ".join(
        part for part in [row["precontext"], row["sentence"], row["ending"]] if part
    ).strip()
    row["sense"] = " ".join(
        part for part in [row["homonym"], row["judged_meaning"], row["example_sentence"]] if part
    ).strip()
    row["combined"] = (
        f"homonym: {row['homonym']} meaning: {row['judged_meaning']} "
        f"example: {row['example_sentence']} story: {row['story']}"
    )
    row["ending_combined"] = (
        f"meaning: {row['judged_meaning']} example: {row['example_sentence']} "
        f"ending: {row['ending']}"
    )
    return row


def prepare_samples(samples: Sequence[Dict[str, object]]) -> List[Dict[str, str]]:
    return [prepare_sample(sample) for sample in samples]


def text_column(rows: Sequence[Dict[str, str]], column: str) -> List[str]:
    return [row.get(column, "") for row in rows]


def story_group_key(sample: Dict[str, object]) -> Tuple[str, str, str]:
    """Group samples that share the same story and ending but differ by sense.

    AmbiStory rows are usually paired by the same precontext + ambiguous
    sentence + ending/no-ending. Those paired rows represent competing senses
    and are useful for ranking/contrast calibration at prediction time.
    """
    return (
        safe_text(sample.get("precontext", "")),
        safe_text(sample.get("sentence", "")),
        safe_text(sample.get("ending", "")),
    )


def lexical_numeric_features(rows: Sequence[Dict[str, str]]) -> np.ndarray:
    """Hand-engineered sense-aware features.

    These features are deliberately robust: lengths, homonym occurrences,
    overlap and coverage between the story, ending, candidate definition and
    example sentence.
    """
    features: List[List[float]] = []
    for row in rows:
        story = row["story"]
        precontext = row["precontext"]
        sentence = row["sentence"]
        ending = row["ending"]
        meaning = row["judged_meaning"]
        example = row["example_sentence"]
        homonym = row["homonym"]

        token_sets = {
            "story": set(tokenize(story)),
            "precontext": set(tokenize(precontext)),
            "sentence": set(tokenize(sentence)),
            "ending": set(tokenize(ending)),
            "meaning": set(tokenize(meaning)),
            "example": set(tokenize(example)),
            "sense": set(tokenize(f"{meaning} {example} {homonym}")),
        }

        def jaccard(a: set, b: set) -> float:
            return len(a & b) / max(1, len(a | b))

        def coverage(a: set, b: set) -> float:
            return len(a & b) / max(1, len(b))

        raw_texts = [story, precontext, sentence, ending, meaning, example, homonym]
        nums: List[float] = []
        nums.extend(float(len(text)) for text in raw_texts)
        nums.extend(float(len(tokenize(text))) for text in raw_texts)
        nums.extend(
            [
                1.0 if ending else 0.0,
                float(story.lower().count(homonym.lower())) if homonym else 0.0,
                float(sentence.lower().count(homonym.lower())) if homonym else 0.0,
                float(ending.lower().count(homonym.lower())) if homonym else 0.0,
            ]
        )

        for left in ["story", "precontext", "sentence", "ending"]:
            for right in ["sense", "meaning", "example"]:
                nums.append(jaccard(token_sets[left], token_sets[right]))
                nums.append(coverage(token_sets[left], token_sets[right]))
                nums.append(coverage(token_sets[right], token_sets[left]))

        features.append(nums)

    return np.asarray(features, dtype=np.float32)


class AmbiStoryVectorizer(BaseEstimator, TransformerMixin):
    """Feature extractor for AmbiStory samples.

    It combines word TF-IDF, character TF-IDF, ending-focused TF-IDF,
    pairwise TF-IDF cosine similarities and compact lexical features.
    """

    def __init__(
        self,
        word_ngram_max: int = 2,
        word_max_features: int = 30000,
        char_max_features: int = 20000,
        ending_max_features: int = 12000,
        pair_max_features: int = 40000,
        min_df: int = 2,
        use_char: bool = True,
        use_pair_sims: bool = True,
    ):
        self.word_ngram_max = word_ngram_max
        self.word_max_features = word_max_features
        self.char_max_features = char_max_features
        self.ending_max_features = ending_max_features
        self.pair_max_features = pair_max_features
        self.min_df = min_df
        self.use_char = use_char
        self.use_pair_sims = use_pair_sims

    def fit(self, X: Sequence[Dict[str, object]], y: Sequence[float] | None = None):
        rows = prepare_samples(X)

        self.word_vectorizer_ = TfidfVectorizer(
            ngram_range=(1, self.word_ngram_max),
            min_df=self.min_df,
            max_features=self.word_max_features,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.word_vectorizer_.fit(text_column(rows, "combined"))

        self.ending_vectorizer_ = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=self.ending_max_features,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.ending_vectorizer_.fit(text_column(rows, "ending_combined"))

        if self.use_char:
            self.char_vectorizer_ = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=self.min_df,
                max_features=self.char_max_features,
                sublinear_tf=True,
                strip_accents="unicode",
            )
            self.char_vectorizer_.fit(text_column(rows, "combined"))
        else:
            self.char_vectorizer_ = None

        if self.use_pair_sims:
            pair_corpus: List[str] = []
            for column in [
                "story",
                "precontext",
                "sentence",
                "ending",
                "sense",
                "judged_meaning",
                "example_sentence",
            ]:
                pair_corpus.extend(text_column(rows, column))
            self.pair_vectorizer_ = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                max_features=self.pair_max_features,
                sublinear_tf=True,
                strip_accents="unicode",
            )
            self.pair_vectorizer_.fit(pair_corpus)
        else:
            self.pair_vectorizer_ = None

        numeric = lexical_numeric_features(rows)
        self.numeric_scaler_ = StandardScaler()
        self.numeric_scaler_.fit(numeric)

        if self.use_pair_sims:
            sims = self._pairwise_sims(rows)
            self.sim_scaler_ = StandardScaler()
            self.sim_scaler_.fit(sims)
        else:
            self.sim_scaler_ = None

        return self

    def _cosine_for_columns(self, rows: Sequence[Dict[str, str]], left: str, right: str) -> np.ndarray:
        left_matrix = self.pair_vectorizer_.transform(text_column(rows, left))
        right_matrix = self.pair_vectorizer_.transform(text_column(rows, right))
        return np.asarray(left_matrix.multiply(right_matrix).sum(axis=1)).ravel()

    def _pairwise_sims(self, rows: Sequence[Dict[str, str]]) -> np.ndarray:
        pairs = [
            ("story", "sense"),
            ("story", "judged_meaning"),
            ("story", "example_sentence"),
            ("precontext", "sense"),
            ("sentence", "sense"),
            ("sentence", "judged_meaning"),
            ("sentence", "example_sentence"),
            ("ending", "sense"),
            ("ending", "judged_meaning"),
            ("ending", "example_sentence"),
        ]
        sims = [self._cosine_for_columns(rows, left, right) for left, right in pairs]
        return np.vstack(sims).T.astype(np.float32)

    def transform(self, X: Sequence[Dict[str, object]]):
        rows = prepare_samples(X)
        matrices = [
            self.word_vectorizer_.transform(text_column(rows, "combined")),
            self.ending_vectorizer_.transform(text_column(rows, "ending_combined")),
        ]
        if self.char_vectorizer_ is not None:
            matrices.append(self.char_vectorizer_.transform(text_column(rows, "combined")))

        numeric = self.numeric_scaler_.transform(lexical_numeric_features(rows))
        matrices.append(sparse.csr_matrix(numeric))

        if self.use_pair_sims and self.pair_vectorizer_ is not None:
            sims = self.sim_scaler_.transform(self._pairwise_sims(rows))
            matrices.append(sparse.csr_matrix(sims))

        return sparse.hstack(matrices, format="csr")


class ClippedRoundedRegressor(BaseEstimator, RegressorMixin):
    """Regressor wrapper with helper for valid 1-5 integer predictions."""

    def __init__(self, vectorizer=None, regressor=None):
        self.vectorizer = vectorizer
        self.regressor = regressor

    def fit(self, X: Sequence[Dict[str, object]], y: Sequence[float]):
        self.vectorizer_ = AmbiStoryVectorizer() if self.vectorizer is None else clone(self.vectorizer)
        if self.regressor is None:
            self.regressor_ = SGDRegressor(
                loss="squared_error",
                penalty="elasticnet",
                alpha=0.0001,
                l1_ratio=0.05,
                max_iter=5000,
                tol=1e-4,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=42,
            )
        else:
            self.regressor_ = clone(self.regressor)

        X_features = self.vectorizer_.fit_transform(X, y)
        self.regressor_.fit(X_features, np.asarray(y, dtype=np.float32))
        return self

    def predict(self, X: Sequence[Dict[str, object]]) -> np.ndarray:
        X_features = self.vectorizer_.transform(X)
        predictions = self.regressor_.predict(X_features)
        return np.clip(predictions, 1.0, 5.0)

    def predict_int(self, X: Sequence[Dict[str, object]]) -> np.ndarray:
        return np.rint(self.predict(X)).clip(1, 5).astype(int)


class GroupContrastCalibrator:
    """Amplify within-story differences between competing candidate senses.

    This is the lightweight ranking upgrade. It preserves each story group's
    mean score but increases the spread between candidate senses, which helps
    Spearman-style ordering when the base model is under-confident.
    """

    def __init__(self, scale: float = 1.0):
        self.scale = float(scale)

    def transform(self, samples: Sequence[Dict[str, object]], scores: Sequence[float]) -> np.ndarray:
        adjusted = np.asarray(scores, dtype=np.float32).copy()
        if self.scale == 1.0 or len(adjusted) == 0:
            return np.clip(adjusted, 1.0, 5.0)

        groups: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
        for idx, sample in enumerate(samples):
            groups[story_group_key(sample)].append(idx)

        for indices in groups.values():
            if len(indices) < 2:
                continue
            group_values = adjusted[indices]
            group_mean = float(np.mean(group_values))
            adjusted[indices] = group_mean + self.scale * (group_values - group_mean)

        return np.clip(adjusted, 1.0, 5.0)


class ThresholdCalibrator:
    """Map continuous predictions to integer labels with tuned cut-points."""

    def __init__(self, thresholds: Sequence[float] | None = None):
        self.thresholds = tuple(float(x) for x in (thresholds or (1.5, 2.5, 3.5, 4.5)))
        if len(self.thresholds) != 4:
            raise ValueError("ThresholdCalibrator needs exactly four thresholds.")

    def predict_int(self, scores: Sequence[float]) -> np.ndarray:
        scores_arr = np.asarray(scores, dtype=np.float32)
        labels = np.digitize(scores_arr, bins=np.asarray(self.thresholds, dtype=np.float32)) + 1
        return labels.clip(1, 5).astype(int)


class AmbiStoryEnsemble(BaseEstimator, RegressorMixin):
    """Final submission model combining base models and calibrators."""

    def __init__(
        self,
        models: Sequence[object],
        weights: Sequence[float] | None = None,
        group_scale: float = 1.0,
        thresholds: Sequence[float] | None = None,
        metadata: Dict[str, object] | None = None,
    ):
        self.models = list(models)
        if weights is None:
            self.weights = [1.0 / max(1, len(self.models)) for _ in self.models]
        else:
            weights_arr = np.asarray(weights, dtype=np.float32)
            total = float(np.sum(weights_arr)) or 1.0
            self.weights = (weights_arr / total).tolist()
        self.group_contrast = GroupContrastCalibrator(group_scale)
        self.threshold_calibrator = ThresholdCalibrator(thresholds)
        self.metadata = metadata or {}

    def fit(self, X: Sequence[Dict[str, object]], y: Sequence[float]):
        # Usually this class is assembled after fitting base models. The method
        # exists for sklearn compatibility.
        for model in self.models:
            if hasattr(model, "fit"):
                model.fit(X, y)
        return self

    def predict_raw(self, X: Sequence[Dict[str, object]]) -> np.ndarray:
        if not self.models:
            return np.full(len(X), 3.0, dtype=np.float32)
        weighted = np.zeros(len(X), dtype=np.float32)
        for weight, model in zip(self.weights, self.models):
            weighted += float(weight) * np.asarray(model.predict(X), dtype=np.float32)
        return np.clip(weighted, 1.0, 5.0)

    def predict(self, X: Sequence[Dict[str, object]]) -> np.ndarray:
        raw = self.predict_raw(X)
        return self.group_contrast.transform(X, raw)

    def predict_int(self, X: Sequence[Dict[str, object]]) -> np.ndarray:
        return self.threshold_calibrator.predict_int(self.predict(X))
