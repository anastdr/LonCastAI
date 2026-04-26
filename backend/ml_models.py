import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from backend.ml_features import FeaturePreprocessor


MODEL_ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "models" / "property_ml_models.pkl"


class KNNRegressor:
    def __init__(self, n_neighbors: int = 7):
        self.n_neighbors = n_neighbors
        self.x_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.x_train = x_train
        self.y_train = y_train

    def predict_one(self, features: np.ndarray) -> tuple[float, float]:
        if self.x_train is None or self.y_train is None:
            raise ValueError("KNN model is not fitted")

        distances = np.linalg.norm(self.x_train - features, axis=1)
        neighbor_count = min(self.n_neighbors, len(distances))
        neighbor_indexes = np.argpartition(distances, neighbor_count - 1)[:neighbor_count]
        neighbor_distances = distances[neighbor_indexes]
        neighbor_targets = self.y_train[neighbor_indexes]
        weights = 1.0 / (neighbor_distances + 1e-6)
        prediction = float(np.average(neighbor_targets, weights=weights))
        spread = float(np.std(neighbor_targets))
        return prediction, spread


class DecisionTreeRegressor:
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_leaf: int = 12,
        max_features: int | None = None,
        n_split_candidates: int = 12,
        random_state: int | None = None,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_split_candidates = n_split_candidates
        self.random_state = np.random.default_rng(random_state)
        self.tree: dict[str, Any] | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.tree = self._build_tree(x_train, y_train, depth=0)

    def predict_one(self, features: np.ndarray) -> float:
        if self.tree is None:
            raise ValueError("Decision tree is not fitted")

        node = self.tree
        while "value" not in node:
            if features[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return float(node["value"])

    def _build_tree(self, x_values: np.ndarray, y_values: np.ndarray, depth: int) -> dict[str, Any]:
        if (
            depth >= self.max_depth
            or len(y_values) <= self.min_samples_leaf * 2
            or np.var(y_values) == 0
        ):
            return {"value": float(np.mean(y_values))}

        split = self._find_best_split(x_values, y_values)
        if split is None:
            return {"value": float(np.mean(y_values))}

        feature_index, threshold, left_mask = split
        right_mask = ~left_mask
        return {
            "feature": feature_index,
            "threshold": threshold,
            "left": self._build_tree(x_values[left_mask], y_values[left_mask], depth + 1),
            "right": self._build_tree(x_values[right_mask], y_values[right_mask], depth + 1),
        }

    def _find_best_split(self, x_values: np.ndarray, y_values: np.ndarray) -> tuple[int, float, np.ndarray] | None:
        sample_count, feature_count = x_values.shape
        max_features = self.max_features or max(1, int(math.sqrt(feature_count)))
        feature_indexes = self.random_state.choice(
            feature_count,
            size=min(max_features, feature_count),
            replace=False,
        )

        best_score = math.inf
        best_split = None

        for feature_index in feature_indexes:
            column = x_values[:, feature_index]
            if np.all(column == column[0]):
                continue

            quantiles = np.linspace(0.1, 0.9, self.n_split_candidates)
            thresholds = np.unique(np.quantile(column, quantiles))
            for threshold in thresholds:
                left_mask = column <= threshold
                left_count = int(left_mask.sum())
                right_count = sample_count - left_count
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue

                left_var = np.var(y_values[left_mask]) * left_count
                right_var = np.var(y_values[~left_mask]) * right_count
                score = left_var + right_var

                if score < best_score:
                    best_score = score
                    best_split = (feature_index, float(threshold), left_mask)

        return best_split


class RandomForestRegressor:
    def __init__(
        self,
        n_estimators: int = 45,
        max_depth: int = 11,
        min_samples_leaf: int = 12,
        max_features: int | None = None,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = np.random.default_rng(random_state)
        self.trees: list[DecisionTreeRegressor] = []

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.fit_additional_trees(x_train, y_train, self.n_estimators)

    def fit_additional_trees(self, x_train: np.ndarray, y_train: np.ndarray, n_trees: int) -> None:
        sample_count = len(y_train)

        for _ in range(n_trees):
            bootstrap_indexes = self.random_state.integers(0, sample_count, size=sample_count)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=int(self.random_state.integers(0, 1_000_000)),
            )
            tree.fit(x_train[bootstrap_indexes], y_train[bootstrap_indexes])
            self.trees.append(tree)

    def predict_one(self, features: np.ndarray) -> tuple[float, float]:
        if not self.trees:
            raise ValueError("Random forest is not fitted")

        predictions = np.asarray([tree.predict_one(features) for tree in self.trees], dtype=float)
        return float(np.mean(predictions)), float(np.std(predictions))


def confidence_from_uncertainty(prediction: float, uncertainty: float, missing_count: int, validation_mape: float | None) -> float:
    if prediction <= 0:
        return 0.0

    relative_uncertainty = uncertainty / prediction
    validation_penalty = validation_mape or 0.25
    missing_penalty = min(0.25, missing_count * 0.015)
    confidence = 1.0 - relative_uncertainty - validation_penalty - missing_penalty
    return float(max(0.05, min(0.95, confidence)))


def save_artifact(artifact: dict[str, Any]) -> None:
    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_ARTIFACT_PATH.open("wb") as file:
        pickle.dump(artifact, file)


def load_artifact() -> dict[str, Any] | None:
    if not MODEL_ARTIFACT_PATH.exists():
        return None

    with MODEL_ARTIFACT_PATH.open("rb") as file:
        return pickle.load(file)


def predict_with_artifact(record: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    preprocessor: FeaturePreprocessor = artifact["preprocessor"]
    features = preprocessor.transform_one(record)
    missing_count = preprocessor.missing_feature_count(record)
    metrics = artifact.get("metrics", {})

    knn_prediction, knn_spread = artifact["knn_model"].predict_one(features)
    rf_prediction, rf_spread = artifact["rf_model"].predict_one(features)

    if artifact.get("target_transform") == "log1p":
        knn_log_prediction = knn_prediction
        rf_log_prediction = rf_prediction
        knn_prediction = float(np.expm1(knn_log_prediction))
        rf_prediction = float(np.expm1(rf_log_prediction))
        knn_spread = float(max(0.0, knn_prediction * knn_spread))
        rf_spread = float(max(0.0, rf_prediction * rf_spread))

    ensemble_weights = artifact.get("ensemble_weights", {})
    knn_weight = float(ensemble_weights.get("knn_weight", 0.4))
    rf_weight = float(ensemble_weights.get("rf_weight", 0.6))
    blended_prediction = knn_weight * knn_prediction + rf_weight * rf_prediction
    blended_spread = float(np.mean([knn_spread, rf_spread, abs(knn_prediction - rf_prediction)]))

    confidence = confidence_from_uncertainty(
        blended_prediction,
        blended_spread,
        missing_count,
        metrics.get("blended_mape"),
    )

    return {
        "knn_prediction": int(round(knn_prediction)),
        "random_forest_prediction": int(round(rf_prediction)),
        "blended_ml_prediction": int(round(blended_prediction)),
        "confidence_score": round(confidence, 3),
        "confidence_label": "high" if confidence >= 0.7 else "medium" if confidence >= 0.45 else "low",
        "uncertainty": int(round(blended_spread)),
        "missing_feature_count": missing_count,
        "training_row_count": artifact.get("training_row_count"),
        "ensemble_weights": {
            "knn_weight": knn_weight,
            "rf_weight": rf_weight,
        },
        "metrics": metrics,
    }
