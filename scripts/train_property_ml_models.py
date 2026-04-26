import itertools
import os
import random
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd

from backend.ml_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN, FeaturePreprocessor, compute_baseline_feature
from backend.ml_models import KNNRegressor, RandomForestRegressor, save_artifact
from scripts.processed_dataset_cache import SELECTED_PROPERTY_FEATURES_FILE


DEFAULT_MAX_TRAINING_ROWS = 0
MIN_REQUIRED_ROWS = 500
DEFAULT_CV_FOLDS = 4
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.002
RF_GRID_SEARCH_PATIENCE = 10
DEFAULT_MAX_TARGET_PRICE = 50_000_000

KNN_GRID = {
    "n_neighbors": [2, 3, 4, 5, 7, 9, 12, 15, 25],
}

RF_GRID = {
    "max_depth": [8, 11, 14, 18, None],
    "min_samples_leaf": [2, 4, 8, 16],
    "max_features": [6, 8, 12, 16],
}


def get_int_env(name: str, default: int, minimum: int) -> int:
    raw_value = os.getenv(name)
    if not raw_value:
        return default
    try:
        return max(minimum, int(raw_value))
    except ValueError:
        return default


def get_optional_row_limit() -> int:
    raw_value = os.getenv("ML_MAX_TRAINING_ROWS", "").strip()
    if not raw_value:
        return DEFAULT_MAX_TRAINING_ROWS
    if raw_value.upper() in {"ALL", "FULL", "NONE", "0"}:
        return 0
    try:
        return max(MIN_REQUIRED_ROWS, int(raw_value))
    except ValueError:
        return DEFAULT_MAX_TRAINING_ROWS


def load_training_records() -> tuple[list[dict], np.ndarray]:
    max_target_price = get_int_env("ML_MAX_TARGET_PRICE", DEFAULT_MAX_TARGET_PRICE, 1_000_000)
    print(f"Loading training rows with max target £{max_target_price:,}...", flush=True)

    if Path(SELECTED_PROPERTY_FEATURES_FILE).exists():
        print(f"Loading processed training dataset: {SELECTED_PROPERTY_FEATURES_FILE}", flush=True)
        df = pd.read_csv(SELECTED_PROPERTY_FEATURES_FILE, low_memory=False)
        df = df[
            df["indexed_last_sold_price"].notna()
            & (pd.to_numeric(df["indexed_last_sold_price"], errors="coerce") > 50000)
            & (pd.to_numeric(df["indexed_last_sold_price"], errors="coerce") < max_target_price)
            & df["floor_area"].notna()
            & (pd.to_numeric(df["floor_area"], errors="coerce") > 10)
            & (pd.to_numeric(df["floor_area"], errors="coerce") < 2000)
        ].copy()
        rows = df.to_dict("records")
    else:
        connection = sqlite3.connect("db/database.db")
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()

        rows = cursor.execute(
            """
            select *
            from property_features
            where indexed_last_sold_price is not null
              and indexed_last_sold_price > 50000
              and indexed_last_sold_price < ?
              and floor_area is not null
              and floor_area > 10
              and floor_area < 2000
            """,
            (max_target_price,),
        ).fetchall()
        rows = [dict(row) for row in rows]
        connection.close()

    print(f"Loaded {len(rows):,} internal property-feature rows", flush=True)

    records = []
    targets = []
    for row in rows:
        record = dict(row)
        record.setdefault("training_source", "property_features")
        target = record.get(TARGET_COLUMN)
        if target is None:
            continue
        records.append(record)
        targets.append(float(target))

    print(f"Total raw training rows: {len(records):,}", flush=True)
    return records, np.asarray(targets, dtype=float)


def clean_training_data(records: list[dict], targets: np.ndarray) -> tuple[list[dict], np.ndarray, dict]:
    print("Cleaning missing and invalid values...", flush=True)
    cleaned_records = []
    cleaned_targets = []
    removed_rows = 0
    source_counts = defaultdict(int)

    for record, target in zip(records, targets):
        floor_area = record.get("floor_area")
        if floor_area is None or float(floor_area) <= 10:
            removed_rows += 1
            continue
        if not np.isfinite(target) or target <= 50000:
            removed_rows += 1
            continue

        cleaned_record = dict(record)
        for feature in NUMERIC_FEATURES:
            value = cleaned_record.get(feature)
            try:
                value = float(value)
                if not np.isfinite(value):
                    value = None
            except (TypeError, ValueError):
                value = None
            cleaned_record[feature] = value

        for feature in CATEGORICAL_FEATURES:
            value = cleaned_record.get(feature)
            cleaned_record[feature] = str(value).strip().upper() if value is not None and str(value).strip() else None

        cleaned_record["base_estimator_prediction"] = compute_baseline_feature(cleaned_record)

        cleaned_records.append(cleaned_record)
        cleaned_targets.append(target)
        source_counts[cleaned_record.get("training_source") or "property_features"] += 1

    print(f"Cleaned rows: {len(cleaned_records):,}; removed rows: {removed_rows:,}", flush=True)
    return cleaned_records, np.asarray(cleaned_targets, dtype=float), {
        "removed_rows": removed_rows,
        "source_counts": dict(source_counts),
    }


def stratification_key(record: dict, target: float) -> str:
    price_band = int(np.digitize(target, [350000, 600000, 900000, 1400000, 2200000]))
    property_type = (record.get("property_subtype") or "UNKNOWN").upper()
    if "FLAT" in property_type:
        property_group = "FLAT"
    elif "MAISONETTE" in property_type:
        property_group = "MAISONETTE"
    elif "HOUSE" in property_type:
        property_group = "HOUSE"
    else:
        property_group = "OTHER"

    sector = record.get("postcode_sector") or "UNKNOWN"
    return f"{property_group}|{sector}|{price_band}"


def stratified_sample(records: list[dict], targets: np.ndarray, max_rows: int):
    if max_rows <= 0:
        print(f"Using all {len(records):,} cleaned rows; no ML_MAX_TRAINING_ROWS cap was applied", flush=True)
        return records, targets

    if len(records) <= max_rows:
        print(f"Using all {len(records):,} cleaned rows", flush=True)
        return records, targets

    print(f"Stratified sampling {max_rows:,} rows from {len(records):,} cleaned rows...", flush=True)
    rng = random.Random(42)
    groups = defaultdict(list)
    for index, (record, target) in enumerate(zip(records, targets)):
        groups[stratification_key(record, target)].append(index)

    selected_indexes = []
    for indexes in groups.values():
        group_quota = max(1, round(max_rows * len(indexes) / len(records)))
        selected_indexes.extend(rng.sample(indexes, min(group_quota, len(indexes))))

    if len(selected_indexes) > max_rows:
        selected_indexes = rng.sample(selected_indexes, max_rows)
    elif len(selected_indexes) < max_rows:
        remaining = [index for index in range(len(records)) if index not in set(selected_indexes)]
        selected_indexes.extend(rng.sample(remaining, min(max_rows - len(selected_indexes), len(remaining))))

    selected_indexes = sorted(selected_indexes)
    print(f"Sampled {len(selected_indexes):,} rows", flush=True)
    return (
        [records[index] for index in selected_indexes],
        np.asarray([targets[index] for index in selected_indexes], dtype=float),
    )


def stratified_three_way_split(records: list[dict], targets: np.ndarray, train_fraction=0.7, validation_fraction=0.15):
    rng = random.Random(42)
    groups = defaultdict(list)
    for index, (record, target) in enumerate(zip(records, targets)):
        groups[stratification_key(record, target)].append(index)

    train_indexes = []
    validation_indexes = []
    test_indexes = []

    for indexes in groups.values():
        shuffled = list(indexes)
        rng.shuffle(shuffled)
        group_size = len(shuffled)

        if group_size < 3:
            train_indexes.extend(shuffled)
            continue

        train_count = max(1, int(round(group_size * train_fraction)))
        validation_count = max(1, int(round(group_size * validation_fraction)))
        if train_count + validation_count >= group_size:
            validation_count = 1
            train_count = group_size - 2

        train_indexes.extend(shuffled[:train_count])
        validation_indexes.extend(shuffled[train_count:train_count + validation_count])
        test_indexes.extend(shuffled[train_count + validation_count:])

    return train_indexes, validation_indexes, test_indexes


def kfold_indexes_from_train(records: list[dict], targets: np.ndarray, train_indexes: list[int], folds: int):
    rng = random.Random(42)
    groups = defaultdict(list)
    for index in train_indexes:
        groups[stratification_key(records[index], targets[index])].append(index)

    fold_indexes = [[] for _ in range(folds)]
    for indexes in groups.values():
        shuffled = list(indexes)
        rng.shuffle(shuffled)
        for offset, index in enumerate(shuffled):
            fold_indexes[offset % folds].append(index)

    for fold_number in range(folds):
        validation = fold_indexes[fold_number]
        validation_set = set(validation)
        training = [index for index in train_indexes if index not in validation_set]
        yield training, validation


def records_by_indexes(records: list[dict], indexes: list[int]) -> list[dict]:
    return [records[index] for index in indexes]


def targets_by_indexes(targets: np.ndarray, indexes: list[int]) -> np.ndarray:
    return np.asarray([targets[index] for index in indexes], dtype=float)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mape = float(np.mean(np.abs(errors) / np.maximum(y_true, 1.0)))
    total_variance = float(np.sum((y_true - np.mean(y_true)) ** 2))
    residual_variance = float(np.sum(errors ** 2))
    r2 = 1.0 - (residual_variance / total_variance) if total_variance > 0 else 0.0
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 4),
        "r2": round(r2, 4),
    }


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true) / np.maximum(y_true, 1.0)))


def to_price(log_predictions: np.ndarray) -> np.ndarray:
    return np.expm1(log_predictions)


def evaluate_knn_params(records, targets, train_indexes, folds, params):
    fold_scores = []

    for cv_train_indexes, cv_validation_indexes in kfold_indexes_from_train(records, targets, train_indexes, folds):
        train_records = records_by_indexes(records, cv_train_indexes)
        validation_records = records_by_indexes(records, cv_validation_indexes)
        train_targets = targets_by_indexes(targets, cv_train_indexes)
        validation_targets = targets_by_indexes(targets, cv_validation_indexes)

        preprocessor = FeaturePreprocessor.fit(train_records)
        x_train = preprocessor.transform_many(train_records)
        x_validation = preprocessor.transform_many(validation_records)

        model = KNNRegressor(n_neighbors=params["n_neighbors"])
        model.fit(x_train, np.log1p(train_targets))
        log_predictions = np.asarray([model.predict_one(row)[0] for row in x_validation], dtype=float)
        fold_scores.append(mape(validation_targets, to_price(log_predictions)))

    return {
        "params": params,
        "cv_mape": round(mean(fold_scores), 4),
        "fold_mapes": [round(score, 4) for score in fold_scores],
    }


def fit_rf_with_early_stopping(
    x_train,
    y_train_log,
    x_validation,
    y_validation,
    params,
    max_estimators,
    step_estimators,
    random_state,
):
    model = RandomForestRegressor(
        n_estimators=0,
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        random_state=random_state,
    )

    best_score = float("inf")
    best_tree_count = 0
    scores = []
    stale_rounds = 0

    while len(model.trees) < max_estimators:
        model.fit_additional_trees(x_train, y_train_log, step_estimators)
        log_predictions = np.asarray([model.predict_one(row)[0] for row in x_validation], dtype=float)
        score = mape(y_validation, to_price(log_predictions))
        tree_count = len(model.trees)
        scores.append({"trees": tree_count, "mape": round(score, 4)})

        if score < best_score - EARLY_STOPPING_MIN_DELTA:
            best_score = score
            best_tree_count = tree_count
            stale_rounds = 0
        else:
            stale_rounds += 1

        if stale_rounds >= EARLY_STOPPING_PATIENCE:
            break

    if best_tree_count and len(model.trees) > best_tree_count:
        model.trees = model.trees[:best_tree_count]

    return model, {
        "best_mape": round(best_score, 4),
        "best_tree_count": best_tree_count or len(model.trees),
        "staged_scores": scores,
    }


def evaluate_rf_params(records, targets, train_indexes, folds, params, max_estimators, step_estimators):
    fold_scores = []
    best_tree_counts = []

    for fold_number, (cv_train_indexes, cv_validation_indexes) in enumerate(
        kfold_indexes_from_train(records, targets, train_indexes, folds),
        start=1,
    ):
        train_records = records_by_indexes(records, cv_train_indexes)
        validation_records = records_by_indexes(records, cv_validation_indexes)
        train_targets = targets_by_indexes(targets, cv_train_indexes)
        validation_targets = targets_by_indexes(targets, cv_validation_indexes)

        preprocessor = FeaturePreprocessor.fit(train_records)
        x_train = preprocessor.transform_many(train_records)
        x_validation = preprocessor.transform_many(validation_records)

        _, early_stopping = fit_rf_with_early_stopping(
            x_train,
            np.log1p(train_targets),
            x_validation,
            validation_targets,
            params,
            max_estimators=max_estimators,
            step_estimators=step_estimators,
            random_state=42 + fold_number,
        )
        fold_scores.append(early_stopping["best_mape"])
        best_tree_counts.append(early_stopping["best_tree_count"])

    return {
        "params": params,
        "cv_mape": round(mean(fold_scores), 4),
        "fold_mapes": [round(score, 4) for score in fold_scores],
        "avg_best_tree_count": int(round(mean(best_tree_counts))),
        "best_tree_counts": best_tree_counts,
    }


def search_knn(records, targets, train_indexes, folds):
    results = []
    for n_neighbors in KNN_GRID["n_neighbors"]:
        result = evaluate_knn_params(records, targets, train_indexes, folds, {"n_neighbors": n_neighbors})
        results.append(result)
        print("KNN CV", result, flush=True)
    return sorted(results, key=lambda item: item["cv_mape"])


def search_rf(records, targets, train_indexes, folds, max_estimators, step_estimators, grid_search_patience):
    results = []
    best_score = float("inf")
    stale_configs = 0

    for max_depth, min_samples_leaf, max_features in itertools.product(
        RF_GRID["max_depth"],
        RF_GRID["min_samples_leaf"],
        RF_GRID["max_features"],
    ):
        params = {
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        }
        result = evaluate_rf_params(records, targets, train_indexes, folds, params, max_estimators, step_estimators)
        results.append(result)
        print("Random Forest CV", result, flush=True)

        if result["cv_mape"] < best_score - EARLY_STOPPING_MIN_DELTA:
            best_score = result["cv_mape"]
            stale_configs = 0
        else:
            stale_configs += 1
            print(
                f"Random Forest grid search has not improved for {stale_configs}/{grid_search_patience} configs",
                flush=True,
            )

        if stale_configs >= grid_search_patience:
            print(
                "Stopping Random Forest grid search early because parameter search is no longer improving.",
                flush=True,
            )
            break

    return sorted(results, key=lambda item: item["cv_mape"])


def fit_final_models(records, targets, train_indexes, validation_indexes, best_knn, best_rf, max_estimators, step_estimators):
    train_records = records_by_indexes(records, train_indexes)
    validation_records = records_by_indexes(records, validation_indexes)
    train_targets = targets_by_indexes(targets, train_indexes)
    validation_targets = targets_by_indexes(targets, validation_indexes)

    preprocessor = FeaturePreprocessor.fit(train_records)
    x_train = preprocessor.transform_many(train_records)
    x_validation = preprocessor.transform_many(validation_records)
    y_train_log = np.log1p(train_targets)

    knn_model = KNNRegressor(n_neighbors=best_knn["params"]["n_neighbors"])
    knn_model.fit(x_train, y_train_log)

    rf_model, early_stopping = fit_rf_with_early_stopping(
        x_train,
        y_train_log,
        x_validation,
        validation_targets,
        best_rf["params"],
        max_estimators=max_estimators,
        step_estimators=step_estimators,
        random_state=42,
    )
    return preprocessor, knn_model, rf_model, early_stopping


def evaluate_final_models(preprocessor, knn_model, rf_model, records, targets, indexes):
    evaluation_records = records_by_indexes(records, indexes)
    evaluation_targets = targets_by_indexes(targets, indexes)
    x_eval = preprocessor.transform_many(evaluation_records)

    knn_log_predictions = np.asarray([knn_model.predict_one(row)[0] for row in x_eval], dtype=float)
    rf_log_predictions = np.asarray([rf_model.predict_one(row)[0] for row in x_eval], dtype=float)

    knn_predictions = to_price(knn_log_predictions)
    rf_predictions = to_price(rf_log_predictions)
    blended_predictions = 0.4 * knn_predictions + 0.6 * rf_predictions

    return {
        "knn": regression_metrics(evaluation_targets, knn_predictions),
        "random_forest": regression_metrics(evaluation_targets, rf_predictions),
        "blended": regression_metrics(evaluation_targets, blended_predictions),
        "blended_mape": regression_metrics(evaluation_targets, blended_predictions)["mape"],
    }


def find_best_ensemble_weight(preprocessor, knn_model, rf_model, records, targets, validation_indexes):
    validation_records = records_by_indexes(records, validation_indexes)
    validation_targets = targets_by_indexes(targets, validation_indexes)
    x_validation = preprocessor.transform_many(validation_records)
    knn_predictions = to_price(np.asarray([knn_model.predict_one(row)[0] for row in x_validation], dtype=float))
    rf_predictions = to_price(np.asarray([rf_model.predict_one(row)[0] for row in x_validation], dtype=float))

    best_weight = 0.0
    best_score = float("inf")
    scores = []
    for rf_weight in np.linspace(0.0, 1.0, 11):
        prediction = (1.0 - rf_weight) * knn_predictions + rf_weight * rf_predictions
        score = mape(validation_targets, prediction)
        scores.append({"rf_weight": round(float(rf_weight), 2), "mape": round(float(score), 4)})
        if score < best_score:
            best_score = score
            best_weight = float(rf_weight)

    return {
        "knn_weight": round(1.0 - best_weight, 2),
        "rf_weight": round(best_weight, 2),
        "validation_mape": round(best_score, 4),
        "scores": scores,
    }


def evaluate_final_models_with_weights(preprocessor, knn_model, rf_model, records, targets, indexes, weights):
    evaluation_records = records_by_indexes(records, indexes)
    evaluation_targets = targets_by_indexes(targets, indexes)
    x_eval = preprocessor.transform_many(evaluation_records)
    knn_predictions = to_price(np.asarray([knn_model.predict_one(row)[0] for row in x_eval], dtype=float))
    rf_predictions = to_price(np.asarray([rf_model.predict_one(row)[0] for row in x_eval], dtype=float))
    blended_predictions = weights["knn_weight"] * knn_predictions + weights["rf_weight"] * rf_predictions

    return {
        "knn": regression_metrics(evaluation_targets, knn_predictions),
        "random_forest": regression_metrics(evaluation_targets, rf_predictions),
        "blended": regression_metrics(evaluation_targets, blended_predictions),
        "blended_mape": regression_metrics(evaluation_targets, blended_predictions)["mape"],
    }


def main():
    max_training_rows = get_optional_row_limit()
    cv_folds = get_int_env("ML_CV_FOLDS", DEFAULT_CV_FOLDS, 2)
    max_estimators = get_int_env("ML_RF_MAX_ESTIMATORS", 90, 10)
    step_estimators = get_int_env("ML_RF_STEP_ESTIMATORS", 10, 1)
    rf_grid_patience = get_int_env("ML_RF_GRID_PATIENCE", RF_GRID_SEARCH_PATIENCE, 1)

    print("Starting LonCastAI ML training pipeline...", flush=True)
    records, targets = load_training_records()
    records, targets, cleaning_report = clean_training_data(records, targets)
    records, targets = stratified_sample(records, targets, max_training_rows)

    if len(records) < MIN_REQUIRED_ROWS:
        raise RuntimeError(
            f"Not enough exact sold-price training rows. Found {len(records)}, need at least {MIN_REQUIRED_ROWS}."
        )

    train_indexes, validation_indexes, test_indexes = stratified_three_way_split(records, targets)
    print(f"Starting model search with {len(records):,} sampled rows", flush=True)
    print(f"Training source counts: {cleaning_report.get('source_counts')}", flush=True)
    print(f"Removed rows during cleaning: {cleaning_report['removed_rows']:,}", flush=True)
    print(f"Train rows: {len(train_indexes):,} | Validation rows: {len(validation_indexes):,} | Test rows: {len(test_indexes):,}", flush=True)
    print(f"Cross-validation folds on training split: {cv_folds}", flush=True)

    knn_results = search_knn(records, targets, train_indexes, cv_folds)
    rf_results = search_rf(
        records,
        targets,
        train_indexes,
        cv_folds,
        max_estimators=max_estimators,
        step_estimators=step_estimators,
        grid_search_patience=rf_grid_patience,
    )

    best_knn = knn_results[0]
    best_rf = rf_results[0]
    print("Best KNN:", best_knn, flush=True)
    print("Best Random Forest:", best_rf, flush=True)

    preprocessor, knn_model, rf_model, final_early_stopping = fit_final_models(
        records,
        targets,
        train_indexes,
        validation_indexes,
        best_knn,
        best_rf,
        max_estimators,
        step_estimators,
    )

    ensemble_weights = find_best_ensemble_weight(
        preprocessor,
        knn_model,
        rf_model,
        records,
        targets,
        validation_indexes,
    )
    validation_metrics = evaluate_final_models_with_weights(
        preprocessor,
        knn_model,
        rf_model,
        records,
        targets,
        validation_indexes,
        ensemble_weights,
    )
    test_metrics = evaluate_final_models_with_weights(
        preprocessor,
        knn_model,
        rf_model,
        records,
        targets,
        test_indexes,
        ensemble_weights,
    )

    artifact = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_row_count": len(train_indexes),
        "validation_row_count": len(validation_indexes),
        "test_row_count": len(test_indexes),
        "target_transform": "log1p",
        "preprocessor": preprocessor,
        "knn_model": knn_model,
        "rf_model": rf_model,
        "best_params": {
            "knn": best_knn,
            "random_forest": best_rf,
        },
        "cv_results": {
            "knn": knn_results,
            "random_forest": rf_results,
        },
        "early_stopping": final_early_stopping,
        "ensemble_weights": ensemble_weights,
        "cleaning_report": cleaning_report,
        "metrics": {
            "validation": validation_metrics,
            "test": test_metrics,
            "blended_mape": test_metrics["blended_mape"],
        },
    }
    save_artifact(artifact)

    print("ML pipeline completed and saved.", flush=True)
    print("Validation metrics:", validation_metrics, flush=True)
    print("Test metrics:", test_metrics, flush=True)
    print("Final early stopping:", final_early_stopping, flush=True)
    print("Best ensemble weights:", ensemble_weights, flush=True)


if __name__ == "__main__":
    main()
