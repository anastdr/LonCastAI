import math
from dataclasses import dataclass
from typing import Any

import numpy as np


NUMERIC_FEATURES = [
    "base_estimator_prediction",
    "floor_area",
    "energy_efficiency",
    "latitude",
    "longitude",
    "postcode_average_price",
    "postcode_average_per_sqm",
    "nearest_station_distance_km",
    "nearest_school_distance_km",
    "nearest_primary_school_distance_km",
    "nearest_secondary_school_distance_km",
    "nearby_primary_schools_1km",
    "nearby_secondary_schools_2km",
    "crime_total_12m",
    "crime_avg_monthly_12m",
    "london_hpi_current_index",
    "london_hpi_annual_change_pct",
]

CATEGORICAL_FEATURES = [
    "epc_rating",
    "built_form",
    "property_subtype",
    "postcode_sector",
    "nearest_station_zone",
    "nearest_school_type",
    "crime_level",
]

TARGET_COLUMN = "indexed_last_sold_price"


def compute_baseline_feature(record: dict[str, Any]) -> float | None:
    floor_area = safe_float(record.get("floor_area"))
    avg_per_sqm = safe_float(record.get("postcode_average_per_sqm"))
    avg_price = safe_float(record.get("postcode_average_price"))
    indexed_sale = safe_float(record.get("indexed_last_sold_price"))
    last_sale = safe_float(record.get("last_sold_price"))

    if not math.isnan(floor_area) and floor_area > 0 and not math.isnan(avg_per_sqm):
        estimate = floor_area * avg_per_sqm
    elif not math.isnan(indexed_sale):
        estimate = indexed_sale
    elif not math.isnan(last_sale):
        estimate = last_sale
    elif not math.isnan(avg_price):
        estimate = avg_price
    else:
        return None

    epc_rating = str(record.get("epc_rating") or "").upper()
    if epc_rating in {"A", "B"}:
        estimate *= 1.03
    elif epc_rating in {"F", "G"}:
        estimate *= 0.97

    station_distance = safe_float(record.get("nearest_station_distance_km"))
    if not math.isnan(station_distance):
        if station_distance <= 0.5:
            estimate *= 1.02
        elif station_distance <= 1.0:
            estimate *= 1.01
        elif station_distance >= 2.0:
            estimate *= 0.99

    primary_count = safe_float(record.get("nearby_primary_schools_1km"))
    if not math.isnan(primary_count) and primary_count >= 3:
        estimate *= 1.01

    secondary_count = safe_float(record.get("nearby_secondary_schools_2km"))
    if not math.isnan(secondary_count) and secondary_count >= 2:
        estimate *= 1.01

    crime_level = str(record.get("crime_level") or "").upper()
    if crime_level == "LOW":
        estimate *= 1.01
    elif crime_level == "HIGH":
        estimate *= 0.98

    return float(estimate)


def row_to_feature_dict(row: Any) -> dict[str, Any]:
    record = {
        "floor_area": row.floor_area,
        "energy_efficiency": row.energy_efficiency,
        "latitude": row.latitude,
        "longitude": row.longitude,
        "postcode_average_price": row.postcode_average_price,
        "postcode_average_per_sqm": row.postcode_average_per_sqm,
        "nearest_station_distance_km": row.nearest_station_distance_km,
        "nearest_school_distance_km": row.nearest_school_distance_km,
        "nearest_primary_school_distance_km": row.nearest_primary_school_distance_km,
        "nearest_secondary_school_distance_km": row.nearest_secondary_school_distance_km,
        "nearby_primary_schools_1km": row.nearby_primary_schools_1km,
        "nearby_secondary_schools_2km": row.nearby_secondary_schools_2km,
        "crime_total_12m": row.crime_total_12m,
        "crime_avg_monthly_12m": row.crime_avg_monthly_12m,
        "london_hpi_current_index": row.london_hpi_current_index,
        "london_hpi_annual_change_pct": row.london_hpi_annual_change_pct,
        "epc_rating": row.epc_rating,
        "built_form": row.built_form,
        "property_subtype": row.property_subtype,
        "postcode_sector": row.postcode_sector,
        "nearest_station_zone": row.nearest_station_zone,
        "nearest_school_type": row.nearest_school_type,
        "crime_level": row.crime_level,
    }
    record["base_estimator_prediction"] = compute_baseline_feature(record)
    return record


def safe_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


@dataclass
class FeaturePreprocessor:
    numeric_medians: dict[str, float]
    numeric_means: np.ndarray
    numeric_stds: np.ndarray
    category_maps: dict[str, dict[str, int]]
    feature_names: list[str]
    numeric_features: list[str] | None = None
    categorical_features: list[str] | None = None

    @classmethod
    def fit(cls, records: list[dict[str, Any]]) -> "FeaturePreprocessor":
        numeric_matrix = []
        for record in records:
            numeric_matrix.append([safe_float(record.get(feature)) for feature in NUMERIC_FEATURES])

        numeric_array = np.asarray(numeric_matrix, dtype=float)
        numeric_medians_array = np.nanmedian(numeric_array, axis=0)
        numeric_medians_array = np.where(np.isfinite(numeric_medians_array), numeric_medians_array, 0.0)
        filled_numeric = np.where(np.isnan(numeric_array), numeric_medians_array, numeric_array)

        numeric_means = filled_numeric.mean(axis=0)
        numeric_stds = filled_numeric.std(axis=0)
        numeric_stds = np.where(numeric_stds > 0, numeric_stds, 1.0)

        category_maps: dict[str, dict[str, int]] = {}
        for feature in CATEGORICAL_FEATURES:
            values = sorted(
                {
                    str(record.get(feature)).strip().upper()
                    for record in records
                    if record.get(feature) is not None and str(record.get(feature)).strip()
                }
            )
            category_maps[feature] = {value: index + 1 for index, value in enumerate(values)}

        feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        numeric_medians = {
            feature: float(numeric_medians_array[index])
            for index, feature in enumerate(NUMERIC_FEATURES)
        }

        return cls(
            numeric_medians=numeric_medians,
            numeric_means=numeric_means,
            numeric_stds=numeric_stds,
            category_maps=category_maps,
            feature_names=feature_names,
            numeric_features=list(NUMERIC_FEATURES),
            categorical_features=list(CATEGORICAL_FEATURES),
        )

    def transform_one(self, record: dict[str, Any]) -> np.ndarray:
        numeric_features = getattr(self, "numeric_features", None) or list(NUMERIC_FEATURES[: len(self.numeric_means)])
        categorical_features = getattr(self, "categorical_features", None) or list(CATEGORICAL_FEATURES)
        numeric_values = []
        for feature in numeric_features:
            value = safe_float(record.get(feature))
            if math.isnan(value):
                value = self.numeric_medians.get(feature, 0.0)
            numeric_values.append(value)

        numeric_array = np.asarray(numeric_values, dtype=float)
        numeric_array = (numeric_array - self.numeric_means) / self.numeric_stds

        categorical_values = []
        for feature in categorical_features:
            raw_value = record.get(feature)
            normalized = str(raw_value).strip().upper() if raw_value is not None else ""
            categorical_values.append(float(self.category_maps.get(feature, {}).get(normalized, 0)))

        return np.concatenate([numeric_array, np.asarray(categorical_values, dtype=float)])

    def transform_many(self, records: list[dict[str, Any]]) -> np.ndarray:
        return np.vstack([self.transform_one(record) for record in records])

    def missing_feature_count(self, record: dict[str, Any]) -> int:
        missing = 0
        numeric_features = getattr(self, "numeric_features", None) or list(NUMERIC_FEATURES[: len(self.numeric_means)])
        categorical_features = getattr(self, "categorical_features", None) or list(CATEGORICAL_FEATURES)
        for feature in numeric_features + categorical_features:
            value = record.get(feature)
            if value is None or str(value).strip() == "":
                missing += 1
        return missing
