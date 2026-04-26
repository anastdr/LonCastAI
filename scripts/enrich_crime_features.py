import sqlite3

import pandas as pd

from scripts.ensure_property_feature_schema import ensure_property_feature_schema
from scripts.processed_dataset_cache import SELECTED_PROPERTY_FEATURES_FILE, export_table_to_processed_csv

POSTCODE_DIRECTORY_FILE = "data/raw/london_postcode_directory.csv"
CRIME_FILE = "data/raw/MPS LSOA Level Crime (most recent 24 months).csv"


def load_postcode_to_lsoa_map() -> dict[str, tuple[str | None, str | None]]:
    df = pd.read_csv(POSTCODE_DIRECTORY_FILE, usecols=["pcd", "lsoa11"], low_memory=False)
    df["postcode_clean"] = df["pcd"].astype(str).str.upper().str.replace(" ", "", regex=False).str.strip()
    df["lsoa11"] = df["lsoa11"].astype(str).str.strip()
    df = df[df["postcode_clean"] != ""].copy()
    return {
        row["postcode_clean"]: (row["lsoa11"], None)
        for _, row in df.iterrows()
        if row["lsoa11"] and row["lsoa11"] != "nan"
    }


def build_crime_lookup() -> dict[str, dict]:
    df = pd.read_csv(CRIME_FILE, low_memory=False)
    month_columns = sorted([column for column in df.columns if column.isdigit()])
    latest_12_months = month_columns[-12:]

    df["crime_total_12m"] = df[latest_12_months].sum(axis=1)

    summary = (
        df.groupby(["LSOA Code", "LSOA Name"], as_index=False)["crime_total_12m"]
        .sum()
    )
    summary["crime_avg_monthly_12m"] = summary["crime_total_12m"] / 12.0

    low_threshold = summary["crime_avg_monthly_12m"].quantile(0.33)
    high_threshold = summary["crime_avg_monthly_12m"].quantile(0.66)

    def classify(value: float) -> str:
        if value <= low_threshold:
            return "LOW"
        if value >= high_threshold:
            return "HIGH"
        return "AVERAGE"

    summary["crime_level"] = summary["crime_avg_monthly_12m"].apply(classify)

    return {
        row["LSOA Code"]: {
            "crime_lsoa_name": row["LSOA Name"],
            "crime_total_12m": float(row["crime_total_12m"]),
            "crime_avg_monthly_12m": float(row["crime_avg_monthly_12m"]),
            "crime_level": row["crime_level"],
        }
        for _, row in summary.iterrows()
    }


def main():
    ensure_property_feature_schema()
    postcode_to_lsoa = load_postcode_to_lsoa_map()
    crime_lookup = build_crime_lookup()

    connection = sqlite3.connect("db/database.db")
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    rows = cursor.execute(
        """
        select id, postcode_clean
        from property_features
        """
    ).fetchall()

    updates = []
    for index, row in enumerate(rows, start=1):
        lsoa_code, _ = postcode_to_lsoa.get(row["postcode_clean"], (None, None))
        crime = crime_lookup.get(lsoa_code, {})

        updates.append(
            (
                lsoa_code,
                crime.get("crime_lsoa_name"),
                crime.get("crime_total_12m"),
                crime.get("crime_avg_monthly_12m"),
                crime.get("crime_level"),
                row["id"],
            )
        )

        if len(updates) >= 5000:
            cursor.executemany(
                """
                update property_features
                set crime_lsoa_code = ?,
                    crime_lsoa_name = ?,
                    crime_total_12m = ?,
                    crime_avg_monthly_12m = ?,
                    crime_level = ?
                where id = ?
                """,
                updates,
            )
            connection.commit()
            print(f"Updated {index}/{len(rows)} property rows with crime features")
            updates = []

    if updates:
        cursor.executemany(
            """
            update property_features
            set crime_lsoa_code = ?,
                crime_lsoa_name = ?,
                crime_total_12m = ?,
                crime_avg_monthly_12m = ?,
                crime_level = ?
            where id = ?
            """,
            updates,
        )
        connection.commit()

    connection.close()
    print("Crime enrichment completed successfully.")
    export_table_to_processed_csv("property_features", SELECTED_PROPERTY_FEATURES_FILE)


if __name__ == "__main__":
    main()
