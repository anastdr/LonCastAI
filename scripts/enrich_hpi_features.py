import sqlite3
from datetime import datetime

import pandas as pd

from scripts.ensure_property_feature_schema import ensure_property_feature_schema
from scripts.processed_dataset_cache import SELECTED_PROPERTY_FEATURES_FILE, export_table_to_processed_csv

HPI_FILE = "data/raw/UK-HPI-full-file-2026-01.csv"

PROPERTY_TYPE_TO_HPI_COLUMN = {
    "FLAT": "FlatIndex",
    "HOUSE": "Index",
    "BUNGALOW": "Index",
    "MAISONETTE": "FlatIndex",
}


def normalize_property_subtype(value: str | None) -> str:
    text = (value or "").upper()
    if "FLAT" in text or "MAISONETTE" in text or "APARTMENT" in text:
        return "FLAT"
    return "HOUSE"


def build_hpi_lookup() -> tuple[dict[tuple[str, str], tuple[float | None, float | None]], float | None, float | None]:
    df = pd.read_csv(HPI_FILE, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df = df[df["RegionName"] == "London"].copy()
    df = df[df["Date"].notna()].copy()
    df["month_key"] = df["Date"].dt.strftime("%Y-%m")
    df = df.sort_values("Date")

    lookup = {}
    for _, row in df.iterrows():
        for subtype, column in PROPERTY_TYPE_TO_HPI_COLUMN.items():
            lookup[(subtype, row["month_key"])] = (
                float(row[column]) if pd.notna(row[column]) else None,
                float(row["12m%Change"]) if pd.notna(row["12m%Change"]) else None,
            )

    latest_row = df.iloc[-1]
    latest_house_index = float(latest_row["Index"]) if pd.notna(latest_row["Index"]) else None
    latest_flat_index = float(latest_row["FlatIndex"]) if pd.notna(latest_row["FlatIndex"]) else None
    return lookup, latest_house_index, latest_flat_index


def main():
    ensure_property_feature_schema()
    hpi_lookup, latest_house_index, latest_flat_index = build_hpi_lookup()

    connection = sqlite3.connect("db/database.db")
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    rows = cursor.execute(
        """
        select id, property_subtype, last_sold_price, last_transfer_date
        from property_features
        """
    ).fetchall()

    updates = []
    for index, row in enumerate(rows, start=1):
        subtype = normalize_property_subtype(row["property_subtype"])
        current_index = latest_flat_index if subtype == "FLAT" else latest_house_index
        sale_index = None
        annual_change_pct = None
        indexed_last_sold_price = None

        if row["last_transfer_date"]:
            try:
                month_key = datetime.strptime(row["last_transfer_date"][:10], "%Y-%m-%d").strftime("%Y-%m")
                sale_index, annual_change_pct = hpi_lookup.get((subtype, month_key), (None, None))
            except Exception:
                sale_index = None
                annual_change_pct = None

        if (
            row["last_sold_price"] is not None
            and sale_index is not None
            and current_index is not None
            and sale_index > 0
        ):
            indexed_last_sold_price = float(row["last_sold_price"]) * (current_index / sale_index)

        updates.append(
            (
                indexed_last_sold_price,
                current_index,
                sale_index,
                annual_change_pct,
                row["id"],
            )
        )

        if len(updates) >= 5000:
            cursor.executemany(
                """
                update property_features
                set indexed_last_sold_price = ?,
                    london_hpi_current_index = ?,
                    london_hpi_at_last_sale = ?,
                    london_hpi_annual_change_pct = ?
                where id = ?
                """,
                updates,
            )
            connection.commit()
            print(f"Updated {index}/{len(rows)} property rows with HPI features")
            updates = []

    if updates:
        cursor.executemany(
            """
            update property_features
            set indexed_last_sold_price = ?,
                london_hpi_current_index = ?,
                london_hpi_at_last_sale = ?,
                london_hpi_annual_change_pct = ?
            where id = ?
            """,
            updates,
        )
        connection.commit()

    connection.close()
    print("HPI enrichment completed successfully.")
    export_table_to_processed_csv("property_features", SELECTED_PROPERTY_FEATURES_FILE)


if __name__ == "__main__":
    main()
