import os
import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SELECTED_LONDON_POSTCODES_FILE = PROCESSED_DIR / "selected_london_postcodes.csv"
SELECTED_EPC_PROPERTIES_FILE = PROCESSED_DIR / "selected_epc_properties.csv"
SELECTED_ADDRESS_LOOKUP_FILE = PROCESSED_DIR / "selected_address_lookup.csv"
SELECTED_PROPERTY_FEATURES_FILE = PROCESSED_DIR / "selected_property_features.csv"


def ensure_processed_dir() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def should_refresh_processed_cache() -> bool:
    value = os.getenv("REFRESH_PROCESSED_DATASETS", "").strip().lower()
    return value in {"1", "true", "yes", "y"}


def export_table_to_processed_csv(table_name: str, output_file: Path) -> None:
    ensure_processed_dir()
    with sqlite3.connect(PROJECT_ROOT / "db" / "database.db") as connection:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", connection)
    df.to_csv(output_file, index=False)
    print(f"Saved processed dataset: {output_file} ({len(df):,} rows)")
