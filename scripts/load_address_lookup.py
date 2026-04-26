from pathlib import Path

import pandas as pd

from backend.database import SessionLocal, engine
from backend.models import Base, AddressLookup, LondonPostcode
from scripts.processed_dataset_cache import (
    SELECTED_ADDRESS_LOOKUP_FILE,
    ensure_processed_dir,
    should_refresh_processed_cache,
)
from scripts.postcode_filters import get_postcode_prefix_filters

Base.metadata.create_all(bind=engine)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "price_paid"
CHUNK_SIZE = 100_000

COLUMN_NAMES = [
    "transaction_id",
    "price",
    "transfer_date",
    "postcode",
    "property_type",
    "new_build_flag",
    "tenure",
    "paon",
    "saon",
    "street",
    "locality",
    "town_city",
    "district",
    "county",
    "ppd_category",
    "record_status",
]


def clean_text(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value if value else None


def clean_postcode(value):
    value = clean_text(value)
    if not value:
        return None
    return value.upper().replace(" ", "")


def clean_int(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return int(parsed)


def extract_house_number(paon: str):
    paon = clean_text(paon)
    if not paon:
        return None

    first_token = paon.split()[0].strip(",")
    has_digit = any(ch.isdigit() for ch in first_token)

    if has_digit:
        return first_token

    return None


def build_full_address(saon, paon, street, locality, town_city, postcode):
    parts = [
        clean_text(saon),
        clean_text(paon),
        clean_text(street),
        clean_text(locality),
        clean_text(town_city),
        clean_text(postcode),
    ]
    return ", ".join([p for p in parts if p])


def transform_chunk(df: pd.DataFrame, allowed_postcodes: set[str]) -> pd.DataFrame:
    if "record_status" in df.columns:
        df = df[df["record_status"].fillna("").str.upper() == "A"].copy()

    df["postcode"] = df["postcode"].apply(clean_text)
    df["postcode_clean"] = df["postcode"].apply(clean_postcode)

    df = df[df["postcode_clean"].isin(allowed_postcodes)].copy()
    if df.empty:
        return df

    df["paon"] = df["paon"].apply(clean_text)
    df["saon"] = df["saon"].apply(clean_text)
    df["street"] = df["street"].apply(clean_text)
    df["locality"] = df["locality"].apply(clean_text)
    df["town_city"] = df["town_city"].apply(clean_text)
    df["district"] = df["district"].apply(clean_text)
    df["county"] = df["county"].apply(clean_text)

    df["house_number"] = df["paon"].apply(extract_house_number)
    df["building_name"] = df.apply(
        lambda row: row["paon"] if row["house_number"] != row["paon"] else None,
        axis=1,
    )
    df["unit"] = df["saon"]
    df["full_address"] = df.apply(
        lambda row: build_full_address(
            row["saon"],
            row["paon"],
            row["street"],
            row["locality"],
            row["town_city"],
            row["postcode"],
        ),
        axis=1,
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("Int64")

    df = df[df["postcode_clean"].notna() & df["full_address"].notna()].copy()
    df = df.drop_duplicates(subset=["transaction_id"])
    return df


def main():
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    session = SessionLocal()
    prefixes = get_postcode_prefix_filters()
    london_postcodes = {
        row[0] for row in session.query(LondonPostcode.postcode_clean).all()
    }
    allowed_postcodes = {
        postcode for postcode in london_postcodes
        if not prefixes or any(postcode.startswith(prefix) for prefix in prefixes)
    }

    print(f"Allowed postcode count: {len(allowed_postcodes)}")
    print(f"Using postcode prefixes: {prefixes or 'ALL_LONDON'}")

    session.query(AddressLookup).delete()
    session.commit()

    refresh_cache = should_refresh_processed_cache()
    if SELECTED_ADDRESS_LOOKUP_FILE.exists() and not refresh_cache:
        print(f"Loading processed sold-price dataset: {SELECTED_ADDRESS_LOOKUP_FILE}")
        selected_sales = pd.read_csv(SELECTED_ADDRESS_LOOKUP_FILE, dtype=str)
        selected_sales = selected_sales[selected_sales["postcode_clean"].isin(allowed_postcodes)].copy()
    else:
        if refresh_cache:
            print("Refreshing processed sold-price dataset from raw files...")
        selected_chunks = []

        for file_path in csv_files:
            print(f"Reading {file_path.name} in chunks...")
            for chunk in pd.read_csv(
                file_path,
                header=None,
                names=COLUMN_NAMES,
                dtype=str,
                chunksize=CHUNK_SIZE,
            ):
                prepared = transform_chunk(chunk, allowed_postcodes)
                if prepared.empty:
                    continue
                selected_chunks.append(prepared)
                loaded_count = sum(len(selected_chunk) for selected_chunk in selected_chunks)
                print(f"Selected {loaded_count} sold-price rows so far")

        selected_sales = pd.concat(selected_chunks, ignore_index=True) if selected_chunks else pd.DataFrame()
        if not selected_sales.empty:
            selected_sales = selected_sales.drop_duplicates(subset=["transaction_id"])
        ensure_processed_dir()
        selected_sales.to_csv(SELECTED_ADDRESS_LOOKUP_FILE, index=False)
        print(f"Saved processed sold-price dataset: {SELECTED_ADDRESS_LOOKUP_FILE} ({len(selected_sales):,} rows)")

    records = []
    for _, row in selected_sales.iterrows():
        records.append(
            AddressLookup(
                transaction_id=clean_text(row["transaction_id"]),
                postcode=clean_text(row["postcode"]),
                postcode_clean=clean_text(row["postcode_clean"]),
                house_number=clean_text(row["house_number"]),
                building_name=clean_text(row["building_name"]),
                unit=clean_text(row["unit"]),
                street=clean_text(row["street"]),
                locality=clean_text(row["locality"]),
                town_city=clean_text(row["town_city"]),
                district=clean_text(row["district"]),
                county=clean_text(row["county"]),
                full_address=clean_text(row["full_address"]),
                price=clean_int(row["price"]),
                transfer_date=clean_text(row["transfer_date"]),
                property_type=clean_text(row["property_type"]),
                new_build_flag=clean_text(row["new_build_flag"]),
                tenure=clean_text(row["tenure"]),
            )
        )

    batch_size = 10_000
    for start in range(0, len(records), batch_size):
        session.bulk_save_objects(records[start:start + batch_size])
        session.commit()
        print(f"Loaded sold-price rows {start + 1} to {min(start + batch_size, len(records))}")

    session.close()
    print("Address lookup table loaded successfully.")


if __name__ == "__main__":
    main()
