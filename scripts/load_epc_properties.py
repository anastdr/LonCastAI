from pathlib import Path

import pandas as pd

from backend.database import SessionLocal, engine
from backend.models import Base, EPCProperty, LondonPostcode
from scripts.address_matching import extract_house_number_from_address
from scripts.processed_dataset_cache import (
    SELECTED_EPC_PROPERTIES_FILE,
    ensure_processed_dir,
    should_refresh_processed_cache,
)
from scripts.postcode_filters import get_postcode_prefix_filters

Base.metadata.create_all(bind=engine)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "fuckingBullshit"
CHUNK_SIZE = 50_000
USECOLS = [
    "POSTCODE",
    "ADDRESS1",
    "ADDRESS",
    "POSTTOWN",
    "LOCAL_AUTHORITY_LABEL",
    "CURRENT_ENERGY_RATING",
    "TOTAL_FLOOR_AREA",
    "CURRENT_ENERGY_EFFICIENCY",
    "BUILT_FORM",
    "PROPERTY_TYPE",
    "LODGEMENT_DATE",
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


def clean_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if pd.notna(parsed) else None


def build_full_address(address, posttown, postcode):
    parts = [
        clean_text(address),
        clean_text(posttown),
        clean_text(postcode),
    ]
    return ", ".join([p for p in parts if p])


def transform_chunk(df: pd.DataFrame, allowed_postcodes: set[str]) -> pd.DataFrame:
    df = df.rename(
        columns={
            "POSTCODE": "postcode",
            "ADDRESS1": "address1",
            "ADDRESS": "address",
            "POSTTOWN": "posttown",
            "LOCAL_AUTHORITY_LABEL": "local_authority",
            "CURRENT_ENERGY_RATING": "epc_rating",
            "TOTAL_FLOOR_AREA": "floor_area",
            "CURRENT_ENERGY_EFFICIENCY": "energy_efficiency",
            "BUILT_FORM": "built_form",
            "PROPERTY_TYPE": "property_subtype",
            "LODGEMENT_DATE": "lodgement_date",
        }
    )

    df["postcode"] = df["postcode"].apply(clean_text)
    df["postcode_clean"] = df["postcode"].apply(clean_postcode)
    df = df[df["postcode_clean"].isin(allowed_postcodes)].copy()
    if df.empty:
        return df

    df["address1"] = df["address1"].apply(clean_text)
    df["address"] = df["address"].apply(clean_text)
    df["posttown"] = df["posttown"].apply(clean_text)
    df["local_authority"] = df["local_authority"].apply(clean_text)
    df["full_address_base"] = df["address"].fillna(df["address1"])
    df["house_number"] = (
        df["address"].apply(extract_house_number_from_address)
        .fillna(df["address1"].apply(extract_house_number_from_address))
    )
    df["full_address"] = df.apply(
        lambda row: build_full_address(
            row["full_address_base"],
            row["posttown"],
            row["postcode"],
        ),
        axis=1,
    )
    df["floor_area"] = pd.to_numeric(df["floor_area"], errors="coerce")
    df["energy_efficiency"] = pd.to_numeric(df["energy_efficiency"], errors="coerce")
    df = df[df["postcode_clean"].notna() & df["full_address"].notna()].copy()
    return df


def keep_latest_by_address(existing: dict, chunk: pd.DataFrame) -> None:
    for _, row in chunk.iterrows():
        key = row["full_address"]
        existing_row = existing.get(key)
        row_date = clean_text(row.get("lodgement_date")) or ""
        existing_date = clean_text(existing_row.get("lodgement_date")) if existing_row is not None else ""

        if existing_row is None or row_date >= (existing_date or ""):
            existing[key] = row.to_dict()


def main():
    csv_files = sorted(list(DATA_DIR.rglob("*.csv")) + list(DATA_DIR.rglob("*.CSV")))
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

    refresh_cache = should_refresh_processed_cache()
    if SELECTED_EPC_PROPERTIES_FILE.exists() and not refresh_cache:
        print(f"Loading processed EPC dataset: {SELECTED_EPC_PROPERTIES_FILE}")
        selected_epc = pd.read_csv(SELECTED_EPC_PROPERTIES_FILE, dtype=str)
        selected_epc = selected_epc[selected_epc["postcode_clean"].isin(allowed_postcodes)].copy()
    else:
        if refresh_cache:
            print("Refreshing processed EPC dataset from raw files...")
        latest_rows_by_address: dict[str, dict] = {}

        for file_path in csv_files:
            print(f"Reading {file_path.name} in chunks...")
            for chunk in pd.read_csv(
                file_path,
                dtype=str,
                low_memory=False,
                usecols=USECOLS,
                chunksize=CHUNK_SIZE,
            ):
                prepared = transform_chunk(chunk, allowed_postcodes)
                if prepared.empty:
                    continue
                keep_latest_by_address(latest_rows_by_address, prepared)
                print(f"Tracked EPC addresses so far: {len(latest_rows_by_address)}")

        selected_epc = pd.DataFrame(latest_rows_by_address.values())
        ensure_processed_dir()
        selected_epc.to_csv(SELECTED_EPC_PROPERTIES_FILE, index=False)
        print(f"Saved processed EPC dataset: {SELECTED_EPC_PROPERTIES_FILE} ({len(selected_epc):,} rows)")

    session.query(EPCProperty).delete()
    session.commit()

    postcode_lookup = {
        row.postcode_clean: (row.latitude, row.longitude)
        for row in session.query(LondonPostcode).all()
        if row.postcode_clean in allowed_postcodes
    }

    records = []
    for row in selected_epc.to_dict("records"):
        lat, lon = postcode_lookup.get(row["postcode_clean"], (None, None))
        records.append(
            EPCProperty(
                postcode=clean_text(row["postcode"]),
                postcode_clean=clean_text(row["postcode_clean"]),
                address1=clean_text(row["address1"]),
                address=clean_text(row["address"]),
                posttown=clean_text(row["posttown"]),
                local_authority=clean_text(row["local_authority"]),
                full_address=clean_text(row["full_address"]),
                house_number=clean_text(row["house_number"]),
                epc_rating=clean_text(row.get("epc_rating")),
                floor_area=clean_float(row["floor_area"]),
                energy_efficiency=clean_float(row["energy_efficiency"]),
                built_form=clean_text(row.get("built_form")),
                property_subtype=clean_text(row.get("property_subtype")),
                lodgement_date=clean_text(row.get("lodgement_date")),
                latitude=lat,
                longitude=lon,
            )
        )

    batch_size = 10_000
    for start in range(0, len(records), batch_size):
        session.bulk_save_objects(records[start:start + batch_size])
        session.commit()
        print(f"Saved EPC rows {start + 1} to {min(start + batch_size, len(records))}")

    session.close()
    print("EPC properties loaded successfully.")


if __name__ == "__main__":
    main()
