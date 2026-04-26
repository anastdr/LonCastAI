from pathlib import Path
import pandas as pd

from backend.database import SessionLocal, engine
from backend.models import Base, LondonPostcode
from scripts.processed_dataset_cache import SELECTED_LONDON_POSTCODES_FILE, ensure_processed_dir
from scripts.postcode_filters import filter_dataframe_by_postcode_prefixes, get_postcode_prefix_filters

Base.metadata.create_all(bind=engine)

BASE_DIR = Path(__file__).resolve().parent.parent
FILE_PATH = BASE_DIR / "data" / "processed" / "london_postcodes.csv"

prefixes = get_postcode_prefix_filters()
df = pd.read_csv(FILE_PATH)
df = filter_dataframe_by_postcode_prefixes(df, "postcode_clean", prefixes)

print(f"Using postcode prefixes: {prefixes or 'ALL_LONDON'}")
print(f"London postcode rows to load: {len(df)}")
ensure_processed_dir()
df.to_csv(SELECTED_LONDON_POSTCODES_FILE, index=False)
print(f"Saved processed postcode dataset: {SELECTED_LONDON_POSTCODES_FILE}")

session = SessionLocal()

# Delete old rows so you do not duplicate data
session.query(LondonPostcode).delete()
session.commit()

for _, row in df.iterrows():
    record = LondonPostcode(
        postcode=row["postcode"],
        postcode_with_space=row["postcode_with_space"],
        postcode_clean=row["postcode_clean"],
        local_authority_code=row["local_authority_code"],
        latitude=float(row["latitude"]) if pd.notna(row["latitude"]) else None,
        longitude=float(row["longitude"]) if pd.notna(row["longitude"]) else None,
    )
    session.add(record)

session.commit()
session.close()

print("London postcodes loaded successfully.")
