from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "data" / "raw" / "london_postcode_directory.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "london_postcodes.csv"

# London local authority codes
LONDON_LA_CODES = {
    "E09000001",  # City of London
    "E09000002",  # Barking and Dagenham
    "E09000003",  # Barnet
    "E09000004",  # Bexley
    "E09000005",  # Brent
    "E09000006",  # Bromley
    "E09000007",  # Camden
    "E09000008",  # Croydon
    "E09000009",  # Ealing
    "E09000010",  # Enfield
    "E09000011",  # Greenwich
    "E09000012",  # Hackney
    "E09000013",  # Hammersmith and Fulham
    "E09000014",  # Haringey
    "E09000015",  # Harrow
    "E09000016",  # Havering
    "E09000017",  # Hillingdon
    "E09000018",  # Hounslow
    "E09000019",  # Islington
    "E09000020",  # Kensington and Chelsea
    "E09000021",  # Kingston upon Thames
    "E09000022",  # Lambeth
    "E09000023",  # Lewisham
    "E09000024",  # Merton
    "E09000025",  # Newham
    "E09000026",  # Redbridge
    "E09000027",  # Richmond upon Thames
    "E09000028",  # Southwark
    "E09000029",  # Sutton
    "E09000030",  # Tower Hamlets
    "E09000031",  # Waltham Forest
    "E09000032",  # Wandsworth
    "E09000033",  # Westminster
}

# Read CSV
df = pd.read_csv(RAW_FILE, dtype=str)

print("Original columns:")
print(df.columns.tolist())
print(f"Original row count: {len(df)}")

# Keep only rows in London local authorities
df = df[df["oslaua"].isin(LONDON_LA_CODES)].copy()

print(f"London-only row count: {len(df)}")

# Keep only useful columns
df = df[["pcd", "pcds", "oslaua", "lat", "long"]].copy()

# Rename columns
df = df.rename(columns={
    "pcd": "postcode",
    "pcds": "postcode_with_space",
    "oslaua": "local_authority_code",
    "lat": "latitude",
    "long": "longitude",
})

# Clean postcode fields
df["postcode"] = df["postcode"].astype(str).str.upper().str.replace(" ", "", regex=False).str.strip()
df["postcode_with_space"] = df["postcode_with_space"].astype(str).str.upper().str.strip()

# Create postcode_clean explicitly
df["postcode_clean"] = df["postcode"]

# Convert lat/long to numeric
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

# Drop duplicate postcodes just in case
df = df.drop_duplicates(subset=["postcode_clean"])

# Reorder columns
df = df[[
    "postcode",
    "postcode_with_space",
    "postcode_clean",
    "local_authority_code",
    "latitude",
    "longitude",
]]

# Save processed file
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved cleaned file to: {OUTPUT_FILE}")
print(df.head())