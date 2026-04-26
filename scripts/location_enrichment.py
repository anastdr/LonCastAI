import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from backend.database import SessionLocal
from backend.models import LondonPostcode

BASE_DIR = Path(__file__).resolve().parent.parent
STATIONS_CSV_FILE = BASE_DIR / "data" / "raw" / "London stations.csv"
STATIONS_KML_FILE = BASE_DIR / "data" / "raw" / "stations.kml"
SCHOOLS_FILE = BASE_DIR / "data" / "raw" / "edubasealldata20260409.csv"
HOSPITAL_FILE_CANDIDATES = [
    BASE_DIR / "data" / "raw" / "London hospitals.csv",
    BASE_DIR / "data" / "raw" / "hospitals.csv",
    BASE_DIR / "data" / "raw" / "Hospital.csv",
]
GRID_SIZE_DEGREES = 0.02
MIN_STATION_COUNT_FOR_ENRICHMENT = 50


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    earth_radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * earth_radius_km * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def make_grid_key(latitude: float, longitude: float) -> tuple[int, int]:
    return (
        int(math.floor(latitude / GRID_SIZE_DEGREES)),
        int(math.floor(longitude / GRID_SIZE_DEGREES)),
    )


def neighboring_keys(latitude: float, longitude: float, radius_cells: int = 1) -> list[tuple[int, int]]:
    lat_key, lon_key = make_grid_key(latitude, longitude)
    keys = []
    for d_lat in range(-radius_cells, radius_cells + 1):
        for d_lon in range(-radius_cells, radius_cells + 1):
            keys.append((lat_key + d_lat, lon_key + d_lon))
    return keys


def build_grid_index(points: Iterable[dict]) -> dict[tuple[int, int], list[dict]]:
    index = defaultdict(list)
    for point in points:
        if point["latitude"] is None or point["longitude"] is None:
            continue
        index[make_grid_key(point["latitude"], point["longitude"])].append(point)
    return index


def load_stations() -> list[dict]:
    if STATIONS_CSV_FILE.exists():
        df = pd.read_csv(STATIONS_CSV_FILE, low_memory=False)
        df = df[["Station", "Latitude", "Longitude", "Zone"]].copy()
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
        df = df[df["Latitude"].notna() & df["Longitude"].notna()].copy()
        stations = [
            {
                "name": str(row["Station"]).strip(),
                "latitude": float(row["Latitude"]),
                "longitude": float(row["Longitude"]),
                "zone": str(row["Zone"]).strip() if pd.notna(row["Zone"]) else None,
                "mode": "STATION",
            }
            for _, row in df.iterrows()
            if str(row["Station"]).strip()
        ]
    else:
        stations = []

    if len(stations) >= MIN_STATION_COUNT_FOR_ENRICHMENT:
        return stations

    print(
        f"Warning: station file contains {len(stations)} usable stations. "
        "Station enrichment needs a fuller dataset."
    )
    return []


def load_schools() -> list[dict]:
    df = pd.read_csv(SCHOOLS_FILE, encoding="latin1", low_memory=False)
    df = df[df["EstablishmentStatus (name)"].fillna("").str.upper() == "OPEN"].copy()
    df = df[df["GOR (name)"].fillna("").str.upper() == "LONDON"].copy()
    df = df[["EstablishmentName", "PhaseOfEducation (name)", "Postcode"]].copy()
    df["postcode_clean"] = (
        df["Postcode"].fillna("").astype(str).str.upper().str.replace(" ", "", regex=False).str.strip()
    )
    df = df[df["postcode_clean"] != ""].copy()

    session = SessionLocal()
    postcode_lookup = {
        row.postcode_clean: (row.latitude, row.longitude)
        for row in session.query(LondonPostcode).all()
    }
    session.close()

    schools = []
    for _, row in df.iterrows():
        coordinates = postcode_lookup.get(row["postcode_clean"])
        if not coordinates:
            continue

        latitude, longitude = coordinates
        if latitude is None or longitude is None:
            continue

        phase = str(row["PhaseOfEducation (name)"]).strip().upper()
        if "PRIMARY" in phase:
            school_type = "PRIMARY"
        elif "SECONDARY" in phase:
            school_type = "SECONDARY"
        else:
            school_type = "OTHER"

        schools.append(
            {
                "name": str(row["EstablishmentName"]).strip(),
                "school_type": school_type,
                "latitude": float(latitude),
                "longitude": float(longitude),
            }
        )

    return schools


def load_hospitals() -> list[dict]:
    hospital_file = next((path for path in HOSPITAL_FILE_CANDIDATES if path.exists()), None)
    if hospital_file is None:
        print("Warning: no hospital dataset found. Hospital enrichment will be skipped.")
        return []

    df = pd.read_csv(hospital_file, low_memory=False)
    normalized_columns = {column.lower().strip(): column for column in df.columns}

    name_column = (
        normalized_columns.get("hospital")
        or normalized_columns.get("name")
        or normalized_columns.get("site name")
        or normalized_columns.get("organisationname")
    )
    latitude_column = normalized_columns.get("latitude") or normalized_columns.get("lat")
    longitude_column = normalized_columns.get("longitude") or normalized_columns.get("lon") or normalized_columns.get("lng")
    postcode_column = normalized_columns.get("postcode")

    if not name_column:
        print(f"Warning: hospital dataset {hospital_file.name} has no recognizable name column.")
        return []

    session = SessionLocal()
    postcode_lookup = {
        row.postcode_clean: (row.latitude, row.longitude)
        for row in session.query(LondonPostcode).all()
    }
    session.close()

    hospitals = []
    for _, row in df.iterrows():
        name = str(row[name_column]).strip() if pd.notna(row[name_column]) else None
        if not name:
            continue

        latitude = None
        longitude = None

        if latitude_column and longitude_column:
            latitude = pd.to_numeric(row[latitude_column], errors="coerce")
            longitude = pd.to_numeric(row[longitude_column], errors="coerce")

        if (pd.isna(latitude) or pd.isna(longitude)) and postcode_column:
            postcode_clean = str(row[postcode_column]).upper().replace(" ", "").strip()
            latitude, longitude = postcode_lookup.get(postcode_clean, (None, None))

        if latitude is None or longitude is None or pd.isna(latitude) or pd.isna(longitude):
            continue

        hospitals.append(
            {
                "name": name,
                "latitude": float(latitude),
                "longitude": float(longitude),
            }
        )

    print(f"Loaded {len(hospitals)} hospitals from {hospital_file.name}")
    return hospitals


def find_nearest_point(
    latitude: float,
    longitude: float,
    grid_index: dict,
    radius_cells: int = 2,
    fallback_points: Optional[list[dict]] = None,
) -> tuple[Optional[dict], Optional[float]]:
    candidates = []
    for key in neighboring_keys(latitude, longitude, radius_cells=radius_cells):
        candidates.extend(grid_index.get(key, []))

    if not candidates and fallback_points:
        candidates = fallback_points

    best_point = None
    best_distance = None
    for point in candidates:
        distance = haversine_km(latitude, longitude, point["latitude"], point["longitude"])
        if best_distance is None or distance < best_distance:
            best_point = point
            best_distance = distance

    return best_point, best_distance


def summarize_schools(latitude: float, longitude: float, grid_index: dict, fallback_points: Optional[list[dict]] = None) -> dict:
    candidates = []
    for key in neighboring_keys(latitude, longitude, radius_cells=2):
        candidates.extend(grid_index.get(key, []))

    if not candidates and fallback_points:
        candidates = fallback_points

    nearest_school = None
    nearest_school_distance = None
    nearest_primary = None
    nearest_secondary = None
    primary_count_1km = 0
    secondary_count_2km = 0

    for school in candidates:
        distance = haversine_km(latitude, longitude, school["latitude"], school["longitude"])

        if nearest_school_distance is None or distance < nearest_school_distance:
            nearest_school = school
            nearest_school_distance = distance

        if school["school_type"] == "PRIMARY":
            if distance <= 1.0:
                primary_count_1km += 1
            if nearest_primary is None or distance < nearest_primary:
                nearest_primary = distance

        if school["school_type"] == "SECONDARY":
            if distance <= 2.0:
                secondary_count_2km += 1
            if nearest_secondary is None or distance < nearest_secondary:
                nearest_secondary = distance

    return {
        "nearest_school_name": nearest_school["name"] if nearest_school else None,
        "nearest_school_type": nearest_school["school_type"] if nearest_school else None,
        "nearest_school_distance_km": nearest_school_distance,
        "nearest_primary_school_distance_km": nearest_primary,
        "nearest_secondary_school_distance_km": nearest_secondary,
        "nearby_primary_schools_1km": primary_count_1km,
        "nearby_secondary_schools_2km": secondary_count_2km,
    }
