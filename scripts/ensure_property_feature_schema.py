from sqlalchemy import text

from backend.database import engine


COLUMN_DEFINITIONS = {
    "indexed_last_sold_price": "FLOAT",
    "nearest_station_name": "VARCHAR",
    "nearest_station_zone": "VARCHAR",
    "nearest_station_distance_km": "FLOAT",
    "nearest_school_name": "VARCHAR",
    "nearest_school_type": "VARCHAR",
    "nearest_school_distance_km": "FLOAT",
    "nearest_primary_school_distance_km": "FLOAT",
    "nearest_secondary_school_distance_km": "FLOAT",
    "nearby_primary_schools_1km": "INTEGER",
    "nearby_secondary_schools_2km": "INTEGER",
    "nearest_hospital_name": "VARCHAR",
    "nearest_hospital_distance_km": "FLOAT",
    "crime_lsoa_code": "VARCHAR",
    "crime_lsoa_name": "VARCHAR",
    "crime_total_12m": "FLOAT",
    "crime_avg_monthly_12m": "FLOAT",
    "crime_level": "VARCHAR",
    "london_hpi_current_index": "FLOAT",
    "london_hpi_at_last_sale": "FLOAT",
    "london_hpi_annual_change_pct": "FLOAT",
}


def ensure_property_feature_schema() -> None:
    with engine.begin() as connection:
        existing_rows = connection.execute(text("PRAGMA table_info(property_features)")).fetchall()
        existing_columns = {row[1] for row in existing_rows}

        for column_name, column_type in COLUMN_DEFINITIONS.items():
            if column_name in existing_columns:
                continue

            connection.execute(
                text(f"ALTER TABLE property_features ADD COLUMN {column_name} {column_type}")
            )


if __name__ == "__main__":
    ensure_property_feature_schema()
    print("property_features schema is up to date.")
