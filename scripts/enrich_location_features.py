import sqlite3

from scripts.ensure_property_feature_schema import ensure_property_feature_schema
from scripts.location_enrichment import (
    build_grid_index,
    find_nearest_point,
    load_hospitals,
    load_schools,
    load_stations,
    summarize_schools,
)
from scripts.processed_dataset_cache import SELECTED_PROPERTY_FEATURES_FILE, export_table_to_processed_csv


def main():
    ensure_property_feature_schema()

    stations = load_stations()
    schools = load_schools()
    hospitals = load_hospitals()
    station_index = build_grid_index(stations)
    school_index = build_grid_index(schools)
    hospital_index = build_grid_index(hospitals)

    print(f"Loaded {len(stations)} stations")
    print(f"Loaded {len(schools)} schools")
    print(f"Loaded {len(hospitals)} hospitals")

    connection = sqlite3.connect("db/database.db")
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    rows = cursor.execute(
        """
        select id, latitude, longitude
        from property_features
        where latitude is not null and longitude is not null
        """
    ).fetchall()

    updates = []
    for index, row in enumerate(rows, start=1):
        nearest_station, station_distance = find_nearest_point(
            row["latitude"],
            row["longitude"],
            station_index,
            radius_cells=2,
            fallback_points=stations if stations else None,
        )
        school_summary = summarize_schools(
            row["latitude"],
            row["longitude"],
            school_index,
            fallback_points=schools,
        )
        nearest_hospital, hospital_distance = find_nearest_point(
            row["latitude"],
            row["longitude"],
            hospital_index,
            radius_cells=2,
            fallback_points=hospitals if hospitals else None,
        )

        updates.append(
            (
                nearest_station["name"] if nearest_station else None,
                nearest_station.get("zone") if nearest_station else None,
                station_distance,
                school_summary["nearest_school_name"],
                school_summary["nearest_school_type"],
                school_summary["nearest_school_distance_km"],
                school_summary["nearest_primary_school_distance_km"],
                school_summary["nearest_secondary_school_distance_km"],
                school_summary["nearby_primary_schools_1km"],
                school_summary["nearby_secondary_schools_2km"],
                nearest_hospital["name"] if nearest_hospital else None,
                hospital_distance,
                row["id"],
            )
        )

        if len(updates) >= 5000:
            cursor.executemany(
                """
                update property_features
                set nearest_station_name = ?,
                    nearest_station_zone = ?,
                    nearest_station_distance_km = ?,
                    nearest_school_name = ?,
                    nearest_school_type = ?,
                    nearest_school_distance_km = ?,
                    nearest_primary_school_distance_km = ?,
                    nearest_secondary_school_distance_km = ?,
                    nearby_primary_schools_1km = ?,
                    nearby_secondary_schools_2km = ?,
                    nearest_hospital_name = ?,
                    nearest_hospital_distance_km = ?
                where id = ?
                """,
                updates,
            )
            connection.commit()
            print(f"Updated {index}/{len(rows)} property rows with location features")
            updates = []

    if updates:
        cursor.executemany(
            """
            update property_features
            set nearest_station_name = ?,
                nearest_station_zone = ?,
                nearest_station_distance_km = ?,
                nearest_school_name = ?,
                nearest_school_type = ?,
                nearest_school_distance_km = ?,
                nearest_primary_school_distance_km = ?,
                nearest_secondary_school_distance_km = ?,
                nearby_primary_schools_1km = ?,
                nearby_secondary_schools_2km = ?,
                nearest_hospital_name = ?,
                nearest_hospital_distance_km = ?
            where id = ?
            """,
            updates,
        )
        connection.commit()

    connection.close()
    print("Location feature enrichment completed successfully.")
    export_table_to_processed_csv("property_features", SELECTED_PROPERTY_FEATURES_FILE)


if __name__ == "__main__":
    main()
