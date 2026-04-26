import argparse
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DB = PROJECT_ROOT / "db" / "database.db"
DEFAULT_OUTPUT_DB = PROJECT_ROOT / "db" / "submission_database.db"
DEFAULT_PREFIXES = ("W8", "SW7", "SW5", "SW3")


def clean_prefixes(raw_prefixes: str) -> list[str]:
    # Normalise user-supplied prefixes once so matching is consistent across
    # loaders, export scripts, and README examples.
    return [
        prefix.strip().upper().replace(" ", "")
        for prefix in raw_prefixes.split(",")
        if prefix.strip()
    ]


def build_prefix_where_clause(prefixes: list[str], column_name: str) -> tuple[str, list[str]]:
    if not prefixes:
        return "1 = 1", []

    conditions = [f"UPPER(REPLACE({column_name}, ' ', '')) LIKE ?" for _ in prefixes]
    values = [f"{prefix}%" for prefix in prefixes]
    return " OR ".join(conditions), values


def copy_filtered_table(
    source_connection: sqlite3.Connection,
    output_connection: sqlite3.Connection,
    table_name: str,
    postcode_column: str,
    prefixes: list[str],
) -> int:
    # Recreate each table with only the selected postcode scope so the
    # submitted database stays small enough for GitHub and marking.
    where_clause, values = build_prefix_where_clause(prefixes, postcode_column)

    schema_row = source_connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    if schema_row is None:
        raise RuntimeError(f"Table {table_name} was not found in the source database.")

    output_connection.execute(f"DROP TABLE IF EXISTS {table_name}")
    output_connection.execute(schema_row[0])

    column_rows = source_connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    column_names = [row[1] for row in column_rows]
    columns_sql = ", ".join(column_names)
    placeholders = ", ".join(["?"] * len(column_names))

    rows = source_connection.execute(
        f"SELECT {columns_sql} FROM {table_name} WHERE {where_clause}",
        values,
    )
    output_connection.executemany(
        f"INSERT INTO {table_name} ({columns_sql}) VALUES ({placeholders})",
        rows,
    )

    count = output_connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"Copied {count:,} rows into {table_name}")
    return count


def create_submission_database(source_db: Path, output_db: Path, prefixes: list[str]) -> None:
    if not source_db.exists():
        raise FileNotFoundError(f"Source database was not found: {source_db}")

    output_db.parent.mkdir(parents=True, exist_ok=True)
    if output_db.exists():
        output_db.unlink()

    print(f"Creating compact database: {output_db}")
    print(f"Using source database: {source_db}")
    print(f"Keeping postcode prefixes: {', '.join(prefixes) if prefixes else 'all'}")

    with sqlite3.connect(source_db) as source_connection, sqlite3.connect(output_db) as output_connection:
        # The app only needs the enriched feature table and postcode lookup for
        # the final demo, not the full raw import tables.
        copy_filtered_table(
            source_connection,
            output_connection,
            "property_features",
            "postcode_clean",
            prefixes,
        )
        copy_filtered_table(
            source_connection,
            output_connection,
            "london_postcodes",
            "postcode_clean",
            prefixes,
        )

        # The website does not need the raw EPC or full sold-price tables once
        # property_features has been built. These small indexes keep lookups fast.
        output_connection.execute("CREATE INDEX idx_property_features_postcode ON property_features(postcode_clean)")
        output_connection.execute("CREATE INDEX idx_property_features_sector ON property_features(postcode_sector)")
        output_connection.execute("CREATE INDEX idx_london_postcodes_postcode ON london_postcodes(postcode_clean)")
        output_connection.commit()
        output_connection.execute("VACUUM")

    size_mb = output_db.stat().st_size / (1024 * 1024)
    print(f"Finished. Compact database size: {size_mb:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a compact SQLite database for GitHub/submission from selected postcode prefixes."
    )
    parser.add_argument("--source-db", type=Path, default=DEFAULT_SOURCE_DB)
    parser.add_argument("--output-db", type=Path, default=DEFAULT_OUTPUT_DB)
    parser.add_argument("--prefixes", default=",".join(DEFAULT_PREFIXES))
    args = parser.parse_args()

    create_submission_database(args.source_db, args.output_db, clean_prefixes(args.prefixes))


if __name__ == "__main__":
    main()
