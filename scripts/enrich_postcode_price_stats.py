from collections import defaultdict
from datetime import datetime

from backend.database import SessionLocal
from backend.models import PropertyFeature
from scripts.processed_dataset_cache import SELECTED_PROPERTY_FEATURES_FILE, export_table_to_processed_csv


def get_transaction_recency_weight(transfer_date: str) -> float:
    """
    Higher weight for more recent transactions.
    """
    if not transfer_date:
        return 0.2

    try:
        sale_date = datetime.strptime(transfer_date[:10], "%Y-%m-%d")
    except Exception:
        return 0.2

    today = datetime.today()
    days_old = (today - sale_date).days

    if days_old <= 365:
        return 1.0
    elif days_old <= 3 * 365:
        return 0.7
    elif days_old <= 5 * 365:
        return 0.5
    else:
        return 0.25


def main():
    session = SessionLocal()

    print("Loading property_features...")
    rows = session.query(PropertyFeature).all()
    print(f"Loaded {len(rows)} property_features rows")

    sector_price_totals = defaultdict(float)
    sector_price_weights = defaultdict(float)

    sector_ppsqm_totals = defaultdict(float)
    sector_ppsqm_weights = defaultdict(float)

    print("Calculating postcode sector averages...")

    for row in rows:
        sector = row.postcode_sector

        if sector and row.last_sold_price is not None:
            weight = get_transaction_recency_weight(row.last_transfer_date)
            sector_price_totals[sector] += float(row.last_sold_price) * weight
            sector_price_weights[sector] += weight

        if (
            sector
            and row.last_sold_price is not None
            and row.floor_area is not None
            and row.floor_area > 0
        ):
            weight = get_transaction_recency_weight(row.last_transfer_date)
            price_per_sqm = float(row.last_sold_price) / float(row.floor_area)
            sector_ppsqm_totals[sector] += price_per_sqm * weight
            sector_ppsqm_weights[sector] += weight

    print(f"Sectors with average price: {len(sector_price_weights)}")
    print(f"Sectors with average price per sqm: {len(sector_ppsqm_weights)}")

    print("Writing postcode sector averages back to property_features...")

    total = len(rows)

    for i, row in enumerate(rows, start=1):
        sector = row.postcode_sector

        if sector in sector_price_weights and sector_price_weights[sector] > 0:
            row.postcode_average_price = (
                sector_price_totals[sector] / sector_price_weights[sector]
            )
        else:
            row.postcode_average_price = None

        if sector in sector_ppsqm_weights and sector_ppsqm_weights[sector] > 0:
            row.postcode_average_per_sqm = (
                sector_ppsqm_totals[sector] / sector_ppsqm_weights[sector]
            )
        else:
            row.postcode_average_per_sqm = None

        if i % 10000 == 0 or i == total:
            print(f"Updated {i}/{total} rows")

    session.commit()
    session.close()

    print("Postcode sector enrichment completed successfully.")
    print("postcode_average_price and postcode_average_per_sqm updated.")
    export_table_to_processed_csv("property_features", SELECTED_PROPERTY_FEATURES_FILE)


if __name__ == "__main__":
    main()
