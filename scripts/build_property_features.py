import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from backend.database import SessionLocal
from backend.models import EPCProperty, AddressLookup, PropertyFeature
from scripts.ensure_property_feature_schema import ensure_property_feature_schema
from scripts.processed_dataset_cache import SELECTED_PROPERTY_FEATURES_FILE, export_table_to_processed_csv
from scripts.address_matching import (
    build_epc_core_address,
    build_sold_core_address,
    extract_building_name_from_epc,
    extract_house_number_from_address,
    extract_unit,
    get_last_address_part,
    normalize_house_number,
    normalize_text,
)


@dataclass
class SoldCandidate:
    row: AddressLookup
    core_address: Optional[str]
    unit: Optional[str]
    house_number: Optional[str]
    street: Optional[str]
    building_name: Optional[str]


def get_postcode_sector(postcode: str):
    """
    Returns postcode sector in the format:
    outward code + first character of inward code

    Examples:
    SW6 4AB  -> SW6 4
    W14 9HS  -> W14 9
    NW11 8AA -> NW11 8
    E2 7DP   -> E2 7
    """
    if not postcode:
        return None

    postcode = str(postcode).upper().strip()

    if " " not in postcode and len(postcode) > 3:
        postcode = postcode[:-3] + " " + postcode[-3:]

    parts = postcode.split()

    if len(parts) != 2:
        return None

    outward, inward = parts[0], parts[1]

    if not inward:
        return None

    return f"{outward} {inward[0]}"


def date_key(value: Optional[str]) -> str:
    if not value:
        return ""
    return value[:10]


def choose_latest(existing: Optional[SoldCandidate], candidate: SoldCandidate) -> SoldCandidate:
    if existing is None:
        return candidate

    return candidate if date_key(candidate.row.transfer_date) >= date_key(existing.row.transfer_date) else existing


def build_sold_candidate(row: AddressLookup) -> SoldCandidate:
    inferred_house_number = (
        normalize_house_number(row.house_number)
        or extract_house_number_from_address(row.full_address)
    )

    return SoldCandidate(
        row=row,
        core_address=build_sold_core_address(
            row.unit,
            inferred_house_number,
            row.building_name,
            row.street,
        ),
        unit=normalize_text(row.unit),
        house_number=inferred_house_number,
        street=normalize_text(row.street),
        building_name=normalize_text(row.building_name),
    )


def score_candidate(
    epc_core: Optional[str],
    epc_unit: Optional[str],
    epc_house_number: Optional[str],
    epc_street: Optional[str],
    epc_building_name: Optional[str],
    candidate: SoldCandidate,
) -> int:
    score = 0

    if epc_core and candidate.core_address == epc_core:
        score += 8

    if epc_unit:
        if candidate.unit == epc_unit:
            score += 6
        elif candidate.unit:
            score -= 4

    if epc_house_number:
        if candidate.house_number == epc_house_number:
            score += 5
        elif candidate.house_number:
            score -= 3

    if epc_street:
        if candidate.street == epc_street:
            score += 3
        elif candidate.street:
            score -= 1

    if epc_building_name:
        if candidate.building_name == epc_building_name:
            score += 3
        elif candidate.building_name:
            score -= 1

    return score


def pick_best_candidate(
    epc: EPCProperty,
    by_postcode: Dict[str, List[SoldCandidate]],
    by_core: Dict[Tuple[str, str], SoldCandidate],
    by_house: Dict[Tuple[str, str], List[SoldCandidate]],
    by_unit: Dict[Tuple[str, str], List[SoldCandidate]],
) -> Optional[AddressLookup]:
    postcode_clean = epc.postcode_clean
    if not postcode_clean:
        return None

    epc_source_address = epc.address or epc.address1 or epc.full_address
    epc_core = build_epc_core_address(epc.address, epc.address1)
    epc_unit = extract_unit(epc_source_address)
    epc_house_number = normalize_house_number(epc.house_number) or extract_house_number_from_address(epc_source_address)
    epc_street = normalize_text(get_last_address_part(epc.address))
    epc_building_name = extract_building_name_from_epc(epc.address)

    if epc_core:
        core_match = by_core.get((postcode_clean, epc_core))
        if core_match:
            return core_match.row

    candidate_pool: List[SoldCandidate] = []

    if epc_unit and (postcode_clean, epc_unit) in by_unit:
        candidate_pool = by_unit[(postcode_clean, epc_unit)]
    elif epc_house_number and (postcode_clean, epc_house_number) in by_house:
        candidate_pool = by_house[(postcode_clean, epc_house_number)]
    else:
        candidate_pool = by_postcode.get(postcode_clean, [])

    if not candidate_pool:
        return None

    scored_candidates = []
    for candidate in candidate_pool:
        score = score_candidate(
            epc_core=epc_core,
            epc_unit=epc_unit,
            epc_house_number=epc_house_number,
            epc_street=epc_street,
            epc_building_name=epc_building_name,
            candidate=candidate,
        )
        scored_candidates.append((score, date_key(candidate.row.transfer_date), candidate))

    scored_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score, _, best_candidate = scored_candidates[0]

    # If two different sold addresses get the same score, skip the match to avoid
    # attaching the wrong flat's transaction.
    same_score_distinct_addresses = {
        item[2].row.full_address
        for item in scored_candidates
        if item[0] == best_score
    }
    if len(same_score_distinct_addresses) > 1:
        return None

    if best_score < 5:
        return None

    return best_candidate.row


def main():
    ensure_property_feature_schema()
    start_time = time.time()
    session = SessionLocal()

    print("Clearing old property_features...")
    session.query(PropertyFeature).delete()
    session.commit()

    print("Loading sold-price rows...")
    sold_rows = session.query(AddressLookup).all()
    print(f"Loaded {len(sold_rows)} sold-price rows")

    print("Building sold-price indexes...")
    sold_by_postcode: Dict[str, List[SoldCandidate]] = defaultdict(list)
    sold_by_core: Dict[Tuple[str, str], SoldCandidate] = {}
    sold_by_house: Dict[Tuple[str, str], List[SoldCandidate]] = defaultdict(list)
    sold_by_unit: Dict[Tuple[str, str], List[SoldCandidate]] = defaultdict(list)

    for row in sold_rows:
        if not row.postcode_clean:
            continue

        candidate = build_sold_candidate(row)
        sold_by_postcode[row.postcode_clean].append(candidate)

        if candidate.core_address:
            key = (row.postcode_clean, candidate.core_address)
            sold_by_core[key] = choose_latest(sold_by_core.get(key), candidate)

        if candidate.house_number:
            sold_by_house[(row.postcode_clean, candidate.house_number)].append(candidate)

        if candidate.unit:
            sold_by_unit[(row.postcode_clean, candidate.unit)].append(candidate)

    print(f"Indexed {len(sold_by_core)} postcode+address-core combinations")

    print("Loading EPC rows...")
    epc_rows = session.query(EPCProperty).all()
    print(f"Loaded {len(epc_rows)} EPC rows")

    total_rows = len(epc_rows)
    matched_count = 0
    unmatched_count = 0
    records = []

    print("Building property_features...")

    for i, epc in enumerate(epc_rows, start=1):
        sold = pick_best_candidate(
            epc=epc,
            by_postcode=sold_by_postcode,
            by_core=sold_by_core,
            by_house=sold_by_house,
            by_unit=sold_by_unit,
        )

        if sold:
            matched_count += 1
        else:
            unmatched_count += 1

        records.append(
            PropertyFeature(
                full_address=epc.full_address,
                postcode=epc.postcode,
                postcode_clean=epc.postcode_clean,
                postcode_sector=get_postcode_sector(epc.postcode),
                house_number=epc.house_number,
                latitude=epc.latitude,
                longitude=epc.longitude,
                epc_rating=epc.epc_rating,
                floor_area=epc.floor_area,
                energy_efficiency=epc.energy_efficiency,
                built_form=epc.built_form,
                property_subtype=epc.property_subtype,
                last_sold_price=sold.price if sold else None,
                last_transfer_date=sold.transfer_date if sold else None,
                indexed_last_sold_price=None,
                postcode_average_price=None,
                postcode_average_per_sqm=None,
                nearest_station_name=None,
                nearest_station_zone=None,
                nearest_station_distance_km=None,
                nearest_school_name=None,
                nearest_school_type=None,
                nearest_school_distance_km=None,
                nearest_primary_school_distance_km=None,
                nearest_secondary_school_distance_km=None,
                nearby_primary_schools_1km=None,
                nearby_secondary_schools_2km=None,
                nearest_hospital_name=None,
                nearest_hospital_distance_km=None,
                crime_lsoa_code=None,
                crime_lsoa_name=None,
                crime_total_12m=None,
                crime_avg_monthly_12m=None,
                crime_level=None,
                london_hpi_current_index=None,
                london_hpi_at_last_sale=None,
                london_hpi_annual_change_pct=None,
            )
        )

        if i % 500 == 0 or i == total_rows:
            elapsed = time.time() - start_time
            print(
                f"Processed {i}/{total_rows} rows | "
                f"matched: {matched_count} | unmatched: {unmatched_count} | "
                f"time: {elapsed:.1f}s"
            )

    print("Saving property_features in chunks...")
    chunk_size = 10000

    for start in range(0, len(records), chunk_size):
        end = start + chunk_size
        session.bulk_save_objects(records[start:end])
        session.commit()
        print(f"Saved rows {start + 1} to {min(end, len(records))}")

    session.close()

    elapsed = time.time() - start_time
    print("Property features built successfully.")
    print(f"Final matched: {matched_count}")
    print(f"Final unmatched: {unmatched_count}")
    print(f"Finished in {elapsed:.2f} seconds")
    export_table_to_processed_csv("property_features", SELECTED_PROPERTY_FEATURES_FILE)


if __name__ == "__main__":
    main()
