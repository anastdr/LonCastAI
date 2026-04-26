import re
from typing import Optional


UNIT_WORDS = (
    "FLAT",
    "APARTMENT",
    "APT",
    "UNIT",
    "ROOM",
    "STUDIO",
    "MAISONETTE",
)

UNIT_PATTERN = re.compile(
    r"\b(?:FLAT|APARTMENT|APT|UNIT|ROOM|STUDIO|MAISONETTE)\s+([A-Z0-9-]+)\b"
)

FLOOR_UNIT_PATTERN = re.compile(
    r"\b("
    r"BASEMENT|GROUND|FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|"
    r"LOWER GROUND|UPPER GROUND|LOWER|UPPER|MIDDLE|TOP"
    r")\s+(?:FLOOR\s+)?FLAT\b"
)

ABBREVIATIONS = {
    "APARTMENT": "FLAT",
    "APT": "FLAT",
    "RD": "ROAD",
    "ST": "STREET",
    "AVE": "AVENUE",
    "AV": "AVENUE",
    "LN": "LANE",
    "CL": "CLOSE",
    "CT": "COURT",
    "PL": "PLACE",
    "SQ": "SQUARE",
    "GDNS": "GARDENS",
    "TCE": "TERRACE",
    "DRV": "DRIVE",
}


def clean_text(value) -> Optional[str]:
    if value is None:
        return None

    value = str(value).strip()
    return value or None


def normalize_text(value) -> Optional[str]:
    value = clean_text(value)
    if not value:
        return None

    text = value.upper()
    text = text.replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    parts = [ABBREVIATIONS.get(part, part) for part in text.split()]

    if not parts:
        return None

    return " ".join(parts)


def normalize_house_number(value) -> Optional[str]:
    value = normalize_text(value)
    if not value:
        return None

    return value.replace(" ", "")


def extract_unit(value) -> Optional[str]:
    normalized = normalize_text(value)
    if not normalized:
        return None

    floor_match = FLOOR_UNIT_PATTERN.search(normalized)
    if floor_match:
        return floor_match.group(0)

    unit_match = UNIT_PATTERN.search(normalized)
    if unit_match:
        return f"FLAT {unit_match.group(1)}"

    return None


def strip_leading_unit(value) -> Optional[str]:
    normalized = normalize_text(value)
    if not normalized:
        return None

    updated = FLOOR_UNIT_PATTERN.sub("", normalized, count=1).strip()
    updated = UNIT_PATTERN.sub("", updated, count=1).strip()
    updated = re.sub(r"\s+", " ", updated).strip()
    return updated or None


def extract_house_number_from_address(value) -> Optional[str]:
    remaining = strip_leading_unit(value) or normalize_text(value)
    if not remaining:
        return None

    for token in remaining.split():
        if any(ch.isdigit() for ch in token):
            return normalize_house_number(token)

    return None


def get_last_address_part(value) -> Optional[str]:
    value = clean_text(value)
    if not value:
        return None

    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        return None

    return parts[-1]


def extract_building_name_from_epc(value) -> Optional[str]:
    value = clean_text(value)
    if not value:
        return None

    first_part = value.split(",")[0].strip()
    without_unit = strip_leading_unit(first_part)

    if not without_unit:
        return None

    if any(ch.isdigit() for ch in without_unit):
        return None

    return normalize_text(without_unit)


def build_epc_core_address(address: Optional[str], address1: Optional[str]) -> Optional[str]:
    return normalize_text(address or address1)


def build_sold_core_address(unit, house_number, building_name, street) -> Optional[str]:
    parts = [clean_text(unit), clean_text(building_name)]

    if house_number and not building_name:
        parts.append(clean_text(house_number))
    elif house_number:
        building_text = normalize_text(building_name) or ""
        normalized_house = normalize_text(house_number) or ""
        if normalized_house and normalized_house not in building_text.split():
            parts.append(clean_text(house_number))

    parts.append(clean_text(street))
    combined = " ".join(part for part in parts if part)
    return normalize_text(combined)
