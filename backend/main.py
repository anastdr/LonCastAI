import os
import math
import smtplib
from functools import lru_cache
from pathlib import Path
from email.message import EmailMessage

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .database import engine, SessionLocal
from . import models
from datetime import datetime
from sqlalchemy import func 
from scripts.ensure_property_feature_schema import ensure_property_feature_schema
from scripts.location_enrichment import build_grid_index, find_nearest_point, load_schools, load_stations
from backend.ml_features import row_to_feature_dict
from backend.ml_models import load_artifact, predict_with_artifact


models.Base.metadata.create_all(bind=engine)
ensure_property_feature_schema()


# Apply only small, explainable local-feature adjustments so the baseline
# remains interpretable rather than turning into a second opaque model.
def apply_location_feature_adjustments(row, estimate_value: float, explanations: list[str]) -> float:
    adjusted = estimate_value

    if row.nearest_station_distance_km is not None:
        if row.nearest_station_distance_km <= 0.5:
            adjusted *= 1.02
            explanations.append("Very close station access slightly increases the estimate.")
        elif row.nearest_station_distance_km <= 1.0:
            adjusted *= 1.01
            explanations.append("Good station access gives the estimate a small uplift.")
        elif row.nearest_station_distance_km >= 2.0:
            adjusted *= 0.99
            explanations.append("Being farther from a station slightly reduces the estimate.")

    if row.nearby_primary_schools_1km is not None and row.nearby_primary_schools_1km >= 3:
        adjusted *= 1.01
        explanations.append("Several nearby primary schools slightly increase the estimate.")

    if row.nearby_secondary_schools_2km is not None and row.nearby_secondary_schools_2km >= 2:
        adjusted *= 1.01
        explanations.append("Access to nearby secondary schools slightly increases the estimate.")

    if row.crime_level == "LOW":
        adjusted *= 1.01
        explanations.append("Lower recent crime in the surrounding area slightly increases the estimate.")
    elif row.crime_level == "HIGH":
        adjusted *= 0.98
        explanations.append("Higher recent crime in the surrounding area slightly reduces the estimate.")

    return adjusted


def format_distance_km(distance_km: float | None) -> str | None:
    if distance_km is None:
        return None
    if distance_km < 1:
        return f"{int(round(distance_km * 1000))} m"
    return f"{distance_km:.2f} km"


@lru_cache(maxsize=1)
def get_map_amenity_indexes():
    # Build the amenity lookup once at startup-time use so repeated searches
    # can reuse the same in-memory spatial index.
    stations = load_stations()
    schools = load_schools()
    return {
        "stations": stations,
        "schools": schools,
        "station_index": build_grid_index(stations),
        "school_index": build_grid_index(schools),
    }


def get_nearest_map_amenities(latitude: float | None, longitude: float | None) -> list[dict]:
    if latitude is None or longitude is None:
        return []

    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return []

    if not (math.isfinite(lat) and math.isfinite(lon)):
        return []

    indexes = get_map_amenity_indexes()
    amenities = []

    nearest_station, station_distance = find_nearest_point(
        lat,
        lon,
        indexes["station_index"],
        radius_cells=2,
        fallback_points=indexes["stations"],
    )
    if nearest_station:
        amenities.append(
            {
                "type": "station",
                "name": nearest_station["name"],
                "latitude": nearest_station["latitude"],
                "longitude": nearest_station["longitude"],
                "distance_km": station_distance,
                "label": "Closest station",
                "detail": f"Zone {nearest_station.get('zone')}" if nearest_station.get("zone") else None,
            }
        )

    nearest_school, school_distance = find_nearest_point(
        lat,
        lon,
        indexes["school_index"],
        radius_cells=2,
        fallback_points=indexes["schools"],
    )
    if nearest_school:
        amenities.append(
            {
                "type": "school",
                "name": nearest_school["name"],
                "latitude": nearest_school["latitude"],
                "longitude": nearest_school["longitude"],
                "distance_km": school_distance,
                "label": "Closest school",
                "detail": nearest_school.get("school_type"),
            }
        )

    return amenities


def blend_baseline_and_ml(baseline_prediction: int, ml_prediction: dict) -> dict:
    # Keep the blended output conservative when the ML model strongly disagrees
    # with the baseline or reports low confidence.
    ml_value = int(ml_prediction["blended_ml_prediction"])
    if baseline_prediction <= 0 or ml_value <= 0:
        return {
            "mixed_prediction": baseline_prediction,
            "ml_weight": 0.0,
            "baseline_weight": 1.0,
            "disagreement_ratio": None,
            "confidence_score": 0.15,
            "confidence_label": "low",
            "warning": "ML prediction was not safely comparable with the baseline, so the mixed prediction uses the baseline estimate.",
        }

    lower = min(baseline_prediction, ml_value)
    higher = max(baseline_prediction, ml_value)
    disagreement_ratio = higher / lower
    model_confidence = float(ml_prediction.get("confidence_score") or 0.35)

    if disagreement_ratio >= 2.5:
        ml_weight = 0.15
        warning = (
            "The ML and baseline estimates are very far apart, so the mixed prediction puts most weight on the baseline and lowers confidence."
        )
    elif disagreement_ratio >= 1.8:
        ml_weight = 0.25
        warning = (
            "The ML and baseline estimates disagree substantially, so the mixed prediction uses a conservative ML weight."
        )
    elif disagreement_ratio >= 1.35:
        ml_weight = 0.35
        warning = (
            "The ML and baseline estimates differ noticeably, so the mixed prediction is weighted towards the baseline."
        )
    else:
        ml_weight = 0.5
        warning = None

    if model_confidence < 0.45:
        ml_weight = min(ml_weight, 0.25)
    if model_confidence < 0.25:
        ml_weight = min(ml_weight, 0.15)

    baseline_weight = 1.0 - ml_weight
    mixed_prediction = int(round(baseline_weight * baseline_prediction + ml_weight * ml_value))

    adjusted_confidence = model_confidence
    if disagreement_ratio >= 2.5:
        adjusted_confidence = min(adjusted_confidence, 0.25)
    elif disagreement_ratio >= 1.8:
        adjusted_confidence = min(adjusted_confidence, 0.35)
    elif disagreement_ratio >= 1.35:
        adjusted_confidence = min(adjusted_confidence, 0.5)

    return {
        "mixed_prediction": mixed_prediction,
        "ml_weight": round(ml_weight, 2),
        "baseline_weight": round(baseline_weight, 2),
        "disagreement_ratio": round(disagreement_ratio, 2),
        "confidence_score": round(adjusted_confidence, 3),
        "confidence_label": "high" if adjusted_confidence >= 0.7 else "medium" if adjusted_confidence >= 0.45 else "low",
        "warning": warning,
    }


def get_sale_recency_weight(transfer_date: str) -> float:
    """
    Returns how strongly to trust a matched sold price.
    More recent sales get a higher weight in the final estimate.
    """
    if not transfer_date:
        return 0.65

    try:
        sale_date = datetime.strptime(transfer_date[:10], "%Y-%m-%d")
    except Exception:
        return 0.65

    today = datetime.today()
    days_old = (today - sale_date).days

    if days_old <= 365:
        return 0.9
    elif days_old <= 3 * 365:
        return 0.85
    elif days_old <= 5 * 365:
        return 0.8
    elif days_old <= 8 * 365:
        return 0.75
    else:
        return 0.65


def sale_anchor_weight_for_ml(transfer_date: str | None) -> float:
    return min(0.85, max(0.65, get_sale_recency_weight(transfer_date)))


def anchor_ml_to_market_evidence(
    ml_prediction: dict,
    sale_anchor_price: float | None,
    transfer_date: str | None,
    local_comparable: dict | None = None,
) -> dict:
    # Pull extreme ML outputs back towards observed market evidence so large
    # postcode outliers do not dominate the final estimate.
    if sale_anchor_price is None or sale_anchor_price <= 0:
        if local_comparable is None or local_comparable.get("estimate") is None:
            return ml_prediction
        market_anchor = float(local_comparable["estimate"])
        anchor_weight = 0.65 if local_comparable.get("source") == "same postcode" else 0.5
        anchor_reason = "nearby sold comparables"
    else:
        market_anchor = float(sale_anchor_price)
        anchor_weight = sale_anchor_weight_for_ml(transfer_date)
        anchor_reason = "the HPI-indexed exact sale"

    raw_ml_value = float(ml_prediction["blended_ml_prediction"])
    if raw_ml_value <= 0:
        return ml_prediction

    ratio = max(raw_ml_value, market_anchor) / min(raw_ml_value, market_anchor)
    if ratio < 1.35:
        return ml_prediction

    anchored_value = anchor_weight * market_anchor + (1.0 - anchor_weight) * raw_ml_value
    anchored_value = max(anchored_value, market_anchor * 0.85)
    if sale_anchor_price is not None:
        anchored_value = max(anchored_value, market_anchor * 0.9)
    adjusted_confidence = min(float(ml_prediction.get("confidence_score") or 0.35), 0.45)

    return {
        **ml_prediction,
        "raw_blended_ml_prediction": ml_prediction["blended_ml_prediction"],
        "blended_ml_prediction": int(round(anchored_value)),
        "confidence_score": round(adjusted_confidence, 3),
        "confidence_label": "medium" if adjusted_confidence >= 0.45 else "low",
        "sale_anchor_weight": round(anchor_weight, 2),
        "sale_anchor_adjusted": True,
        "sale_anchor_disagreement_ratio": round(ratio, 2),
        "sale_anchor_reason": anchor_reason,
    }


def get_local_comparable_estimate(db, row) -> dict | None:
    if row.floor_area is None or row.floor_area <= 0:
        return None

    same_postcode = (
        db.query(models.PropertyFeature)
        .filter(models.PropertyFeature.postcode_clean == row.postcode_clean)
        .filter(models.PropertyFeature.id != row.id)
        .filter(models.PropertyFeature.indexed_last_sold_price.isnot(None))
        .filter(models.PropertyFeature.floor_area.isnot(None))
        .filter(models.PropertyFeature.floor_area > 10)
        .all()
    )

    candidates = [
        candidate
        for candidate in same_postcode
        if candidate.indexed_last_sold_price
        and candidate.indexed_last_sold_price > 50000
        and candidate.floor_area
        and candidate.floor_area > 10
    ]
    source = "same postcode"

    if len(candidates) < 2 and row.postcode_sector:
        sector_candidates = (
            db.query(models.PropertyFeature)
            .filter(models.PropertyFeature.postcode_sector == row.postcode_sector)
            .filter(models.PropertyFeature.id != row.id)
            .filter(models.PropertyFeature.indexed_last_sold_price.isnot(None))
            .filter(models.PropertyFeature.floor_area.isnot(None))
            .filter(models.PropertyFeature.floor_area > 10)
            .all()
        )
        candidates = [
            candidate
            for candidate in sector_candidates
            if candidate.indexed_last_sold_price
            and candidate.indexed_last_sold_price > 50000
            and candidate.floor_area
            and candidate.floor_area > 10
        ]
        source = "postcode sector"

    if not candidates:
        return None

    target_area = float(row.floor_area)

    def candidate_score(candidate):
        area_gap = abs(float(candidate.floor_area) - target_area) / max(target_area, 1.0)
        same_subtype_bonus = 0 if candidate.property_subtype == row.property_subtype else 0.25
        return area_gap + same_subtype_bonus

    selected = sorted(candidates, key=candidate_score)[:3]
    if not selected:
        return None

    prices_per_sqm = [
        float(candidate.indexed_last_sold_price) / float(candidate.floor_area)
        for candidate in selected
        if candidate.floor_area and candidate.floor_area > 0
    ]
    if not prices_per_sqm:
        return None

    average_per_sqm = sum(prices_per_sqm) / len(prices_per_sqm)
    estimate = average_per_sqm * target_area

    return {
        "estimate": float(estimate),
        "average_per_sqm": float(average_per_sqm),
        "count": len(prices_per_sqm),
        "source": source,
        "addresses": [candidate.full_address for candidate in selected],
    }

app = FastAPI()

ML_ARTIFACT_CACHE = None


class ContactRequest(BaseModel):
    name: str
    email: str
    message: str


def get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise HTTPException(
            status_code=503,
            detail=f"Contact email is not configured yet. Missing {name} in the server .env file.",
        )
    return value


def validate_contact_request(contact: ContactRequest) -> None:
    if not contact.name.strip():
        raise HTTPException(status_code=400, detail="Please enter your name.")
    if "@" not in contact.email or "." not in contact.email:
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    if len(contact.message.strip()) < 10:
        raise HTTPException(status_code=400, detail="Please enter a message with at least 10 characters.")


@app.post("/contact")
def send_contact_message(contact: ContactRequest):
    validate_contact_request(contact)

    smtp_host = get_required_env("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = get_required_env("SMTP_USERNAME")
    smtp_password = get_required_env("SMTP_PASSWORD")
    smtp_from = os.getenv("SMTP_FROM_EMAIL", smtp_username).strip()
    contact_to = os.getenv("CONTACT_TO_EMAIL", "aderk001@campus.goldsmiths.ac.uk").strip()
    use_tls = os.getenv("SMTP_USE_TLS", "true").strip().lower() not in {"0", "false", "no"}

    email_message = EmailMessage()
    email_message["Subject"] = f"LonCastAI contact form message from {contact.name.strip()}"
    email_message["From"] = smtp_from
    email_message["To"] = contact_to
    email_message["Reply-To"] = contact.email.strip()
    email_message.set_content(
        "\n".join(
            [
                "New LonCastAI contact form message",
                "",
                f"Name: {contact.name.strip()}",
                f"Email: {contact.email.strip()}",
                "",
                "Message:",
                contact.message.strip(),
            ]
        )
    )

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as smtp:
            if use_tls:
                smtp.starttls()
            smtp.login(smtp_username, smtp_password)
            smtp.send_message(email_message)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"The message could not be sent by the email server: {exc}",
        ) from exc

    return {"success": True, "message": "Thank you. Your message has been sent."}


def get_ml_artifact():
    global ML_ARTIFACT_CACHE
    if ML_ARTIFACT_CACHE is None:
        ML_ARTIFACT_CACHE = load_artifact()
    return ML_ARTIFACT_CACHE


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

@app.get("/")
def root():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "API is working", "frontend": "Missing frontend/index.html"}

@app.get("/check-postcode")
def check_postcode(postcode: str):
    db = SessionLocal()
    cleaned_postcode = postcode.upper().replace(" ", "").strip()

    result = (
        db.query(models.LondonPostcode)
        .filter(models.LondonPostcode.postcode_clean == cleaned_postcode)
        .first()
    )

    db.close()

    if not result:
        return {"valid_london_postcode": False}

    return {
        "valid_london_postcode": True,
        "postcode": result.postcode_with_space,
        "postcode_clean": result.postcode_clean,
        "local_authority_code": result.local_authority_code,
        "latitude": result.latitude,
        "longitude": result.longitude,
    }

@app.get("/address-suggestions")
def address_suggestions(postcode: str, door_number: str = ""):
    db = SessionLocal()

    cleaned_postcode = postcode.upper().replace(" ", "").strip()
    cleaned_door = door_number.strip()

    query = db.query(models.AddressLookup).filter(
        models.AddressLookup.postcode_clean == cleaned_postcode
    )

    if cleaned_door:
        query = query.filter(models.AddressLookup.house_number.contains(cleaned_door))

    results = query.limit(20).all()
    db.close()

    return [
        {
            "id": r.id,
            "full_address": r.full_address,
            "postcode": r.postcode,
            "house_number": r.house_number,
            "street": r.street,
            "town_city": r.town_city,
            "price": r.price,
            "transfer_date": r.transfer_date,
        }
        for r in results
    ]
@app.get("/epc-suggestions")
def epc_suggestions(postcode: str, door_number: str = ""):
    db = SessionLocal()
    cleaned_postcode = postcode.upper().replace(" ", "").strip()
    cleaned_door = door_number.strip()

    query = db.query(models.EPCProperty).filter(
        models.EPCProperty.postcode_clean == cleaned_postcode
    )

    if cleaned_door:
        query = query.filter(
            (models.EPCProperty.house_number.contains(cleaned_door)) |
            (models.EPCProperty.full_address.contains(cleaned_door))
        )

    results = query.limit(20).all()
    db.close()

    return [
        {
            "id": r.id,
            "full_address": r.full_address,
            "postcode": r.postcode,
            "house_number": r.house_number,
            "epc_rating": r.epc_rating,
            "floor_area": r.floor_area,
        }
        for r in results
    ]


@app.get("/property-search")
def property_search(postcode: str, query: str = "", limit: int = 20):
    db = SessionLocal()
    cleaned_postcode = postcode.upper().replace(" ", "").strip()
    cleaned_query = query.strip()

    rows_query = db.query(models.PropertyFeature).filter(
        models.PropertyFeature.postcode_clean == cleaned_postcode
    )

    if cleaned_query:
        rows_query = rows_query.filter(
            (models.PropertyFeature.full_address.contains(cleaned_query))
            | (models.PropertyFeature.house_number.contains(cleaned_query))
        )

    rows = rows_query.limit(max(1, min(limit, 50))).all()
    db.close()

    return [
        {
            "id": row.id,
            "full_address": row.full_address,
            "postcode": row.postcode,
            "postcode_sector": row.postcode_sector,
            "house_number": row.house_number,
            "latitude": row.latitude,
            "longitude": row.longitude,
            "floor_area": row.floor_area,
            "epc_rating": row.epc_rating,
            "property_subtype": row.property_subtype,
            "last_sold_price": row.last_sold_price,
            "sale_price_source": "exact_matched_sale" if row.last_sold_price is not None else "area_fallback",
        }
        for row in rows
    ]


@app.get("/postcode-suggestions")
def postcode_suggestions(query: str, limit: int = 10):
    cleaned_query = query.upper().replace(" ", "").strip()
    if not cleaned_query:
        return []

    db = SessionLocal()
    rows = (
        db.query(models.PropertyFeature.postcode, models.PropertyFeature.postcode_clean)
        .filter(models.PropertyFeature.postcode_clean.startswith(cleaned_query))
        .distinct()
        .order_by(models.PropertyFeature.postcode_clean)
        .limit(max(1, min(limit, 10)))
        .all()
    )
    db.close()

    return [
        {
            "postcode": row.postcode,
            "postcode_clean": row.postcode_clean,
        }
        for row in rows
    ]


@app.get("/data-coverage")
def data_coverage():
    db = SessionLocal()
    postcode_rows = db.query(models.PropertyFeature.postcode_clean).distinct().all()
    prefixes = sorted(
        {
            row[0][:-3]
            for row in postcode_rows
            if row[0] and len(row[0]) > 3
        }
    )
    postcode_count = db.query(models.PropertyFeature.postcode_clean).distinct().count()
    property_count = db.query(models.PropertyFeature).count()
    db.close()

    return {
        "postcode_prefixes": prefixes,
        "postcode_count": postcode_count,
        "property_count": property_count,
    }


@app.get("/map-config")
def map_config():
    os_maps_api_key = os.getenv("OS_MAPS_API_KEY", "").strip()
    os_maps_layer = os.getenv("OS_MAPS_LAYER", "Light_3857").strip() or "Light_3857"
    return {
        "provider": "ordnance_survey" if os_maps_api_key else "fallback",
        "os_maps_api_key": os_maps_api_key,
        "os_maps_layer": os_maps_layer,
    }


@app.get("/debug-epc-postcode")
def debug_epc_postcode(postcode: str):
    db = SessionLocal()
    cleaned_postcode = postcode.upper().replace(" ", "").strip()

    results = (
        db.query(models.EPCProperty)
        .filter(models.EPCProperty.postcode_clean == cleaned_postcode)
        .limit(20)
        .all()
    )

    db.close()

    return [
        {
            "id": r.id,
            "postcode": r.postcode,
            "postcode_clean": r.postcode_clean,
            "address1": getattr(r, "address1", None),
            "address": getattr(r, "address", None),
            "posttown": getattr(r, "posttown", None),
            "local_authority": getattr(r, "local_authority", None),
            "full_address": r.full_address,
            "house_number": r.house_number,
            "epc_rating": r.epc_rating,
            "floor_area": r.floor_area,
        }
        for r in results
    ]




@app.get("/property-features-preview")
def property_features_preview(limit: int = 20):
    db = SessionLocal()

    rows = db.query(models.PropertyFeature).limit(limit).all()
    db.close()

    return [
        {
            "id": r.id,
            "full_address": r.full_address,
            "postcode": r.postcode,
            "postcode_sector": r.postcode_sector,
            "house_number": r.house_number,
            "last_sold_price": r.last_sold_price,
            "last_transfer_date": r.last_transfer_date,
            "floor_area": r.floor_area,
            "postcode_average_price": r.postcode_average_price,
            "postcode_average_per_sqm": r.postcode_average_per_sqm,
            "epc_rating": r.epc_rating,
            "indexed_last_sold_price": r.indexed_last_sold_price,
            "nearest_station_name": r.nearest_station_name,
            "nearest_station_zone": r.nearest_station_zone,
            "nearest_station_distance_km": r.nearest_station_distance_km,
            "nearest_school_name": r.nearest_school_name,
            "nearest_school_type": r.nearest_school_type,
            "nearest_school_distance_km": r.nearest_school_distance_km,
            "nearest_primary_school_distance_km": r.nearest_primary_school_distance_km,
            "nearest_secondary_school_distance_km": r.nearest_secondary_school_distance_km,
            "nearby_primary_schools_1km": r.nearby_primary_schools_1km,
            "nearby_secondary_schools_2km": r.nearby_secondary_schools_2km,
            "nearest_hospital_name": r.nearest_hospital_name,
            "nearest_hospital_distance_km": r.nearest_hospital_distance_km,
            "crime_lsoa_code": r.crime_lsoa_code,
            "crime_lsoa_name": r.crime_lsoa_name,
            "crime_total_12m": r.crime_total_12m,
            "crime_avg_monthly_12m": r.crime_avg_monthly_12m,
            "crime_level": r.crime_level,
            "london_hpi_current_index": r.london_hpi_current_index,
            "london_hpi_at_last_sale": r.london_hpi_at_last_sale,
            "london_hpi_annual_change_pct": r.london_hpi_annual_change_pct,
        }
        for r in rows
    ]

@app.get("/estimate")
def estimate(property_id: int):
    db = SessionLocal()

    row = db.query(models.PropertyFeature).filter(
        models.PropertyFeature.id == property_id
    ).first()

    if not row:
        db.close()
        return {"error": "Property not found"}

    estimate_value = None
    explanations = []
    baseline_from_area = None
    local_comparable = get_local_comparable_estimate(db, row)

    # Main baseline: postcode sector average per sqm × floor area
    if (
        row.postcode_average_per_sqm is not None
        and row.floor_area is not None
        and row.floor_area > 0
    ):
        baseline_from_area = row.postcode_average_per_sqm * row.floor_area
        explanations.append(
            f"The property size is {row.floor_area:.0f} sqm, and the average price in this postcode sector is about £{row.postcode_average_per_sqm:,.0f} per sqm."
        )
        explanations.append(
            "The baseline estimate multiplies the postcode sector average price per square metre by the property's floor area."
        )

    if local_comparable is not None:
        explanations.append(
            f"A local comparable estimate was calculated from {local_comparable['count']} sold propert{'y' if local_comparable['count'] == 1 else 'ies'} in the {local_comparable['source']}, averaging about £{local_comparable['average_per_sqm']:,.0f} per sqm."
        )

    sale_anchor_price = row.indexed_last_sold_price or row.last_sold_price

    if row.last_sold_price is not None:
        transfer_text = f" on {row.last_transfer_date[:10]}" if row.last_transfer_date else ""
        explanations.append(
            f"The exact matched last sold price is £{row.last_sold_price:,.0f}{transfer_text}."
        )
        if row.indexed_last_sold_price is not None:
            explanations.append(
                f"After house price index adjustment, that sale is treated as approximately £{row.indexed_last_sold_price:,.0f}."
            )
    else:
        explanations.append(
            "No exact matched last sold price is available for this property, so last-sale evidence is not shown as a direct property sale."
        )

    # Blend with matched sold price using recency weighting
    if baseline_from_area is not None and sale_anchor_price is not None:
        sold_weight = get_sale_recency_weight(row.last_transfer_date)
        if baseline_from_area > 0:
            sale_area_ratio = max(sale_anchor_price, baseline_from_area) / min(sale_anchor_price, baseline_from_area)
            if sale_area_ratio >= 1.8:
                sold_weight = max(sold_weight, 0.85)
            elif sale_area_ratio >= 1.35:
                sold_weight = max(sold_weight, 0.8)
        baseline_weight = 1.0 - sold_weight

        area_component = baseline_from_area
        if local_comparable is not None:
            area_component = 0.7 * local_comparable["estimate"] + 0.3 * baseline_from_area
            explanations.append(
                "The area component gives more weight to nearby sold comparables than to the broader postcode-sector average."
            )

        estimate_value = baseline_weight * area_component + sold_weight * sale_anchor_price
        sale_floor = sale_anchor_price * 0.9
        if estimate_value < sale_floor:
            estimate_value = sale_floor
            explanations.append(
                "Because an exact HPI-adjusted sale exists, the baseline is not allowed to fall more than 10% below that sale anchor."
            )

        explanations.append(
            f"The baseline is blended with the property's matched sold price using {int(sold_weight * 100)}% sale weight and {int(baseline_weight * 100)}% postcode-area weight."
        )

        if row.indexed_last_sold_price is not None:
            explanations.append(
                "The matched sold price is adjusted forward using the London house price index before blending."
            )

        if sold_weight >= 0.8:
            explanations.append("The exact matched sold price has a very strong influence because area averages understate this property's market evidence.")
        elif sold_weight >= 0.7:
            explanations.append("A recent matched sold price has a strong influence on the estimate.")
        elif sold_weight >= 0.5:
            explanations.append("A moderately recent matched sold price influences the estimate.")
        else:
            explanations.append("An older matched sold price is used with lower weight in the estimate.")

    elif sale_anchor_price is not None:
        estimate_value = float(sale_anchor_price)
        explanations.append(
            "The estimate is anchored to the property's matched sold price."
        )
        if row.indexed_last_sold_price is not None:
            explanations.append(
                "That sold price is adjusted using the London house price index to reflect more recent market levels."
            )

    elif baseline_from_area is not None:
        if local_comparable is not None:
            estimate_value = 0.75 * local_comparable["estimate"] + 0.25 * baseline_from_area
            explanations.append(
                "No direct sold-price match was available, so the estimate primarily uses nearby sold comparables and then blends in postcode-sector pricing."
            )
        else:
            estimate_value = baseline_from_area
            explanations.append(
                "No direct sold-price match was available, so the estimate relies on postcode sector pricing and floor area."
            )
        explanations.append(
            "The loaded price-paid data does not contain a safe exact last-sold match for this property."
        )

    elif row.postcode_average_price is not None:
        estimate_value = float(row.postcode_average_price)
        explanations.append(
            "The estimate falls back to the average matched sold price in this postcode sector."
        )
        explanations.append(
            "No safe exact last-sold match was found in the currently loaded price-paid data."
        )

    else:
        db.close()
        return {
            "property_id": row.id,
            "full_address": row.full_address,
            "estimated_price": None,
            "price_range": None,
            "explanations": [
                "There is not enough pricing data available yet for this property."
            ],
            "features": {
                "postcode": row.postcode,
                "postcode_sector": row.postcode_sector,
                "house_number": row.house_number,
                "epc_rating": row.epc_rating,
                "floor_area": row.floor_area,
                "last_sold_price": row.last_sold_price,
                "last_transfer_date": row.last_transfer_date,
                "postcode_average_price": row.postcode_average_price,
                "postcode_average_per_sqm": row.postcode_average_per_sqm,
            },
        }

    # Small EPC adjustment
    if row.epc_rating in ["A", "B"]:
        estimate_value *= 1.03
        explanations.append("A stronger EPC rating slightly increases the estimate.")
    elif row.epc_rating in ["F", "G"]:
        estimate_value *= 0.97
        explanations.append("A weaker EPC rating slightly reduces the estimate.")

    estimate_value = apply_location_feature_adjustments(row, estimate_value, explanations)

    station_distance_text = format_distance_km(row.nearest_station_distance_km)
    if row.nearest_station_name and station_distance_text:
        zone_text = f" (Zone {row.nearest_station_zone})" if row.nearest_station_zone else ""
        explanations.append(
            f"The closest station is {row.nearest_station_name}{zone_text}, about {station_distance_text} away."
        )

    school_distance_text = format_distance_km(row.nearest_school_distance_km)
    if row.nearest_school_name and school_distance_text:
        school_type = row.nearest_school_type.lower() if row.nearest_school_type else "school"
        explanations.append(
            f"The closest school is {row.nearest_school_name}, a {school_type}, about {school_distance_text} away."
        )

    if row.london_hpi_current_index is not None:
        if row.london_hpi_annual_change_pct is not None:
            explanations.append(
                f"The London house price index used here is {row.london_hpi_current_index:.1f}, with annual change of {row.london_hpi_annual_change_pct:.1f}%."
            )
        else:
            explanations.append(
                f"The London house price index used here is {row.london_hpi_current_index:.1f}."
            )

    if row.crime_avg_monthly_12m is not None:
        area_name = row.crime_lsoa_name or "the local LSOA"
        level_text = (row.crime_level or "AVERAGE").lower()
        explanations.append(
            f"The crime rate in {area_name} is classified as {level_text}, averaging about {row.crime_avg_monthly_12m:.1f} recorded crimes per month over the latest 12 months."
        )

    estimate_value = int(round(estimate_value))

    result = {
        "property_id": row.id,
        "full_address": row.full_address,
        "estimated_price": estimate_value,
        "base_estimator_prediction": estimate_value,
        "price_range": {
            "min": int(round(estimate_value * 0.95)),
            "max": int(round(estimate_value * 1.05)),
        },
        "explanations": explanations,
        "features": {
            "postcode": row.postcode,
            "postcode_sector": row.postcode_sector,
            "house_number": row.house_number,
            "epc_rating": row.epc_rating,
            "floor_area": row.floor_area,
            "energy_efficiency": row.energy_efficiency,
            "built_form": row.built_form,
            "property_subtype": row.property_subtype,
            "last_sold_price": row.last_sold_price,
            "indexed_last_sold_price": row.indexed_last_sold_price,
            "sale_price_source": "exact_matched_sale" if row.last_sold_price is not None else "area_fallback",
            "last_transfer_date": row.last_transfer_date,
            "postcode_average_price": row.postcode_average_price,
            "postcode_average_per_sqm": row.postcode_average_per_sqm,
            "local_comparable_estimate": local_comparable,
            "latitude": row.latitude,
            "longitude": row.longitude,
            "nearest_station_name": row.nearest_station_name,
            "nearest_station_zone": row.nearest_station_zone,
            "nearest_station_distance_km": row.nearest_station_distance_km,
            "nearest_school_name": row.nearest_school_name,
            "nearest_school_type": row.nearest_school_type,
            "nearest_school_distance_km": row.nearest_school_distance_km,
            "nearest_primary_school_distance_km": row.nearest_primary_school_distance_km,
            "nearest_secondary_school_distance_km": row.nearest_secondary_school_distance_km,
            "nearby_primary_schools_1km": row.nearby_primary_schools_1km,
            "nearby_secondary_schools_2km": row.nearby_secondary_schools_2km,
            "nearest_hospital_name": row.nearest_hospital_name,
            "nearest_hospital_distance_km": row.nearest_hospital_distance_km,
            "map_amenities": get_nearest_map_amenities(row.latitude, row.longitude),
            "crime_lsoa_code": row.crime_lsoa_code,
            "crime_lsoa_name": row.crime_lsoa_name,
            "crime_total_12m": row.crime_total_12m,
            "crime_avg_monthly_12m": row.crime_avg_monthly_12m,
            "crime_level": row.crime_level,
            "london_hpi_current_index": row.london_hpi_current_index,
            "london_hpi_at_last_sale": row.london_hpi_at_last_sale,
            "london_hpi_annual_change_pct": row.london_hpi_annual_change_pct,
        },
    }

    artifact = get_ml_artifact()
    if artifact is None:
        result["ml_predictions"] = {
            "available": False,
            "message": "ML models have not been trained yet. Run scripts/train_property_ml_models.py to create predictions.",
        }
    else:
        raw_ml_prediction = predict_with_artifact(row_to_feature_dict(row), artifact)
        ml_prediction = anchor_ml_to_market_evidence(
            raw_ml_prediction,
            sale_anchor_price if row.last_sold_price is not None else None,
            row.last_transfer_date if row.last_sold_price is not None else None,
            local_comparable,
        )
        blend = blend_baseline_and_ml(estimate_value, ml_prediction)
        mixed_prediction = blend["mixed_prediction"]
        adjusted_ml_prediction = {
            **ml_prediction,
            "raw_confidence_score": ml_prediction["confidence_score"],
            "raw_confidence_label": ml_prediction["confidence_label"],
            "confidence_score": blend["confidence_score"],
            "confidence_label": blend["confidence_label"],
            "baseline_weight_in_mixed": blend["baseline_weight"],
            "ml_weight_in_mixed": blend["ml_weight"],
            "baseline_ml_disagreement_ratio": blend["disagreement_ratio"],
            "blend_warning": blend["warning"],
        }
        result["ml_predictions"] = {
            "available": True,
            **adjusted_ml_prediction,
        }
        result["mixed_prediction"] = mixed_prediction
        result["prediction_summary"] = {
            "baseline_prediction": estimate_value,
            "ml_prediction": ml_prediction["blended_ml_prediction"],
            "mixed_prediction": mixed_prediction,
        }
        result["explanations"].append(
            f"The ML estimate blends K-nearest neighbours and random forest predictions with {blend['confidence_label']} confidence ({blend['confidence_score']})."
        )
        if ml_prediction.get("sale_anchor_adjusted"):
            result["explanations"].append(
                f"The raw ML estimate was £{ml_prediction['raw_blended_ml_prediction']:,}, but it was adjusted towards {ml_prediction.get('sale_anchor_reason', 'strong local market evidence')} because the model under-predicted a high-value property."
            )
        if blend["warning"]:
            result["explanations"].append(blend["warning"])
        result["explanations"].append(
            f"The mixed prediction uses {int(blend['baseline_weight'] * 100)}% baseline and {int(blend['ml_weight'] * 100)}% ML weighting, giving £{mixed_prediction:,}."
        )

    db.close()
    return result


if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
