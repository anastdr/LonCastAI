from sqlalchemy import Column, Integer, String, Float
from .database import Base

class Address(Base):
    __tablename__ = "addresses"

    id = Column(Integer, primary_key=True, index=True)
    postcode = Column(String, index=True)
    house_number = Column(String, index=True)
    street = Column(String, nullable=True)
    full_address = Column(String, nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

class LondonPostcode(Base):
    __tablename__ = "london_postcodes"

    id = Column(Integer, primary_key=True, index=True)
    postcode = Column(String, index=True)
    postcode_with_space = Column(String)
    postcode_clean = Column(String, unique=True, index=True)
    local_authority_code = Column(String, index=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

class AddressLookup(Base):
    __tablename__ = "address_lookup"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, unique=True, index=True, nullable=True)

    postcode = Column(String, index=True)
    postcode_clean = Column(String, index=True)

    house_number = Column(String, index=True, nullable=True)
    building_name = Column(String, nullable=True)
    unit = Column(String, nullable=True)
    street = Column(String, nullable=True)
    locality = Column(String, nullable=True)
    town_city = Column(String, nullable=True)
    district = Column(String, nullable=True)
    county = Column(String, nullable=True)

    full_address = Column(String, index=True)

    price = Column(Integer, nullable=True)
    transfer_date = Column(String, nullable=True)
    property_type = Column(String, nullable=True)
    new_build_flag = Column(String, nullable=True)
    tenure = Column(String, nullable=True)

class EPCProperty(Base):
    __tablename__ = "epc_properties"

    id = Column(Integer, primary_key=True, index=True)

    postcode = Column(String, index=True)
    postcode_clean = Column(String, index=True)

    address1 = Column(String, nullable=True)
    address = Column(String, nullable=True)
    posttown = Column(String, nullable=True)
    local_authority = Column(String, nullable=True)

    full_address = Column(String, index=True)

    house_number = Column(String, index=True, nullable=True)

    epc_rating = Column(String, nullable=True)
    floor_area = Column(Float, nullable=True)
    energy_efficiency = Column(Float, nullable=True)
    built_form = Column(String, nullable=True)
    property_subtype = Column(String, nullable=True)
    lodgement_date = Column(String, nullable=True)

    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

class PropertyFeature(Base):
    __tablename__ = "property_features"

    id = Column(Integer, primary_key=True, index=True)

    full_address = Column(String, index=True)
    postcode = Column(String, index=True)
    postcode_clean = Column(String, index=True)
    postcode_sector = Column(String, nullable=True, index=True)
    house_number = Column(String, nullable=True)

    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    epc_rating = Column(String, nullable=True)
    floor_area = Column(Float, nullable=True)
    energy_efficiency = Column(Float, nullable=True)
    built_form = Column(String, nullable=True)
    property_subtype = Column(String, nullable=True)

    last_sold_price = Column(Integer, nullable=True)
    last_transfer_date = Column(String, nullable=True)
    indexed_last_sold_price = Column(Float, nullable=True)

    postcode_average_price = Column(Float, nullable=True)
    postcode_average_per_sqm = Column(Float, nullable=True)

    nearest_station_name = Column(String, nullable=True)
    nearest_station_zone = Column(String, nullable=True)
    nearest_station_distance_km = Column(Float, nullable=True)

    nearest_school_name = Column(String, nullable=True)
    nearest_school_type = Column(String, nullable=True)
    nearest_school_distance_km = Column(Float, nullable=True)
    nearest_primary_school_distance_km = Column(Float, nullable=True)
    nearest_secondary_school_distance_km = Column(Float, nullable=True)
    nearby_primary_schools_1km = Column(Integer, nullable=True)
    nearby_secondary_schools_2km = Column(Integer, nullable=True)

    nearest_hospital_name = Column(String, nullable=True)
    nearest_hospital_distance_km = Column(Float, nullable=True)

    crime_lsoa_code = Column(String, nullable=True)
    crime_lsoa_name = Column(String, nullable=True)
    crime_total_12m = Column(Float, nullable=True)
    crime_avg_monthly_12m = Column(Float, nullable=True)
    crime_level = Column(String, nullable=True)

    london_hpi_current_index = Column(Float, nullable=True)
    london_hpi_at_last_sale = Column(Float, nullable=True)
    london_hpi_annual_change_pct = Column(Float, nullable=True)
