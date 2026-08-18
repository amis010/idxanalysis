import json
import ssl
import sys
import urllib.request
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

# Week 6 - Feature Engineering and Market Metrics.
# Takes the week 4-5 cleaned Sold and Listing datasets and engineers the
# market indicators (price ratios, PPSF, time-to-close fields, school
# district) that power the Tableau dashboards, then produces segmented
# summary tables grouped by key dimensions.

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_DIR = PROJECT_ROOT / "outputs" / "week4_5_cleaned"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week6_features"

SCHOOL_DISTRICT_GEOJSON_URL = (
    "https://hub.arcgis.com/api/v3/datasets/b0e3b936426a47ce9d9a2e77e2bb86cc_0/downloads/data"
    "?format=geojson&spatialRefId=4326&where=1%3D1"
)

METRIC_COLUMNS = [
    "PriceRatio", "CloseToOriginalListRatio", "PricePerSqFt", "DaysOnMarket",
    "Year", "Month", "YrMo", "ListingToContractDays", "ContractToCloseDays",
    "SchoolDistrictName", "SchoolDistrictType",
]

# Segment dimensions requested for Week 6, as (group_columns, label) pairs.
SEGMENT_GROUPS = [
    (["PropertyType", "PropertySubType"], "property_type"),
    (["CountyOrParish", "MLSAreaMajor"], "county_area"),
    (["ListOfficeName", "BuyerOfficeName"], "office"),
]

SEGMENT_METRICS = [
    "PriceRatio", "CloseToOriginalListRatio", "PricePerSqFt",
    "DaysOnMarket", "ListingToContractDays", "ContractToCloseDays",
]


def safe_ratio(numerator, denominator):
    """Element-wise numerator / denominator, NaN wherever the denominator
    is missing or not strictly positive (avoids division by zero / negative
    denominators producing misleading ratios)."""
    denominator_valid = denominator.where(denominator > 0)
    return numerator / denominator_valid


def add_price_metrics(df):
    df = df.copy()

    df["PriceRatio"] = safe_ratio(df["ClosePrice"], df["OriginalListPrice"])
    # Same formula as PriceRatio - the Week 6 spec lists it as a distinct
    # deliverable column ("Close to Original List Ratio"), so both are kept.
    df["CloseToOriginalListRatio"] = df["PriceRatio"]
    df["PricePerSqFt"] = safe_ratio(df["ClosePrice"], df["LivingArea"])

    return df


def add_time_metrics(df):
    df = df.copy()

    close_date = pd.to_datetime(df["CloseDate"], errors="coerce")
    df["Year"] = close_date.dt.year
    df["Month"] = close_date.dt.month
    df["YrMo"] = close_date.dt.to_period("M").astype(str)
    df.loc[close_date.isna(), "YrMo"] = pd.NA

    listing_date = pd.to_datetime(df["ListingContractDate"], errors="coerce")
    contract_date = pd.to_datetime(df["PurchaseContractDate"], errors="coerce")

    df["ListingToContractDays"] = (contract_date - listing_date).dt.days
    df["ContractToCloseDays"] = (close_date - contract_date).dt.days

    return df


def fetch_school_districts(url=SCHOOL_DISTRICT_GEOJSON_URL):
    """Fetch CA school district boundary polygons (2024-25) from the
    California Open Data / ArcGIS Hub GeoJSON endpoint. Returns a list of
    (polygon, properties) tuples and an STRtree built over the polygons for
    fast point-in-polygon lookup."""
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=context, timeout=60) as response:
        geojson_bytes = response.read()

    geojson = json.loads(geojson_bytes)
    features = geojson["features"]

    polygons = [shape(feature["geometry"]) for feature in features]
    properties = [feature["properties"] for feature in features]

    tree = STRtree(polygons)

    return tree, properties


def add_school_district(df, tree, properties):
    """Spatial-join each row's (Longitude, Latitude) point against the CA
    school district polygons. Rows with missing/invalid coordinates (already
    flagged in week 4-5 as missing_coordinates_flag / implausible_coordinates_flag)
    simply get no match and are left as NaN here."""
    df = df.copy()

    df["SchoolDistrictName"] = pd.NA
    df["SchoolDistrictType"] = pd.NA

    has_coords = df["Latitude"].notna() & df["Longitude"].notna()
    coord_rows = df.index[has_coords]

    if len(coord_rows) == 0:
        return df

    points = np.array(
        [Point(lon, lat) for lon, lat in zip(df.loc[coord_rows, "Longitude"], df.loc[coord_rows, "Latitude"])],
        dtype=object,
    )

    query_idx, tree_idx = tree.query(points, predicate="intersects")

    # A point can straddle two polygons at a shared boundary; keep the first
    # match per point, which is enough for a district label.
    _, first_match_pos = np.unique(query_idx, return_index=True)
    query_idx = query_idx[first_match_pos]
    tree_idx = tree_idx[first_match_pos]

    matched_rows = coord_rows[query_idx]
    df.loc[matched_rows, "SchoolDistrictName"] = [properties[i]["DistrictName"] for i in tree_idx]
    df.loc[matched_rows, "SchoolDistrictType"] = [properties[i]["DistrictType"] for i in tree_idx]

    return df


def build_segment_summary(df, group_columns, dataset_name):
    summary = (
        df.groupby(group_columns, dropna=False)[SEGMENT_METRICS]
        .agg(["mean", "median", "count"])
    )
    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.reset_index()

    print(f"\n{dataset_name}: segment summary grouped by {group_columns} "
          f"({len(summary):,} segment(s)):")
    print(summary.head())

    return summary


def run_feature_engineering(input_path, dataset_name, output_prefix, tree, properties):
    print(f"\n{'=' * 70}\nEngineering features - {dataset_name}\n{'=' * 70}")

    df = pd.read_csv(input_path, low_memory=False)
    print(f"{dataset_name}: loaded {len(df):,} rows, {len(df.columns)} columns from {input_path.name}")

    print("\nStep 1: Computing price ratio metrics (PriceRatio, CloseToOriginalListRatio, PricePerSqFt)...")
    df = add_price_metrics(df)

    print("\nStep 2: Deriving time-based metrics (Year, Month, YrMo, ListingToContractDays, ContractToCloseDays)...")
    df = add_time_metrics(df)

    print("\nStep 3: Joining school district by (Latitude, Longitude)...")
    df = add_school_district(df, tree, properties)
    matched = df["SchoolDistrictName"].notna().sum()
    print(f"{dataset_name}: matched {matched:,} of {len(df):,} rows to a school district.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features_path = OUTPUT_DIR / f"{output_prefix}_residential_features.csv"
    df.to_csv(features_path, index=False)
    print(f"\nSaved engineered dataset: {features_path}")

    print(f"\nSample output - {dataset_name} rows with populated engineered metrics:")
    sample_columns = [
        "ListingKey", "ClosePrice", "OriginalListPrice", "LivingArea",
    ] + METRIC_COLUMNS
    sample = df.dropna(subset=["PriceRatio", "PricePerSqFt", "SchoolDistrictName"]).head(10)
    print(sample[sample_columns].to_string(index=False))

    for group_columns, label in SEGMENT_GROUPS:
        summary = build_segment_summary(df, group_columns, dataset_name)
        summary_path = OUTPUT_DIR / f"segment_summary_{label}_{output_prefix}.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Saved segment summary: {summary_path}")

    return df


def main():
    print("Fetching CA school district boundaries (2024-25) for spatial join...")
    tree, properties = fetch_school_districts()
    print(f"Fetched {len(properties):,} school district polygons.")

    run_feature_engineering(
        CLEANED_DIR / "sold_residential_cleaned.csv", "Sold", "sold", tree, properties,
    )
    run_feature_engineering(
        CLEANED_DIR / "listing_residential_cleaned.csv", "Listing", "listing", tree, properties,
    )


if __name__ == "__main__":
    main()
