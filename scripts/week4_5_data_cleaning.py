import sys
from pathlib import Path

import pandas as pd

# Weeks 4-5 - Data Cleaning and Preparation.
# Takes the week 3 mortgage-rate-enriched Sold and Listing datasets and
# produces analysis-ready cleaned datasets. Every invalid-data check below
# FLAGS the offending rows (adds a boolean column) rather than dropping
# them, so row counts never change here - later analysis steps decide
# whether/how to exclude flagged rows.

sys.path.insert(0, str(Path(__file__).resolve().parent))
from week2_eda import CORE_FIELDS, NUMERIC_FIELDS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENRICHED_DIR = PROJECT_ROOT / "outputs" / "week3_enriched"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week4_5_cleaned"

HIGH_MISSING_THRESHOLD = 90.0

DATE_FIELDS = [
    "CloseDate", "PurchaseContractDate", "ListingContractDate", "ContractStatusChangeDate",
]

# Standard rectangular bounding box covering all of California.
CA_LAT_BOUNDS = (32.5, 42.0)
CA_LON_BOUNDS = (-124.5, -114.0)

FLAG_COLUMNS = [
    "close_price_invalid_flag", "living_area_invalid_flag", "days_on_market_invalid_flag",
    "bedrooms_invalid_flag", "bathrooms_invalid_flag",
    "listing_after_close_flag", "purchase_after_close_flag", "negative_timeline_flag",
    "missing_coordinates_flag", "zero_coordinates_flag", "positive_longitude_flag",
    "implausible_coordinates_flag",
]


def convert_date_fields(df, dataset_name):
    df = df.copy()

    for field in DATE_FIELDS:
        if field not in df.columns:
            continue

        before_dtype = df[field].dtype
        before_na = df[field].isna().sum()

        df[field] = pd.to_datetime(df[field], errors="coerce")

        newly_invalid = df[field].isna().sum() - before_na
        print(f"{dataset_name}: {field} converted {before_dtype} -> {df[field].dtype}"
              f" ({newly_invalid} value(s) could not be parsed and became NaT)")

    return df


def drop_high_missing_columns(df, dataset_name):
    """Drop columns above HIGH_MISSING_THRESHOLD missing, same rule week 2
    already applies - except CORE_FIELDS are never dropped for missingness
    alone, since they're the fields the rest of the analysis depends on."""
    missing_pct = (df.isnull().sum() / len(df)) * 100
    high_missing = missing_pct[missing_pct > HIGH_MISSING_THRESHOLD]

    to_drop = [c for c in high_missing.index if c not in CORE_FIELDS]
    protected = [c for c in high_missing.index if c in CORE_FIELDS]

    print(f"\n{dataset_name}: dropping {len(to_drop)} column(s) above "
          f"{HIGH_MISSING_THRESHOLD}% missing:")
    print(to_drop)
    if protected:
        print(f"{dataset_name}: core field(s) also above {HIGH_MISSING_THRESHOLD}% missing "
              f"but retained: {protected}")

    dropped_report = high_missing.loc[to_drop].rename("missing_pct").to_frame()
    dropped_report["dropped"] = True

    return df.drop(columns=to_drop), dropped_report


def ensure_numeric_types(df, dataset_name):
    df = df.copy()

    for field in NUMERIC_FIELDS:
        if field not in df.columns:
            continue

        before_dtype = df[field].dtype
        before_na = df[field].isna().sum()

        df[field] = pd.to_numeric(df[field], errors="coerce")

        coerced = df[field].isna().sum() - before_na
        print(f"{dataset_name}: {field} dtype {before_dtype} -> {df[field].dtype} "
              f"({coerced} value(s) coerced to NaN as non-numeric)")

    return df


def _conditional_flag(index, mask_present, condition_result):
    """Boolean flag that is True/False where mask_present is True, and
    pd.NA (not applicable - can't evaluate) elsewhere."""
    flag = pd.Series(pd.NA, index=index, dtype="boolean")
    flag.loc[mask_present] = condition_result
    return flag


def flag_invalid_numerics(df):
    df = df.copy()

    checks = [
        ("close_price_invalid_flag", "ClosePrice", lambda s: s <= 0),
        ("living_area_invalid_flag", "LivingArea", lambda s: s <= 0),
        ("days_on_market_invalid_flag", "DaysOnMarket", lambda s: s < 0),
        ("bedrooms_invalid_flag", "BedroomsTotal", lambda s: s < 0),
        ("bathrooms_invalid_flag", "BathroomsTotalInteger", lambda s: s < 0),
    ]

    for flag_name, field, condition in checks:
        if field not in df.columns:
            continue

        present = df[field].notna()
        df[flag_name] = _conditional_flag(df.index, present, condition(df.loc[present, field]))

    return df


def flag_date_consistency(df):
    """listing_after_close_flag / purchase_after_close_flag / negative_timeline_flag
    mark violations of the expected order ListingContractDate -> PurchaseContractDate
    -> CloseDate. NaN when either date needed for the comparison is missing."""
    df = df.copy()

    def order_violation(earlier_field, later_field):
        both_present = df[earlier_field].notna() & df[later_field].notna()
        violation = df.loc[both_present, earlier_field] > df.loc[both_present, later_field]
        return _conditional_flag(df.index, both_present, violation)

    df["listing_after_close_flag"] = order_violation("ListingContractDate", "CloseDate")
    df["purchase_after_close_flag"] = order_violation("PurchaseContractDate", "CloseDate")
    df["negative_timeline_flag"] = order_violation("ListingContractDate", "PurchaseContractDate")

    return df


def flag_geo_quality(df):
    df = df.copy()

    lat_missing = df["Latitude"].isna()
    lon_missing = df["Longitude"].isna()
    df["missing_coordinates_flag"] = lat_missing | lon_missing

    both_present = ~lat_missing & ~lon_missing
    lat, lon = df.loc[both_present, "Latitude"], df.loc[both_present, "Longitude"]

    df["zero_coordinates_flag"] = _conditional_flag(
        df.index, both_present, (lat == 0) | (lon == 0)
    )
    df["positive_longitude_flag"] = _conditional_flag(
        df.index, both_present, lon > 0
    )

    lat_min, lat_max = CA_LAT_BOUNDS
    lon_min, lon_max = CA_LON_BOUNDS
    df["implausible_coordinates_flag"] = _conditional_flag(
        df.index, both_present,
        (lat < lat_min) | (lat > lat_max) | (lon < lon_min) | (lon > lon_max),
    )

    return df


def summarize_flags(df, dataset_name):
    rows = {}

    for col in FLAG_COLUMNS:
        if col not in df.columns:
            continue

        rows[col] = {
            "true_count": (df[col] == True).sum(),   # noqa: E712
            "false_count": (df[col] == False).sum(),  # noqa: E712
            "not_applicable_count": df[col].isna().sum(),
        }

    summary = pd.DataFrame.from_dict(rows, orient="index")
    print(f"\n{dataset_name}: flag summary (true / false / not-applicable counts):")
    print(summary)

    return summary


def run_cleaning(input_path, dataset_name, output_prefix):
    print(f"\n{'=' * 70}\nCleaning {dataset_name}\n{'=' * 70}")

    df = pd.read_csv(input_path, low_memory=False)
    rows_before, cols_before = len(df), len(df.columns)
    print(f"{dataset_name}: loaded {rows_before:,} rows, {cols_before} columns from {input_path.name}")

    print(f"\nStep 1: Converting date fields to datetime...")
    df = convert_date_fields(df, dataset_name)

    print(f"\nStep 2: Removing redundant columns (>{HIGH_MISSING_THRESHOLD}% missing)...")
    df, dropped_report = drop_high_missing_columns(df, dataset_name)

    print(f"\nStep 3: Ensuring numeric fields are properly typed...")
    df = ensure_numeric_types(df, dataset_name)

    print(f"\nStep 4: Flagging invalid numeric values...")
    df = flag_invalid_numerics(df)

    print(f"\nStep 5: Flagging date consistency violations...")
    df = flag_date_consistency(df)

    print(f"\nStep 6: Flagging geographic data quality issues...")
    df = flag_geo_quality(df)

    rows_after, cols_after = len(df), len(df.columns)
    flags_added = cols_after - (cols_before - len(dropped_report))

    print(f"\n{dataset_name}: {rows_before:,} -> {rows_after:,} rows "
          "(unchanged - invalid/inconsistent data is flagged, not dropped)")
    print(f"{dataset_name}: {cols_before} -> {cols_after} columns "
          f"({len(dropped_report)} dropped for high missingness, {flags_added} flag columns added)")

    flag_summary = summarize_flags(df, dataset_name)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_path = OUTPUT_DIR / f"{output_prefix}_residential_cleaned.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"\nSaved cleaned dataset: {cleaned_path}")

    dropped_path = OUTPUT_DIR / f"dropped_columns_{output_prefix}.csv"
    dropped_report.to_csv(dropped_path)
    print(f"Saved dropped-columns report: {dropped_path}")

    flag_summary_path = OUTPUT_DIR / f"flag_summary_{output_prefix}.csv"
    flag_summary.to_csv(flag_summary_path)
    print(f"Saved flag summary: {flag_summary_path}")

    return df


def main():
    run_cleaning(
        ENRICHED_DIR / "sold_residential_with_mortgage_rate.csv", "Sold", "sold",
    )
    run_cleaning(
        ENRICHED_DIR / "listing_residential_with_mortgage_rate.csv", "Listing", "listing",
    )


if __name__ == "__main__":
    main()
