import ssl
import sys
import urllib.request
from io import StringIO
from pathlib import Path

import certifi
import pandas as pd

# Week 3 - Mortgage Rate Enrichment.
# Enrich the combined Sold and Listing (Residential) datasets with the
# national 30-year fixed mortgage rate (FRED series MORTGAGE30US),
# joined on a year-month key derived from transaction dates.

sys.path.insert(0, str(Path(__file__).resolve().parent))
from week1_monthly_aggregation import (
    SOLD_PATTERN,
    LISTING_PATTERN,
    load_and_concat_files,
    filter_residential,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week3_enriched"

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"


def fetch_monthly_mortgage_rate(url=FRED_URL):
    """Fetch the weekly MORTGAGE30US series from FRED and resample it to
    monthly averages, keyed by year_month (a pandas Period)."""
    # Fetch via urllib with certifi's CA bundle rather than pd.read_csv(url)
    # directly, since some Python installs (e.g. python.org builds on macOS)
    # don't have a working default trust store for HTTPS.
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=context, timeout=30) as response:
        csv_text = response.read().decode("utf-8")

    mortgage = pd.read_csv(StringIO(csv_text), parse_dates=["observation_date"])
    mortgage.columns = ["date", "rate_30yr_fixed"]

    mortgage["year_month"] = mortgage["date"].dt.to_period("M")

    mortgage_monthly = (
        mortgage.groupby("year_month")["rate_30yr_fixed"]
        .mean()
        .reset_index()
    )

    return mortgage_monthly


def add_year_month(df, date_column):
    df = df.copy()
    df["year_month"] = pd.to_datetime(df[date_column]).dt.to_period("M")
    return df


def merge_mortgage_rate(df, mortgage_monthly):
    return df.merge(mortgage_monthly, on="year_month", how="left")


def validate_no_null_rates(df, dataset_name):
    null_count = df["rate_30yr_fixed"].isnull().sum()
    print(f"\n{dataset_name}: {null_count} row(s) with null rate_30yr_fixed after merge.")

    if null_count > 0:
        unmatched_months = (
            df.loc[df["rate_30yr_fixed"].isnull(), "year_month"]
            .unique()
        )
        raise ValueError(
            f"{dataset_name}: merge validation failed - {null_count} row(s) have no "
            f"matching mortgage rate. Unmatched year_month value(s): {sorted(unmatched_months)}"
        )

    print(f"{dataset_name}: validation passed - no null rate values after merge.")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1: Fetching MORTGAGE30US from FRED and resampling to monthly averages...")
    mortgage_monthly = fetch_monthly_mortgage_rate()
    print(f"Fetched {len(mortgage_monthly):,} monthly rate observations "
          f"({mortgage_monthly['year_month'].min()} to {mortgage_monthly['year_month'].max()}).")

    print("\nStep 2: Loading combined Sold and Listing datasets and filtering to Residential...")
    sold = filter_residential(
        load_and_concat_files(file_pattern=SOLD_PATTERN, dataset_name="Sold"),
        dataset_name="Sold",
    )
    listings = filter_residential(
        load_and_concat_files(file_pattern=LISTING_PATTERN, dataset_name="Listing"),
        dataset_name="Listing",
    )

    print("\nStep 3: Creating year_month join keys...")
    sold = add_year_month(sold, "CloseDate")
    listings = add_year_month(listings, "ListingContractDate")

    print("\nStep 4: Merging mortgage rates onto both datasets...")
    sold_with_rates = merge_mortgage_rate(sold, mortgage_monthly)
    listings_with_rates = merge_mortgage_rate(listings, mortgage_monthly)

    print("\nStep 5: Validating merge completeness...")
    validate_no_null_rates(sold_with_rates, "Sold")
    validate_no_null_rates(listings_with_rates, "Listings")

    print("\nPreview - Sold with rates:")
    print(sold_with_rates[["CloseDate", "year_month", "ClosePrice", "rate_30yr_fixed"]].head())

    print("\nPreview - Listings with rates:")
    print(listings_with_rates[["ListingContractDate", "year_month", "ListPrice", "rate_30yr_fixed"]].head())

    sold_path = OUTPUT_DIR / "sold_residential_with_mortgage_rate.csv"
    listings_path = OUTPUT_DIR / "listing_residential_with_mortgage_rate.csv"

    sold_with_rates.to_csv(sold_path, index=False)
    listings_with_rates.to_csv(listings_path, index=False)

    print(f"\nSaved enriched Sold dataset: {sold_path}")
    print(f"Saved enriched Listing dataset: {listings_path}")


if __name__ == "__main__":
    main()
