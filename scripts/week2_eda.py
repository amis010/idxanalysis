import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Week 2-3 - Foundational EDA on the combined Sold dataset.
# Step 1: Dataset Understanding.
# Runs on the UNFILTERED combined Sold data so property types and
# missing-value patterns can be documented before the Residential filter
# (already applied separately in week1_monthly_aggregation.py) is examined.

sys.path.insert(0, str(Path(__file__).resolve().parent))
from week1_monthly_aggregation import (
    RAW_DATA_DIR,
    SOLD_PATTERN,
    load_and_concat_files,
    filter_residential,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"

HIGH_MISSING_THRESHOLD = 90.0

# Core fields to retain even if partially missing (never auto-dropped
# for missingness alone).
CORE_FIELDS = [
    "PropertyType", "CountyOrParish", "ClosePrice", "ListPrice",
    "OriginalListPrice", "LivingArea", "LotSizeAcres", "BedroomsTotal",
    "BathroomsTotalInteger", "DaysOnMarket", "YearBuilt",
]

# Administrative / system metadata fields, not used for market analysis.
# Everything else in the dataset is treated as a market analysis field.
NUMERIC_FIELDS = [
    "ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea",
    "LotSizeAcres", "BedroomsTotal", "BathroomsTotalInteger",
    "DaysOnMarket", "YearBuilt",
]

METADATA_FIELDS = [
    "ListingKey", "ListingKeyNumeric", "ListingId", "SourceFile",
    "OriginatingSystemName", "OriginatingSystemSubName", "BuyerAgentAOR",
    "ListAgentAOR", "BuyerOfficeAOR", "ListAgentEmail", "ListAgentFirstName",
    "ListAgentLastName", "ListAgentFullName", "CoListAgentFirstName",
    "CoListAgentLastName", "BuyerAgentMlsId", "BuyerAgentFirstName",
    "BuyerAgentLastName", "CoBuyerAgentFirstName", "ListOfficeName",
    "BuyerOfficeName", "CoListOfficeName", "BuilderName", "ElementarySchool",
    "ElementarySchoolDistrict", "MiddleOrJuniorSchool",
    "MiddleOrJuniorSchoolDistrict", "HighSchool", "HighSchoolDistrict",
    "BuyerAgencyCompensationType", "BuyerAgencyCompensation", "latfilled",
    "lonfilled", "Latitude", "Longitude", "StreetNumberNumeric",
    "UnparsedAddress",
]


def print_shape(df):
    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")


def print_dtypes(df):
    print("\nColumn dtypes:")
    print(df.dtypes)


def missing_value_report(df):
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100

    report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct,
    }).sort_values("missing_pct", ascending=False)

    report["flagged_high_missing"] = report["missing_pct"] > HIGH_MISSING_THRESHOLD

    return report


def print_missing_overview(report):
    print("\nMissing values by column (sorted descending, first look):")
    print(report)


def print_missing_value_analysis(report):
    high_missing = report[report["flagged_high_missing"]].index.tolist()

    print(f"\nColumns above {HIGH_MISSING_THRESHOLD}% missing (flagged):")
    print(high_missing)

    core_flagged = [c for c in high_missing if c in CORE_FIELDS]
    print(f"\nCore fields retained regardless of missingness: {CORE_FIELDS}")
    if core_flagged:
        print(f"NOTE: core fields also above {HIGH_MISSING_THRESHOLD}% missing (still retained): {core_flagged}")

    print(f"\nDecision: {len(high_missing)} column(s) flagged as high-missing for awareness only. "
          "None are dropped here - core fields are always retained even if partially missing, "
          "and non-core high-missing columns are left in place pending a later cleaning step.")


def print_property_types(df):
    print("\nUnique PropertyType values found:")
    print(df["PropertyType"].unique())


def print_field_split(df):
    metadata_fields = [c for c in df.columns if c in METADATA_FIELDS]
    market_analysis_fields = [c for c in df.columns if c not in METADATA_FIELDS]

    print(f"\nMetadata fields ({len(metadata_fields)}):")
    print(metadata_fields)

    print(f"\nMarket analysis fields ({len(market_analysis_fields)}):")
    print(market_analysis_fields)


def numeric_distribution_summary(df, columns):
    rows = {}

    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()

        rows[column] = {
            "count": series.count(),
            "min": series.min(),
            "p25": series.quantile(0.25),
            "median": series.median(),
            "mean": series.mean(),
            "p75": series.quantile(0.75),
            "p95": series.quantile(0.95),
            "p99": series.quantile(0.99),
            "max": series.max(),
            "std": series.std(),
        }

    return pd.DataFrame.from_dict(rows, orient="index")


def save_distribution_plots(df, columns, plots_dir):
    plots_dir.mkdir(parents=True, exist_ok=True)

    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()

        fig, ax = plt.subplots()
        ax.hist(series, bins=50)
        ax.set_title(f"{column} - Histogram")
        fig.savefig(plots_dir / f"{column}_hist.png")
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.boxplot(series, vert=False)
        ax.set_title(f"{column} - Boxplot")
        fig.savefig(plots_dir / f"{column}_box.png")
        plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading combined (unfiltered) Sold dataset for EDA - Step 1: Dataset Understanding...")
    sold = load_and_concat_files(file_pattern=SOLD_PATTERN, dataset_name="Sold")

    print_shape(sold)
    print_dtypes(sold)

    report = missing_value_report(sold)
    print_missing_overview(report)
    print_property_types(sold)
    print_field_split(sold)

    print("\nStep 2: Missing Value Analysis...")
    print_missing_value_analysis(report)

    report_path = OUTPUT_DIR / "missing_value_report.csv"
    report.to_csv(report_path)
    print(f"\nSaved missing value report: {report_path}")

    print("\nStep 3: Filter to Residential and save filtered dataset...")
    sold_residential = filter_residential(sold, dataset_name="Sold")

    filtered_path = OUTPUT_DIR / "sold_residential_filtered.csv"
    sold_residential.to_csv(filtered_path, index=False)
    print(f"\nSaved filtered Residential dataset: {filtered_path}")

    print("\nStep 4: Numeric Distribution Review...")
    distribution_summary = numeric_distribution_summary(sold_residential, NUMERIC_FIELDS)
    print(distribution_summary)

    distribution_path = OUTPUT_DIR / "numeric_distribution_summary.csv"
    distribution_summary.to_csv(distribution_path)
    print(f"\nSaved numeric distribution summary: {distribution_path}")

    plots_dir = OUTPUT_DIR / "plots"
    save_distribution_plots(sold_residential, NUMERIC_FIELDS, plots_dir)
    print(f"\nSaved distribution plots to: {plots_dir}")


if __name__ == "__main__":
    main()
