import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Week 2 - Foundational EDA on the combined Sold and Listing datasets.
# Step 1: Dataset Understanding.
# Runs on the UNFILTERED combined data (both Sold and Listing) so property
# types and missing-value patterns can be documented before the Residential
# filter is examined.

sys.path.insert(0, str(Path(__file__).resolve().parent))
from week1_monthly_aggregation import (
    SOLD_PATTERN,
    LISTING_PATTERN,
    load_and_concat_files,
    filter_residential,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"
PLOTS_DIR = OUTPUT_DIR / "plots"

HIGH_MISSING_THRESHOLD = 90.0
TOP_N_COUNTIES = 15

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


def property_type_share(df):
    counts = df["PropertyType"].value_counts()
    pct = (counts / len(df) * 100).round(2)
    return pd.DataFrame({"count": counts, "pct": pct})


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


def outlier_summary(df, columns):
    """Tukey/IQR rule: values beyond 1.5x the interquartile range past
    Q1/Q3 are flagged as extreme outliers, for later handling in week 4-5."""
    rows = {}

    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()

        if series.empty:
            rows[column] = {
                "lower_bound": None, "upper_bound": None,
                "outlier_count": 0, "outlier_pct": 0.0,
            }
            continue

        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        outliers = series[(series < lower_bound) | (series > upper_bound)]

        rows[column] = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": len(outliers),
            "outlier_pct": (len(outliers) / len(series)) * 100,
        }

    return pd.DataFrame.from_dict(rows, orient="index")


def save_distribution_plots(df, columns, plots_dir, dataset_name):
    """Histograms are clipped to the 1st-99th percentile so the chart isn't
    dominated by a handful of extreme values (e.g. a $989M ClosePrice) -
    those extremes are exactly what the boxplot and outlier_summary() above
    are for. Boxplots are drawn on the raw values so outliers still show."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()

        if series.empty:
            continue

        p01, p99 = series.quantile(0.01), series.quantile(0.99)
        clipped = series[(series >= p01) & (series <= p99)]

        fig, ax = plt.subplots()
        ax.hist(clipped, bins=50)
        ax.set_title(f"{dataset_name}: {column} (1st-99th percentile)")
        fig.savefig(plots_dir / f"{column}_hist.png")
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.boxplot(series.values, vert=False)
        ax.set_title(f"{dataset_name}: {column} (raw, outliers shown)")
        fig.savefig(plots_dir / f"{column}_box.png")
        plt.close(fig)


def print_close_vs_list(df, dataset_name):
    valid = df.dropna(subset=["ClosePrice", "ListPrice"])
    valid = valid[(valid["ClosePrice"] > 0) & (valid["ListPrice"] > 0)]
    total = len(valid)

    if total == 0:
        print(f"\n{dataset_name}: no rows with both ClosePrice and ListPrice present.")
        return

    above = (valid["ClosePrice"] > valid["ListPrice"]).sum()
    below = (valid["ClosePrice"] < valid["ListPrice"]).sum()
    at_list = (valid["ClosePrice"] == valid["ListPrice"]).sum()

    print(f"\n{dataset_name}: of {total:,} rows with both ClosePrice and ListPrice > 0 - "
          f"{above:,} ({above / total * 100:.1f}%) closed above list, "
          f"{below:,} ({below / total * 100:.1f}%) closed below list, "
          f"{at_list:,} ({at_list / total * 100:.1f}%) closed at list.")


def print_date_consistency_spotcheck(df, dataset_name):
    """Lightweight awareness check only - full flag columns for date
    consistency are built out in the week 4-5 cleaning script."""
    valid = df.dropna(subset=["ListingContractDate", "CloseDate"]).copy()

    if valid.empty:
        print(f"\n{dataset_name}: no rows with both ListingContractDate and CloseDate present.")
        return

    valid["ListingContractDate"] = pd.to_datetime(valid["ListingContractDate"], errors="coerce")
    valid["CloseDate"] = pd.to_datetime(valid["CloseDate"], errors="coerce")

    violations = (valid["CloseDate"] < valid["ListingContractDate"]).sum()

    print(f"\n{dataset_name}: {violations:,} of {len(valid):,} rows have CloseDate before "
          "ListingContractDate (apparent date consistency issue).")


def print_county_median_prices(df, dataset_name, top_n=TOP_N_COUNTIES):
    valid = df.dropna(subset=["CountyOrParish", "ClosePrice"])
    valid = valid[valid["ClosePrice"] > 0]

    if valid.empty:
        print(f"\n{dataset_name}: no rows with both CountyOrParish and a positive ClosePrice.")
        return None

    county_medians = (
        valid.groupby("CountyOrParish")["ClosePrice"]
        .median()
        .sort_values(ascending=False)
        .head(top_n)
    )

    print(f"\n{dataset_name}: top {top_n} counties by median ClosePrice:")
    print(county_medians)

    return county_medians


def run_eda(file_pattern, dataset_name, output_prefix):
    print(f"\n{'=' * 70}\nRunning EDA for {dataset_name}\n{'=' * 70}")

    print(f"Loading combined (unfiltered) {dataset_name} dataset - Step 1: Dataset Understanding...")
    df = load_and_concat_files(file_pattern=file_pattern, dataset_name=dataset_name)

    print_shape(df)
    print_dtypes(df)

    report = missing_value_report(df)
    print_missing_overview(report)

    type_share = property_type_share(df)
    print(f"\n{dataset_name}: PropertyType share (Residential vs. other):")
    print(type_share)
    type_share_path = OUTPUT_DIR / f"property_type_share_{output_prefix}.csv"
    type_share.to_csv(type_share_path)
    print(f"Saved property type share: {type_share_path}")

    print_field_split(df)

    print(f"\nStep 2: Missing Value Analysis ({dataset_name})...")
    print_missing_value_analysis(report)

    report_path = OUTPUT_DIR / f"missing_value_report_{output_prefix}.csv"
    report.to_csv(report_path)
    print(f"\nSaved missing value report: {report_path}")

    print(f"\nStep 3: Filter to Residential and save filtered dataset ({dataset_name})...")
    residential = filter_residential(df, dataset_name=dataset_name)

    filtered_path = OUTPUT_DIR / f"{output_prefix}_residential_filtered.csv"
    residential.to_csv(filtered_path, index=False)
    print(f"\nSaved filtered Residential dataset: {filtered_path}")

    print(f"\nStep 4: Numeric Distribution Review ({dataset_name})...")
    distribution_summary = numeric_distribution_summary(residential, NUMERIC_FIELDS)
    print(distribution_summary)

    distribution_path = OUTPUT_DIR / f"numeric_distribution_summary_{output_prefix}.csv"
    distribution_summary.to_csv(distribution_path)
    print(f"\nSaved numeric distribution summary: {distribution_path}")

    outliers = outlier_summary(residential, NUMERIC_FIELDS)
    print(f"\nExtreme outlier summary (IQR rule, {dataset_name}):")
    print(outliers)

    outliers_path = OUTPUT_DIR / f"outlier_summary_{output_prefix}.csv"
    outliers.to_csv(outliers_path)
    print(f"Saved outlier summary: {outliers_path}")

    plots_dir = PLOTS_DIR / output_prefix
    save_distribution_plots(residential, NUMERIC_FIELDS, plots_dir, dataset_name)
    print(f"Saved distribution plots to: {plots_dir}")

    print(f"\nStep 5: Suggested Intern Questions ({dataset_name})...")
    print_close_vs_list(residential, dataset_name)
    print_date_consistency_spotcheck(residential, dataset_name)
    print_county_median_prices(residential, dataset_name)

    return residential


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_eda(SOLD_PATTERN, "Sold", "sold")
    run_eda(LISTING_PATTERN, "Listing", "listing")


if __name__ == "__main__":
    main()
