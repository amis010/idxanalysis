import sys
from pathlib import Path

import pandas as pd

# Week 7 - Outlier Detection and Data Quality.
# Takes the Week 6 feature-engineered Sold and Listing datasets and applies
# IQR-based outlier detection to ClosePrice, LivingArea, and DaysOnMarket.
# Follows the Week 4-5 convention of flagging rather than deleting: outlier
# flag columns are added to a full dataset, and a separate clean filtered
# dataset (business-invalid and outlier rows removed) is saved alongside it.

sys.path.insert(0, str(Path(__file__).resolve().parent))
from week4_5_data_cleaning import _conditional_flag

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "outputs" / "week6_features"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week7_outliers"

# Each field under IQR test is paired with the Week 4-5 business-rule flag
# that already marks it invalid (e.g. ClosePrice <= 0). IQR bounds are
# computed only over rows where that flag is False - the tiered approach
# the assignment calls for: business rules first, then statistics.
IQR_FIELDS = [
    ("ClosePrice", "close_price_invalid_flag", "close_price_outlier_flag"),
    ("LivingArea", "living_area_invalid_flag", "living_area_outlier_flag"),
    ("DaysOnMarket", "days_on_market_invalid_flag", "days_on_market_outlier_flag"),
]


def compute_iqr_bounds(df, field, invalid_flag_col):
    valid_basis = df[field].notna() & (df[invalid_flag_col] == False)  # noqa: E712
    series = df.loc[valid_basis, field]

    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    return valid_basis, {
        "q1": q1, "q3": q3, "iqr": iqr,
        "lower_bound": lower, "upper_bound": upper,
        "valid_basis_count": int(valid_basis.sum()),
    }


def flag_outliers(df, dataset_name):
    """Adds one outlier flag column per IQR_FIELDS entry: True/False on rows
    where the field has a valid, business-rule-valid value; pd.NA where the
    field is missing or already flagged invalid upstream (an already-invalid
    row isn't double-judged as a statistical outlier too)."""
    df = df.copy()
    bounds_rows = {}

    for field, invalid_flag_col, outlier_flag_col in IQR_FIELDS:
        valid_basis, bounds = compute_iqr_bounds(df, field, invalid_flag_col)
        bounds_rows[field] = bounds

        is_outlier = (df.loc[valid_basis, field] < bounds["lower_bound"]) | \
                     (df.loc[valid_basis, field] > bounds["upper_bound"])
        df[outlier_flag_col] = _conditional_flag(df.index, valid_basis, is_outlier)

        print(f"{dataset_name}: {field} - Q1={bounds['q1']:.2f}, Q3={bounds['q3']:.2f}, "
              f"IQR={bounds['iqr']:.2f}, bounds=[{bounds['lower_bound']:.2f}, {bounds['upper_bound']:.2f}], "
              f"{int(is_outlier.sum()):,} outlier(s) of {bounds['valid_basis_count']:,} valid row(s)")

    bounds_report = pd.DataFrame.from_dict(bounds_rows, orient="index")

    return df, bounds_report


def summarize_flags(df, flag_columns, dataset_name):
    rows = {}

    for col in flag_columns:
        rows[col] = {
            "true_count": (df[col] == True).sum(),   # noqa: E712
            "false_count": (df[col] == False).sum(),  # noqa: E712
            "not_applicable_count": df[col].isna().sum(),
        }

    summary = pd.DataFrame.from_dict(rows, orient="index")
    print(f"\n{dataset_name}: outlier flag summary (true / false / not-applicable counts):")
    print(summary)

    return summary


def build_clean_filtered(df, dataset_name):
    """A row is dropped if, for any IQR field, it's business-invalid (True)
    or a statistical outlier (True). NA (field not applicable) never causes
    a drop."""
    drop_mask = pd.Series(False, index=df.index)

    for _, invalid_flag_col, outlier_flag_col in IQR_FIELDS:
        # fillna(False): NA (field not applicable / already excluded from
        # the outlier basis) must never cause a drop, only True should. A
        # bare `|=` against a nullable-boolean column would otherwise let
        # pd.NA propagate through the OR and get treated as a drop by
        # .loc[~drop_mask].
        drop_mask |= (df[invalid_flag_col] == True).fillna(False)   # noqa: E712
        drop_mask |= (df[outlier_flag_col] == True).fillna(False)    # noqa: E712

    clean = df.loc[~drop_mask].copy()

    print(f"\n{dataset_name}: {len(df):,} -> {len(clean):,} rows after removing "
          f"business-invalid and statistical-outlier rows ({drop_mask.sum():,} dropped).")

    return clean


def median_snapshot(df):
    return {field: df[field].median() for field, _, _ in IQR_FIELDS}


def write_comparison_section(dataset_name, before_df, after_df):
    before_medians = median_snapshot(before_df)
    after_medians = median_snapshot(after_df)

    lines = [f"## {dataset_name}", ""]
    lines.append(f"- Row count: {len(before_df):,} before filtering -> "
                  f"{len(after_df):,} after filtering "
                  f"({len(before_df) - len(after_df):,} rows removed, "
                  f"{(len(before_df) - len(after_df)) / len(before_df) * 100:.1f}%).")

    for field, _, _ in IQR_FIELDS:
        before_val, after_val = before_medians[field], after_medians[field]
        lines.append(f"- Median {field}: {before_val:,.2f} before -> {after_val:,.2f} after "
                      f"({after_val - before_val:+,.2f}).")

    lines.append("")

    return "\n".join(lines)


def run_outlier_detection(input_path, dataset_name, output_prefix):
    print(f"\n{'=' * 70}\nOutlier detection - {dataset_name}\n{'=' * 70}")

    df = pd.read_csv(input_path, low_memory=False)
    print(f"{dataset_name}: loaded {len(df):,} rows, {len(df.columns)} columns from {input_path.name}")

    print(f"\nStep 1-2: Computing IQR bounds and flagging outliers "
          f"({', '.join(f for f, _, _ in IQR_FIELDS)})...")
    flagged_df, bounds_report = flag_outliers(df, dataset_name)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nStep 3: Saving full flagged dataset...")
    flagged_path = OUTPUT_DIR / f"{output_prefix}_residential_outlier_flagged.csv"
    flagged_df.to_csv(flagged_path, index=False)
    print(f"Saved: {flagged_path}")

    print(f"\nStep 4: Building clean filtered dataset...")
    clean_df = build_clean_filtered(flagged_df, dataset_name)

    clean_path = OUTPUT_DIR / f"{output_prefix}_residential_filtered_clean.csv"
    clean_df.to_csv(clean_path, index=False)
    print(f"Saved: {clean_path}")

    print(f"\nStep 5: Saving bounds and flag summary reports...")
    bounds_path = OUTPUT_DIR / f"outlier_bounds_{output_prefix}.csv"
    bounds_report.to_csv(bounds_path)
    print(f"Saved: {bounds_path}")

    outlier_flag_columns = [outlier_flag_col for _, _, outlier_flag_col in IQR_FIELDS]
    flag_summary = summarize_flags(flagged_df, outlier_flag_columns, dataset_name)

    flag_summary_path = OUTPUT_DIR / f"outlier_flag_summary_{output_prefix}.csv"
    flag_summary.to_csv(flag_summary_path)
    print(f"Saved: {flag_summary_path}")

    return flagged_df, clean_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sold_flagged, sold_clean = run_outlier_detection(
        FEATURES_DIR / "sold_residential_features.csv", "Sold", "sold",
    )
    listing_flagged, listing_clean = run_outlier_detection(
        FEATURES_DIR / "listing_residential_features.csv", "Listing", "listing",
    )

    print(f"\n{'=' * 70}\nWriting before/after comparison summary\n{'=' * 70}")

    report_lines = [
        "# Week 7 - Outlier Detection: Before/After Comparison", "",
        "IQR-based outlier flags removed, alongside rows already flagged invalid "
        "by Week 4-5's business rules (e.g. ClosePrice <= 0).", "",
        write_comparison_section("Sold", sold_flagged, sold_clean),
        write_comparison_section("Listing", listing_flagged, listing_clean),
    ]

    summary_path = OUTPUT_DIR / "week7_comparison_summary.md"
    summary_path.write_text("\n".join(report_lines))
    print(f"Saved: {summary_path}")

    print("\nWeek 7 outlier detection complete.")


if __name__ == "__main__":
    main()
