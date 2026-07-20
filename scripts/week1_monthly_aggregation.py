import re
from pathlib import Path
import pandas as pd

# Week 1 – Monthly Dataset Aggregation
# Objective:
# Combine all monthly MLS sold and listing files from January 2024
# through the most recently completed calendar month.
# Then filter both datasets to PropertyType == "Residential"
# and save them as new CSV files.

# This gets the main project folder.
# The script is inside scripts/, so parents[1] goes back to idxanalysis/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw monthly CSV files are stored locally here.
# This folder should NOT be uploaded to GitHub.
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Final combined CSV outputs will be saved here.
# This folder should also NOT be uploaded to GitHub.
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Create outputs folder if it does not already exist.
OUTPUT_DIR.mkdir(exist_ok=True)

# File patterns for the two required dataset types.
SOLD_PATTERN = "CRMLSSold*.csv"
LISTING_PATTERN = "CRMLSListing*.csv"


FILENAME_PATTERN = re.compile(r"^(?P<prefix>.+?)(?P<month>\d{6})(?P<filled>_filled)?\.csv$")


def select_one_file_per_month(files):
    """Some months have both a plain export and a "_filled" export (same
    listings, with latfilled/lonfilled coordinate backfill added). Loading
    both double-counts that month's rows, so this keeps only one file per
    (prefix, month) - preferring the _filled version when both exist, since
    it is a superset of the plain export's columns."""
    grouped = {}

    for file in files:
        match = FILENAME_PATTERN.match(file.name)
        if not match:
            raise ValueError(f"Unexpected file name pattern: {file.name}")

        key = (match.group("prefix"), match.group("month"))
        grouped.setdefault(key, []).append((file, bool(match.group("filled"))))

    selected, skipped = [], []

    for candidates in grouped.values():
        filled_candidates = [f for f, is_filled in candidates if is_filled]
        chosen = filled_candidates[0] if filled_candidates else candidates[0][0]

        selected.append(chosen)
        skipped.extend(f for f, _ in candidates if f != chosen)

    return sorted(selected), sorted(skipped)


def load_and_concat_files(file_pattern, dataset_name):
    all_files = sorted(RAW_DATA_DIR.glob(file_pattern))

    if not all_files:
        raise FileNotFoundError(
            f"No {dataset_name} files found in {RAW_DATA_DIR}."
        )

    files, skipped_files = select_one_file_per_month(all_files)

    if skipped_files:
        print(f"\n{dataset_name}: skipping {len(skipped_files)} duplicate monthly "
              f"file(s) in favor of their _filled counterpart:")
        for file in skipped_files:
            print(f" - {file.name}")

    dataframes = []
    total_rows_before_concat = 0

    print(f"\nLoading {dataset_name} files:")

    for file in files:
        df = pd.read_csv(file, low_memory=False)

        row_count = len(df)
        total_rows_before_concat += row_count

        print(f" - {file.name}: {row_count:,} rows")

        # Adds a column showing which original monthly file each row came from.
        df["SourceFile"] = file.name

        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    # Row count confirmation before and after concatenation.
    print(f"\n{dataset_name} rows before concatenation: {total_rows_before_concat:,}")
    print(f"{dataset_name} rows after concatenation: {len(combined_df):,}")

    if "ListingKey" in combined_df.columns:
        duplicate_count = combined_df["ListingKey"].duplicated().sum()
        if duplicate_count > 0:
            print(f"{dataset_name}: dropping {duplicate_count:,} duplicate row(s) "
                  f"with a repeated ListingKey (keeping first occurrence).")
            combined_df = combined_df.drop_duplicates(subset="ListingKey", keep="first")
            print(f"{dataset_name} rows after de-duplication: {len(combined_df):,}")

    return combined_df


def filter_residential(df, dataset_name):
    if "PropertyType" not in df.columns:
        raise KeyError(f"PropertyType column not found in {dataset_name} dataset.")

    rows_before_filter = len(df)

    residential_df = df[df["PropertyType"] == "Residential"].copy()

    rows_after_filter = len(residential_df)

    # Row count confirmation before and after Residential filter.
    print(f"\n{dataset_name} rows before Residential filter: {rows_before_filter:,}")
    print(f"{dataset_name} rows after Residential filter: {rows_after_filter:,}")

    return residential_df


def main():
    print("Starting Week 1 monthly dataset aggregation...")

    # 1. Load and combine all sold transaction monthly files.
    sold_combined = load_and_concat_files(
        file_pattern=SOLD_PATTERN,
        dataset_name="Sold"
    )

    # 2. Load and combine all listing monthly files.
    listing_combined = load_and_concat_files(
        file_pattern=LISTING_PATTERN,
        dataset_name="Listing"
    )

    # 3. Filter both combined datasets to Residential properties only.
    sold_residential = filter_residential(
        df=sold_combined,
        dataset_name="Sold"
    )

    listing_residential = filter_residential(
        df=listing_combined,
        dataset_name="Listing"
    )

    # 4. Save the two required output CSV files.
    sold_output_path = OUTPUT_DIR / "combined_sold_residential.csv"
    listing_output_path = OUTPUT_DIR / "combined_listing_residential.csv"

    sold_residential.to_csv(sold_output_path, index=False)
    listing_residential.to_csv(listing_output_path, index=False)

    print("\nSaved final output files:")
    print(f" - {sold_output_path}")
    print(f" - {listing_output_path}")

    print("\nWeek 1 aggregation complete.")


if __name__ == "__main__":
    main()