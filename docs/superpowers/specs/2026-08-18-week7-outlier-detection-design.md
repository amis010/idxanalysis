# Week 7 — Outlier Detection and Data Quality — Design

## Context

The pipeline runs Week 1 → 6 in sequence, each script consuming the prior week's
output (`scripts/week1_monthly_aggregation.py` through `scripts/week6_feature_engineering.py`).
Week 4-5 (`scripts/week4_5_data_cleaning.py`) already established the project's
core convention for handling bad data: **flag, don't delete**. It adds boolean
business-rule flag columns (e.g. `close_price_invalid_flag` for `ClosePrice <= 0`)
using a shared `_conditional_flag` helper that returns `True`/`False` where the
underlying field is present, and `pd.NA` where it isn't — so nothing is ever
silently dropped and downstream steps can decide what to do with flagged rows.

Week 7 extends this with statistical outlier detection (IQR method) on top of
those existing business-rule flags, and — for the first time — produces an
actual filtered/reduced dataset, alongside the fully-flagged one.

## Task requirements (from the assignment)

- Apply IQR filtering to `ClosePrice`, `LivingArea`, `DaysOnMarket`.
- Add outlier flag columns rather than deleting records outright.
- Save both a full flagged dataset and a clean filtered dataset.
- Use a tiered approach: business rules first (e.g. `ClosePrice <= 0` is
  always invalid), then IQR on the remaining valid data.
- Include a written comparison of dataset size and median values before and
  after filtering.

## Decisions made during brainstorming

1. **Scope: both Sold and Listing datasets.** Even though `ClosePrice` is only
   populated for `Closed`-status rows in the Listing dataset (~126k of
   ~580k), the existing `_conditional_flag` convention already produces `NA`
   for rows where a field doesn't apply — the new outlier flags follow the
   same rule, so no special-casing is needed for Listing vs. Sold.
2. **Input: Week 6 feature-engineered output**
   (`outputs/features/sold_residential_features.csv` /
   `listing_residential_features.csv`), continuing the existing pipeline
   chain rather than reaching back to the Week 4-5 cleaned files.
3. **Fields: exactly `ClosePrice`, `LivingArea`, `DaysOnMarket`** — the
   literal deliverable list, not the broader set mentioned in the task's
   rationale paragraph (`PricePerSqFt`, `CloseToOriginalListRatio`).
4. **IQR baseline excludes pre-flagged invalid rows.** Week 6's output does
   *not* exclude Week 4-5's business-invalid rows from anything it computes —
   confirmed by inspecting `sold_residential_features.csv`, where
   `close_price_invalid_flag` is still present and `PricePerSqFt` etc. were
   computed over all rows. So Week 7 must do the exclusion itself: Q1/Q3 for
   each field are computed only over rows where that field's Week 4-5
   business-rule flag is `False` (not `True`, not `NA`).

## Data quality note (observed, not in scope to fix)

`listing_residential_features.csv` has duplicate columns (`DaysOnMarket.1`,
`LivingArea.1`, `Latitude.1`, etc.) — pandas' auto-rename of duplicate header
names present in the raw CRMLS listing exports. Verified they are
byte-identical to their primary counterparts. Week 7 uses only the primary
(non-`.1`) column names and otherwise ignores this.

## Design

### New script: `scripts/week7_outlier_detection.py`

Follows the existing per-week script shape: a `PROJECT_ROOT`-relative
`INPUT_DIR`/`OUTPUT_DIR`, a `run_outlier_detection(input_path, dataset_name,
output_prefix)` function called once per dataset from `main()`, printed
step-by-step narration matching the style of `week4_5_data_cleaning.py`.

**Fields under test:** `("ClosePrice", "close_price_invalid_flag")`,
`("LivingArea", "living_area_invalid_flag")`,
`("DaysOnMarket", "days_on_market_invalid_flag")` — each numeric field is
paired with the Week 4-5 business-rule flag column that gates it.

**Step 1 — Compute IQR bounds per field, on the valid basis only.**
For each field:
- `valid_basis = df[field].notna() & (df[invalid_flag_col] == False)`
- `Q1, Q3 = df.loc[valid_basis, field].quantile([0.25, 0.75])`
- `IQR = Q3 - Q1`; `lower = Q1 - 1.5*IQR`; `upper = Q3 + 1.5*IQR`
- Record these five numbers per field for the bounds report.

**Step 2 — Add outlier flag columns**, one per field
(`close_price_outlier_flag`, `living_area_outlier_flag`,
`days_on_market_outlier_flag`), using the same `_conditional_flag`-style
pattern as Week 4-5 (reimplemented locally or imported from
`week4_5_data_cleaning`):
- `True` where `valid_basis` and value is outside `[lower, upper]`
- `False` where `valid_basis` and value is inside `[lower, upper]`
- `pd.NA` where `valid_basis` is `False` (field missing or already
  business-invalid) — an already-invalid row isn't double-judged as a
  statistical outlier too.

**Step 3 — Save the full flagged dataset** (all rows, original columns +
3 new outlier flag columns) to
`outputs/outliers/{prefix}_residential_outlier_flagged.csv`.

**Step 4 — Build the clean filtered dataset.** A row is dropped if, for any
of the 3 fields, its business-invalid flag is `True` OR its outlier flag is
`True`. (`NA` on either — field not applicable — does not cause a drop.)
Save to `outputs/outliers/{prefix}_residential_filtered_clean.csv`.

**Step 5 — Save supporting reports:**
- `outlier_bounds_{prefix}.csv`: one row per field with
  `q1, q3, iqr, lower_bound, upper_bound, valid_basis_count`.
- `outlier_flag_summary_{prefix}.csv`: same shape as Week 4-5's
  `flag_summary_*.csv` (`true_count`/`false_count`/`not_applicable_count`
  per new flag column).

**Step 6 — Before/after comparison.** For each dataset, compute row count
and median `ClosePrice`/`LivingArea`/`DaysOnMarket` before filtering (on the
full flagged dataset) and after (on the clean filtered dataset). Print this
to console (consistent with how every prior week's narrative log doubles as
its record of what happened), and also append it as prose to a single
combined markdown file, `outputs/outliers/week7_comparison_summary.md`,
covering both Sold and Listing in one document.

### Directory

All Week 7 outputs go under a new `outputs/outliers/` directory, matching the
existing `outputs/{eda,enriched,cleaned,features}/` convention.

### Out of scope

- Fixing the duplicate `.1` columns in the Listing dataset.
- Extending IQR treatment to `PricePerSqFt` / `CloseToOriginalListRatio`
  (rationale-only mention, not in the literal deliverable).
- Any change to Weeks 1-6.
