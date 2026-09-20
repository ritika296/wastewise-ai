"""
Data validation (Phase 3 / Module L — Data Quality).

Runs a battery of checks against the raw sales data and produces a
structured report. This is deliberately run BEFORE cleaning, so the
report reflects the data as it actually arrived — the injected issues
from Phase 2 (missing temperature, duplicate rows, negative prices)
should all be caught here, not silently fixed first.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import pandas as pd
import numpy as np

EXPECTED_COLUMNS = {
    "date", "restaurant_id", "restaurant_location", "item_id", "item_name",
    "category", "units_sold", "selling_price", "discount", "promotion_flag",
    "day_of_week", "weekend_flag", "holiday_flag", "weather_condition",
    "temperature", "special_event", "inventory_available", "waste_units",
    "stockout_flag",
}
EXPECTED_CATEGORIES = {"Biryani & Rice", "Main Course", "Breads", "Starters",
                        "Beverages", "Desserts"}
EXPECTED_WEATHER = {"Sunny", "Cloudy", "Rainy", "Hot", "Pleasant"}
FRESHNESS_WARN_DAYS = 3  # a real deployment would flag data older than this


@dataclass
class ValidationReport:
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    checks: dict = field(default_factory=dict)
    overall_score: float = 0.0
    status: str = "unknown"
    generated_at: str = ""

    def to_dict(self):
        return {
            "total_records": self.total_records, "valid_records": self.valid_records,
            "invalid_records": self.invalid_records, "checks": self.checks,
            "overall_score": self.overall_score, "status": self.status,
            "generated_at": self.generated_at,
        }


def validate(df: pd.DataFrame, as_of: datetime = None) -> ValidationReport:
    n = len(df)
    checks = {}
    bad_row_mask = pd.Series(False, index=df.index)

    # 1. Schema check
    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    extra_cols = set(df.columns) - EXPECTED_COLUMNS - {"true_demand"}  # true_demand allowed (validation-only)
    checks["schema"] = {
        "missing_columns": sorted(missing_cols), "unexpected_columns": sorted(extra_cols),
        "status": "pass" if not missing_cols else "fail",
    }

    # 2. Missing values (per column, on expected columns present)
    present_expected = [c for c in EXPECTED_COLUMNS if c in df.columns]
    missing_by_col = df[present_expected].isna().sum()
    missing_total = int(missing_by_col.sum())
    checks["missing_values"] = {
        "total_missing_cells": missing_total,
        "missing_pct": round(100 * missing_total / (n * len(present_expected)), 3) if n else 0,
        "by_column": {k: int(v) for k, v in missing_by_col[missing_by_col > 0].items()},
    }
    bad_row_mask |= df[present_expected].isna().any(axis=1)

    # 3. Duplicates (exact full-row duplicates)
    dup_mask = df.duplicated(keep="first")
    checks["duplicates"] = {"count": int(dup_mask.sum()), "pct": round(100 * dup_mask.mean(), 3)}
    bad_row_mask |= dup_mask

    # 4. Invalid dates
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    invalid_dates = parsed_dates.isna()
    checks["invalid_dates"] = {"count": int(invalid_dates.sum())}
    bad_row_mask |= invalid_dates

    # 5. Negative / implausible values
    neg_price = df["selling_price"] < 0
    neg_sales = df["units_sold"] < 0
    neg_inventory = df["inventory_available"] < 0
    checks["negative_values"] = {
        "negative_prices": int(neg_price.sum()), "negative_sales": int(neg_sales.sum()),
        "negative_inventory": int(neg_inventory.sum()),
    }
    bad_row_mask |= neg_price | neg_sales | neg_inventory

    # 6. Outliers (units_sold beyond 5 std devs of its item's own distribution)
    z = df.groupby("item_id")["units_sold"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else 1)
    )
    outliers = z.abs() > 5
    checks["outliers"] = {"count": int(outliers.sum()), "method": "z-score > 5, per item_id"}

    # 7. Missing inventory (inventory_available present but zero/NaN alongside nonzero sales — inconsistent)
    inconsistent_inventory = (df["inventory_available"].fillna(0) < df["units_sold"].fillna(0))
    checks["inventory_consistency"] = {
        "units_sold_exceeds_inventory": int(inconsistent_inventory.sum()),
    }
    bad_row_mask |= inconsistent_inventory

    # 8. Unexpected categories / weather values
    unexpected_cat = ~df["category"].isin(EXPECTED_CATEGORIES)
    unexpected_weather = ~df["weather_condition"].isin(EXPECTED_WEATHER)
    checks["unexpected_categorical_values"] = {
        "unexpected_categories": int(unexpected_cat.sum()),
        "unexpected_weather_conditions": int(unexpected_weather.sum()),
    }

    # 9. Freshness (most recent date vs. "as of" reference time)
    as_of = as_of or datetime.now(timezone.utc)
    max_date = parsed_dates.max()
    freshness_days = (as_of.replace(tzinfo=None) - max_date).days if pd.notna(max_date) else None
    checks["freshness"] = {
        "latest_record_date": str(max_date.date()) if pd.notna(max_date) else None,
        "days_stale": freshness_days,
        "status": "stale" if (freshness_days is not None and freshness_days > FRESHNESS_WARN_DAYS) else "fresh",
    }

    valid = n - int(bad_row_mask.sum())
    score = round(100 * valid / n, 2) if n else 0.0
    status = "pass" if score >= 95 and not missing_cols else "warning" if score >= 85 else "fail"

    return ValidationReport(
        total_records=n, valid_records=valid, invalid_records=n - valid,
        checks=checks, overall_score=score, status=status,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
