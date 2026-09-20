"""
Data cleaning (Phase 3). Runs AFTER validate.py has reported what's wrong.
Every fix applied here is logged — nothing is silently changed. Rows that
can't be sensibly repaired are dropped, and the drop is counted, not hidden.
"""
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


@dataclass
class CleaningLog:
    actions: list = field(default_factory=list)
    rows_in: int = 0
    rows_out: int = 0

    def log(self, action: str, count: int):
        if count > 0:
            self.actions.append({"action": action, "rows_affected": count})

    def to_dict(self):
        return {"rows_in": self.rows_in, "rows_out": self.rows_out,
                "rows_dropped": self.rows_in - self.rows_out, "actions": self.actions}


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog]:
    log = CleaningLog(rows_in=len(df))
    df = df.copy()

    # 1. Drop exact duplicate rows
    dup_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    log.log("dropped_exact_duplicates", dup_count)

    # 2. Fix negative prices — these are data-entry sign errors, not real
    #    negative prices; take absolute value (documented assumption, not
    #    a silent fabrication — logged explicitly).
    neg_price_mask = df["selling_price"] < 0
    neg_count = int(neg_price_mask.sum())
    df.loc[neg_price_mask, "selling_price"] = df.loc[neg_price_mask, "selling_price"].abs()
    log.log("fixed_negative_prices_took_absolute_value", neg_count)

    # 3. Impute missing temperature — per-item-location seasonal median
    #    (grouped by restaurant + month) rather than a single global
    #    fill, since temperature is strongly seasonal.
    df["date"] = pd.to_datetime(df["date"])
    df["_month"] = df["date"].dt.month
    missing_temp = int(df["temperature"].isna().sum())
    seasonal_median = df.groupby(["restaurant_id", "_month"])["temperature"].transform("median")
    df["temperature"] = df["temperature"].fillna(seasonal_median)
    # residual fallback (should be rare): global median
    df["temperature"] = df["temperature"].fillna(df["temperature"].median())
    df = df.drop(columns=["_month"])
    log.log("imputed_missing_temperature_seasonal_median", missing_temp)

    # 4. Drop rows where units_sold exceeds inventory_available — an
    #    internal inconsistency that shouldn't be "fixed" by guessing;
    #    these are dropped and counted, not imputed.
    inconsistent = df["units_sold"] > df["inventory_available"]
    inconsistent_count = int(inconsistent.sum())
    df = df[~inconsistent].reset_index(drop=True)
    log.log("dropped_inventory_inconsistent_rows", inconsistent_count)

    # 5. Drop any remaining rows with missing values in critical fields
    #    (should be near-zero after step 3, but never assume)
    critical = ["date", "restaurant_id", "item_id", "units_sold", "selling_price"]
    still_missing = df[critical].isna().any(axis=1)
    still_missing_count = int(still_missing.sum())
    df = df[~still_missing].reset_index(drop=True)
    log.log("dropped_remaining_missing_critical_fields", still_missing_count)

    log.rows_out = len(df)
    return df, log
