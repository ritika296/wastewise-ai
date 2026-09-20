"""
WasteWise AI — synthetic restaurant sales dataset generator (Phase 2).

Generates 18 months of daily, item-level sales data across 3 restaurant
locations and 30 menu items, with a genuine demand-generating process
(base popularity, day-of-week effects, seasonality, holidays, promotions,
weather, special events, and a slow trend) plus realistic operational
noise: managers don't prepare exactly what will sell, so some days waste
food and some days run short.

IMPORTANT MODELING NOTE (documented in docs/DATASET.md):
`units_sold` is CENSORED by `inventory_available` — on a stockout day,
units_sold reflects what was available, not true demand. `true_demand`
is included as a latent validation column and must NOT be used as a
model feature (it would not exist in a real deployment) — it exists so
Phase 4 can honestly report how much harder censored-demand forecasting
is than forecasting an uncensored series, rather than quietly training
on ground truth.

Run: python generate_dataset.py   -> writes data/raw/sales_data.csv
Seed: 42 (reproducible)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

RNG = np.random.default_rng(42)
ROOT = Path(__file__).resolve().parent.parent.parent  # wastewise-ai/
OUT = ROOT / "data" / "raw" / "sales_data.csv"

START_DATE = date(2025, 1, 1)
N_DAYS = 548  # ~18 months
DATES = [START_DATE + timedelta(days=i) for i in range(N_DAYS)]

RESTAURANTS = [
    {"restaurant_id": "R01", "restaurant_location": "Bengaluru - Koramangala", "demand_multiplier": 1.15},
    {"restaurant_id": "R02", "restaurant_location": "Pune - Kalyani Nagar", "demand_multiplier": 0.90},
    {"restaurant_id": "R03", "restaurant_location": "Hyderabad - Gachibowli", "demand_multiplier": 1.00},
]

# 30 items across realistic categories, each with its own popularity,
# price, weekend affinity, and weather sensitivity.
MENU_ITEMS = [
    # item_id, item_name, category, base_daily_demand, price, weekend_lift, temp_sensitivity, category_shelf_life_days
    ("I01", "Chicken Biryani", "Biryani & Rice", 85, 249, 0.35, 0.0, 1),
    ("I02", "Veg Biryani", "Biryani & Rice", 45, 199, 0.30, 0.0, 1),
    ("I03", "Mutton Biryani", "Biryani & Rice", 30, 329, 0.40, 0.0, 1),
    ("I04", "Jeera Rice", "Biryani & Rice", 25, 149, 0.10, 0.0, 1),
    ("I05", "Butter Chicken", "Main Course", 55, 269, 0.30, -0.02, 1),
    ("I06", "Paneer Butter Masala", "Main Course", 50, 229, 0.25, -0.02, 1),
    ("I07", "Dal Makhani", "Main Course", 40, 179, 0.15, 0.0, 1),
    ("I08", "Chicken Tikka Masala", "Main Course", 45, 259, 0.28, -0.02, 1),
    ("I09", "Palak Paneer", "Main Course", 30, 209, 0.15, 0.0, 1),
    ("I10", "Kadai Chicken", "Main Course", 35, 249, 0.25, 0.0, 1),
    ("I11", "Butter Naan", "Breads", 90, 45, 0.20, 0.0, 1),
    ("I12", "Garlic Naan", "Breads", 60, 55, 0.20, 0.0, 1),
    ("I13", "Tandoori Roti", "Breads", 70, 30, 0.10, 0.0, 1),
    ("I14", "Laccha Paratha", "Breads", 40, 50, 0.15, 0.0, 1),
    ("I15", "Chicken 65", "Starters", 50, 219, 0.35, 0.01, 1),
    ("I16", "Paneer Tikka", "Starters", 35, 199, 0.25, 0.0, 1),
    ("I17", "Veg Spring Rolls", "Starters", 25, 159, 0.15, 0.0, 1),
    ("I18", "Chicken Lollipop", "Starters", 30, 229, 0.30, 0.01, 1),
    ("I19", "Gulab Jamun", "Desserts", 40, 89, 0.20, -0.01, 3),
    ("I20", "Rasmalai", "Desserts", 25, 99, 0.20, -0.01, 3),
    ("I21", "Gajar Halwa", "Desserts", 15, 109, 0.15, -0.03, 3),
    ("I22", "Ice Cream Sundae", "Desserts", 30, 129, 0.15, 0.05, 2),
    ("I23", "Masala Chai", "Beverages", 100, 40, -0.05, -0.03, 1),
    ("I24", "Cold Coffee", "Beverages", 60, 99, 0.15, 0.04, 2),
    ("I25", "Fresh Lime Soda", "Beverages", 55, 69, 0.10, 0.05, 2),
    ("I26", "Mango Lassi", "Beverages", 45, 89, 0.15, 0.04, 2),
    ("I27", "Mineral Water", "Beverages", 80, 30, 0.05, 0.02, 7),
    ("I28", "Veg Fried Rice", "Biryani & Rice", 20, 179, 0.10, 0.0, 1),
    ("I29", "Chicken Fried Rice", "Biryani & Rice", 25, 219, 0.15, 0.0, 1),
    ("I30", "Soup of the Day", "Starters", 20, 129, -0.10, -0.06, 1),
]

# Simplified Indian holiday calendar for 2025 (subset, illustrative for PoC)
HOLIDAYS = {
    date(2025, 1, 1), date(2025, 1, 14), date(2025, 1, 26), date(2025, 3, 14),
    date(2025, 3, 31), date(2025, 4, 18), date(2025, 5, 1), date(2025, 8, 15),
    date(2025, 8, 27), date(2025, 10, 2), date(2025, 10, 20), date(2025, 10, 21),
    date(2025, 11, 5), date(2025, 12, 25), date(2026, 1, 1), date(2026, 1, 14),
    date(2026, 1, 26),
}

WEATHER_STATES = ["Sunny", "Cloudy", "Rainy", "Hot", "Pleasant"]


def seasonal_temperature(d: date) -> float:
    """Rough North/South-India composite seasonal curve, 15-40C, plus noise."""
    day_of_year = d.timetuple().tm_yday
    base = 27 + 8 * np.sin(2 * np.pi * (day_of_year - 60) / 365)
    return float(np.clip(base + RNG.normal(0, 2.5), 14, 42))


def weather_for(temp: float) -> str:
    if temp > 34:
        return "Hot"
    if temp < 20:
        return RNG.choice(["Cloudy", "Rainy"], p=[0.6, 0.4])
    return RNG.choice(["Sunny", "Cloudy", "Pleasant"], p=[0.45, 0.25, 0.30])


def generate():
    rows = []
    # Pre-generate promotion calendar: each item gets ~6-10 promo days over 18 months, in short bursts
    promo_days = {}
    for item in MENU_ITEMS:
        item_id = item[0]
        n_bursts = RNG.integers(3, 6)
        burst_starts = RNG.choice(N_DAYS - 5, size=n_bursts, replace=False)
        days = set()
        for s in burst_starts:
            days.update(range(s, s + RNG.integers(2, 5)))
        promo_days[item_id] = days

    # Special events: 6 random city-wide demand-spike days over the period (local festival, cricket match, etc.)
    special_event_days = set(RNG.choice(N_DAYS, size=6, replace=False).tolist())

    for day_idx, d in enumerate(DATES):
        dow = d.weekday()  # 0=Mon
        is_weekend = dow >= 5
        is_holiday = d in HOLIDAYS
        temp = seasonal_temperature(d)
        weather = weather_for(temp)
        is_special_event = day_idx in special_event_days
        # slow trend: ~8% organic growth across the 18-month window
        trend = 1.0 + 0.08 * (day_idx / N_DAYS)

        for rest in RESTAURANTS:
            for item in MENU_ITEMS:
                item_id, name, category, base, price, weekend_lift, temp_sens, shelf_life = item

                mult = rest["demand_multiplier"] * trend
                if is_weekend:
                    mult *= (1 + weekend_lift)
                if is_holiday:
                    mult *= 1.35
                if is_special_event:
                    mult *= 1.45
                mult *= (1 + temp_sens * (temp - 27))

                promotion_flag = int(day_idx in promo_days[item_id])
                discount = 0.0
                if promotion_flag:
                    discount = float(RNG.choice([10, 15, 20]))
                    mult *= (1 + discount / 100 * 0.6)  # discount lifts demand, sub-linearly

                selling_price = round(price * (1 - discount / 100), 2)

                true_mean = max(base * mult, 1)
                # Negative-binomial-like overdispersed count via Poisson-Gamma mixture
                shape = 12.0
                true_demand = RNG.gamma(shape, true_mean / shape)
                true_demand = int(round(RNG.poisson(true_demand)))

                # --- Manager's imperfect preparation decision -----------
                # Anchored on a noisy view of recent-average demand (simulating
                # "manager experience"), not the true mean — this is what
                # creates realistic over/under-preparation. Calibrated so the
                # dataset lands at ~17% stockout days and ~27% waste-as-share-
                # of-inventory — a real, worth-solving problem, not chaos and
                # not negligible. (See docs/DATASET.md for the calibration.)
                naive_anchor = true_mean * RNG.normal(1.0, 0.10)
                safety_habit = RNG.normal(1.32, 0.055)  # managers over-prepare on average, risk-averse
                inventory_available = max(int(round(naive_anchor * safety_habit)), 0)

                units_sold = min(true_demand, inventory_available)
                waste_units = max(inventory_available - true_demand, 0)
                stockout_flag = int(true_demand > inventory_available)

                rows.append({
                    "date": d.isoformat(),
                    "restaurant_id": rest["restaurant_id"],
                    "restaurant_location": rest["restaurant_location"],
                    "item_id": item_id,
                    "item_name": name,
                    "category": category,
                    "units_sold": units_sold,
                    "selling_price": selling_price,
                    "discount": discount,
                    "promotion_flag": promotion_flag,
                    "day_of_week": dow,
                    "weekend_flag": int(is_weekend),
                    "holiday_flag": int(is_holiday),
                    "weather_condition": weather,
                    "temperature": round(temp, 1),
                    "special_event": int(is_special_event),
                    "inventory_available": inventory_available,
                    "waste_units": waste_units,
                    "stockout_flag": stockout_flag,
                    "true_demand": true_demand,  # LATENT — validation only, never a model feature
                })

    df = pd.DataFrame(rows)

    # --- Inject realistic data-quality issues (to be caught in Phase 3) ---
    # 1. A small number of missing temperature readings (sensor gaps)
    miss_idx = RNG.choice(df.index, size=int(0.003 * len(df)), replace=False)
    df.loc[miss_idx, "temperature"] = np.nan
    # 2. A handful of duplicate rows (export artefact)
    dupes = df.sample(frac=0.001, random_state=1)
    df = pd.concat([df, dupes], ignore_index=True)
    # 3. A few negative-price data-entry errors
    err_idx = RNG.choice(df.index, size=6, replace=False)
    df.loc[err_idx, "selling_price"] = -df.loc[err_idx, "selling_price"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"Wrote {len(df):,} rows to {OUT}")
    print(f"Date range: {DATES[0]} to {DATES[-1]} ({N_DAYS} days)")
    print(f"Restaurants: {len(RESTAURANTS)} | Menu items: {len(MENU_ITEMS)}")
    print(f"Avg units_sold: {df['units_sold'].mean():.1f} | Avg true_demand: {df['true_demand'].mean():.1f}")
    print(f"Stockout rate: {df['stockout_flag'].mean():.2%} | Avg waste_units: {df['waste_units'].mean():.2f}")
    print(f"Promotion day-rate: {df['promotion_flag'].mean():.2%} | Holiday day-rate: {df['holiday_flag'].mean():.2%}")
    return df


if __name__ == "__main__":
    generate()
