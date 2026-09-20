"""
Phase 6 demo (now using the shared ml/inference/pipeline.py, refactored
in Phase 7 so the copilot and what-if simulator reuse the exact same
recommendation-generation logic instead of a second copy).

Run from wastewise-ai/ root: python ml/inference/run_recommendations_demo.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ml.inference.pipeline import generate_recommendations_for_date, champion_model_and_features

DEMO_DATE = "2026-05-08"


def main():
    champion, _, _ = champion_model_and_features()
    print(f"Using champion: {champion['version_id']} ({champion['model_name']})")

    recommendations = generate_recommendations_for_date(DEMO_DATE)
    recommendations = recommendations.sort_values("expected_shortfall_units", ascending=False)

    out_path = ROOT / "data" / "processed" / f"recommendations_{DEMO_DATE}.json"
    recommendations.to_json(out_path, orient="records", indent=2)
    print(f"\nSaved {len(recommendations)} recommendations to {out_path}")

    print("\n=== Top 5 HIGH SHORTAGE RISK items ===")
    for _, r in recommendations[recommendations.shortage_risk == "HIGH"].head(5).iterrows():
        print(f"\n{r['item_name']} ({r['restaurant_id']})")
        print(f"  Forecast: {r['forecast_demand']} | Available: {r['current_inventory']} | "
              f"Recommended prep: {r['recommended_preparation']}")
        print(f"  Risk: {r['risk_category']}")
        print(f"  Reason: {r['shortage_reason']}")
        print(f"  Revenue at risk: ₹{r['estimated_revenue_at_risk_inr']}")

    print("\n=== Summary ===")
    print(recommendations["risk_category"].value_counts())


if __name__ == "__main__":
    main()
