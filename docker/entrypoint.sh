#!/bin/sh
# Idempotent pipeline bootstrap: only runs the (slow-ish) data/training
# pipeline if artifacts don't already exist — so a container restart
# doesn't retrain from scratch every time, but a fresh volume does.
set -e
cd /app

if [ ! -f "models/champion_model.json" ]; then
  echo "[entrypoint] No trained models found — running full pipeline from scratch..."
  python3 ml/data/generate_dataset.py
  python3 ml/preprocessing/run_pipeline.py
  python3 ml/training/train_models.py
  python3 ml/monitoring/run_mlops_pipeline.py
  echo "[entrypoint] Pipeline complete."
else
  echo "[entrypoint] Existing trained models found — skipping pipeline, starting API directly."
fi

exec sh -c "cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8001"
