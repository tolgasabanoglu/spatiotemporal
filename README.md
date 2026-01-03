# Spatiotemporal

A personal data analysis toolkit that integrates movement, health, and environmental context across space and time.

This project processes GPS tracks (from Garmin), biometric data (steps, stress, sleep), and environmental layers (normalized difference vegetation index (NDVI)) to create enriched, daily spatiotemporal life summaries.

---

## Project Structure

- `data/` — All datasets  
  - `raw/` — Raw data exports (e.g. GPX, JSON, CSV)  
  - `processed/` — Cleaned, merged data  
  - `external/` — Environmental data (e.g. NDVI, elevation)

- `notebooks/` — Jupyter notebooks for exploration  
  - `analysis.ipynb` — fetching the data, A/B test and regression results and visuals


- `garmin/` — Python automation scripts  
  - `parse_garmin.py`
  - `activities.py`
  - `load_to_bigquery.py` — Upload processed JSON data to BigQuery


- `config/` — Configuration and secrets  
  - `config.yaml`

- `env/` — Python environment setup  
  - `environment.yml`

- `.gitignore` — Files/folders to exclude from Git  
- `requirements.txt` — Python dependencies (if using pip)  
- `README.md` — Project overview and documentation



---

## Features

- Parse location data from Garmin
- Import and process Garmin health metrics (steps, sleep, stress)
- Query NDVI via Google Earth Engine - or create synthetic ndvisß
- Aggregate all sources into daily summaries
- Upload processed data to **BigQuery** for storage and querying
- Visualize movement and metrics with maps and charts

---

## A/B Testing

This project includes a framework for **personal A/B testing** to analyze how different habits, routines, and environmental exposures affect key health metrics.

### Examples:
- Several biometrics variables → impact on stress
- High NDVI (green space) vs. low NDVI → impact on average stress
- Seasonal shifts, regression test, etc.

### A/B Testing Components:
- Test plans in YAML: `experiments/ab_test_plan.yaml`
- Run interactively in: `notebooks/05_ab_analysis.ipynb`
- Optional automation via: `scripts/run_ab_analysis.py`
- Outputs: group stats, p-values, charts → saved in `experiments/results/`

---

