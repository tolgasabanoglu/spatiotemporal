# 🧭 Spatiotemporal

A personal data analysis toolkit that integrates movement, health, and environmental context across space and time.

This project processes GPS tracks (from Google Maps or Garmin), biometric data (steps, stress, sleep), and environmental layers (NDVI, elevation) to create enriched, daily spatiotemporal life summaries.

---

## 📁 Project Structure

- `data/` — All datasets  
  - `raw/` — Raw data exports (e.g. GPX, JSON, CSV)  
  - `processed/` — Cleaned, merged data  
  - `external/` — Environmental data (e.g. NDVI, elevation)

- `notebooks/` — Jupyter notebooks for exploration  
  - `01_parse_location.ipynb`  
  - `02_ndvi_merge.ipynb`  
  - `03_health_merge.ipynb`  
  - `04_visualize.ipynb`

- `scripts/` — Python automation scripts  
  - `parse_garmin.py`  
  - `fetch_ndvi.py`  
  - `fetch_elevation.py`  
  - `merge_all.py`

- `config/` — Configuration and secrets  
  - `config.yaml`

- `env/` — Python environment setup  
  - `environment.yml`

- `.gitignore` — Files/folders to exclude from Git  
- `requirements.txt` — Python dependencies (if using pip)  
- `README.md` — Project overview and documentation



---

## 🚀 Features

- Parse location data from Garmin or Google Maps
- Import and process Garmin health metrics (steps, sleep, stress)
- Query NDVI and elevation data via Google Earth Engine
- Aggregate all sources into daily summaries
- Visualize movement and metrics with maps and charts

---

## 🛠️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/tolgasabanoglu/spatiotemporal.git
cd spatiotemporal

conda env create -f env/environment.yml
conda activate spatiotemporal


Example Output (Planned)

date	lat	lon	steps	stress	ndvi	elevation
2025-06-01	40.71	-74.00	12,345	35	0.67	12.3 m

📍 Inspiration

Chronobiology
Human geography
Digital lifelogging
Nature-health connection research

📚 To Do
 Implement Garmin parser
 Integrate Earth Engine NDVI
 Merge health + GPS + environment
 Visualize time-space-health patterns