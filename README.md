# Spatiotemporal

A personal analytics project that connects Garmin biometrics, satellite environmental data, and weather to answer two questions:

1. **What predicts my stress levels?**
2. **Which café should I visit today?**

---

## Methodology

### Part 1 — Stress Analysis

```
Garmin API → BigQuery → SQL views → notebooks/stress_analysis.ipynb
```

Daily biometrics (stress, sleep, heart rate, body battery) are fetched from Garmin Connect, stored in BigQuery, and cleaned via SQL views. The notebook runs three analyses:

- **PCA** — identifies the main dimensions of variation (PC1: strain vs recovery, 59% variance)
- **Logistic regression** — classifies high stress days >50 (test accuracy: 92.5%)

Key finding: charged body battery is the strongest protective factor against high stress days.

| Model | Task | Result |
|-------|------|--------|
| PCA | Dimensionality reduction | PC1 explains 59% variance (strain vs recovery) |
| Logistic Regression | Classify high stress days (>50) | Test accuracy = 92.5% |

---

### Part 2 — Café Recommendation

> **TODO:** Add new LAP Coffee locations

```
LAP Coffee locations + environmental data → K-Means clustering → mood labels
Garmin biometrics (lagged) → mood classifier → café ranking
```

**Phase 1 — Mood clustering (unsupervised)**

K-Means (k=4) on environmental features (NDVI, nightlight, temperature, rain) across all LAP Coffee locations and dates. Produces 4 mood-environment types:

| Mood | Temp | NDVI | Nightlight | Rain |
|------|------|------|------------|------|
| `summer_green` | 21°C | 0.34 | low | dry |
| `sunny_urban` | 24°C | 0.14 | very low | dry |
| `winter_cozy` | 8°C | 0.13 | high | light |
| `rainy_day` | 19°C | 0.16 | mid | heavy |

**Phase 2 — Mood classification (supervised)**

Classifier trained on lagged Garmin biometrics (stress, sleep, body battery, HR over 1–7 days) to predict which mood type you need today.

**Recommendation**

Each café is scored against today's actual environmental conditions:
```
final_score = 60% × environment_match + 40% × proximity
```

---

## Project Structure

```
spatiotemporal/
├── garmin/                    # Garmin API → BigQuery ETL
├── lap-cafe/
│   ├── ingestion/             # Fetch café locations + env data
│   ├── clustering/            # Phase 1: K-Means mood clustering
│   ├── classification/        # Phase 2: biometrics → mood classifier
│   └── recommender.py         # Café scoring + ranking
├── dashboard/                 # Streamlit app
├── notebooks/
│   ├── stress_analysis.ipynb
│   └── lap_mood_clustering.ipynb
└── data/
    ├── raw/                   # Garmin JSON exports
    └── lap-cafe/              # Café env data + cluster outputs
```

---

## Dashboard

![Health & Environment Dashboard](dashboard/screenshots/Screenshot%202026-03-15%20at%2018.04.01.png)
![Stress Trends & Analysis](dashboard/screenshots/Screenshot%202026-03-15%20at%2018.04.35.png)
![Relationships & HRV](dashboard/screenshots/Screenshot%202026-03-15%20at%2018.38.38.png)
![Model Results](dashboard/screenshots/Screenshot%202026-03-15%20at%2018.42.04.png)

---

## Stack

Python · scikit-learn · BigQuery · Streamlit · Google Earth Engine · Plotly

---

## Run

```bash
# Refresh Garmin data
python garmin/parse_garmin.py
python garmin/load_to_bigquery.py
python garmin/deploy_views.py

# Dashboard
streamlit run dashboard/streamlit_app.py
```
