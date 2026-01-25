# Spatiotemporal

A personal health analytics toolkit that integrates movement, health metrics, and environmental context across space and time.

This project processes GPS tracks and activities from Garmin, biometric data (steps, stress, sleep, heart rate, body battery), and environmental layers (NDVI) to create enriched daily summaries and uncover insights about how lifestyle factors affect well-being.

---

## Key Findings

### Running and Next-Day Stress
- **Running days show 5.7 points lower next-day stress** (35.5 vs 41.2)
- Statistically significant (p = 0.005)
- Running acts as a stress buffer for the following day

### Sleep and Next-Day Stress
- **Sleep duration matters**: longer sleep correlates with lower next-day stress (r = -0.146, p = 0.008)
- Sleep composition (deep/REM percentages) showed weaker effects than total hours

### Predictive Model
- Today's stress is the strongest predictor of tomorrow's stress
- Running and sleep hours are the main modifiable factors that reduce next-day stress

---

## Project Structure

```
spatiotemporal/
├── garmin/                    # Data fetching scripts
│   ├── parse_garmin.py       # Fetch daily metrics (steps, sleep, stress, etc.)
│   ├── activities.py         # Fetch activity data (running, cycling, etc.)
│   └── load_to_bigquery.py   # Upload data to BigQuery
│
├── notebooks/                 # Analysis notebooks
│   ├── analysis.ipynb        # Main analysis: PCA, ML models, feature importance
│   ├── next_day_analysis.ipynb # Lagged analysis: running/sleep → next-day stress
│   └── algorithms.py         # Reusable ML algorithms (Random Forest, Logistic Reg, NN)
│
├── data/
│   ├── raw/                  # Raw JSON exports from Garmin API
│   │   └── activities/       # Activity-specific data
│   ├── processed/            # Cleaned, merged datasets
│   └── external/             # Environmental data (NDVI, elevation)
│
├── config/
│   └── config.yaml           # Configuration placeholders
│
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd spatiotemporal
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Garmin credentials

Create a `.env` file in the `garmin/` directory:

```bash
echo "USERNAME=your_garmin_email@example.com" > garmin/.env
echo "PASSWORD=your_garmin_password" >> garmin/.env
```

### 3. (Optional) Configure BigQuery

Place your Google Cloud service account key as `spatiotemporal-key.json` in the project root.

---

## Usage

### Fetch Latest Data

```bash
source venv/bin/activate

# Fetch daily metrics (steps, sleep, stress, heart rate, body battery)
cd garmin && python parse_garmin.py

# Fetch activities (running, cycling, etc.)
python activities.py
```

### Run Analysis

```bash
# Start Jupyter
jupyter notebook notebooks/

# Or run notebooks directly
jupyter nbconvert --to notebook --execute --inplace notebooks/next_day_analysis.ipynb
```

---

## Data Sources

| Source | Metrics |
|--------|---------|
| Garmin Connect | Steps, sleep, stress, heart rate, body battery, activities |
| Activities | Running distance, duration, HR zones, training effect, calories |
| Environmental | NDVI (vegetation index), elevation |

---

## Analysis Notebooks

### `analysis.ipynb`
- Loads data from BigQuery
- PCA analysis to identify key health dimensions
- Random Forest regression for stress prediction
- Logistic regression for high-stress classification
- Feature importance visualization

### `next_day_analysis.ipynb`
- **Lagged analysis**: How today's actions affect tomorrow
- Running → next-day stress (t-test, correlation)
- Sleep quality → next-day stress
- Multiple regression combining all factors
- Exports processed data to CSV

---

## ML Models (algorithms.py)

| Model | Use Case |
|-------|----------|
| Random Forest | Predict continuous stress levels, feature importance |
| Logistic Regression | Classify high vs low stress days |
| Neural Network (MLP) | Non-linear pattern detection |

---

## Future Work

- [ ] Add NDVI (green space exposure) correlation analysis
- [ ] Cumulative sleep debt effects
- [ ] Running intensity zones vs recovery
- [ ] Seasonal patterns in stress/sleep
- [ ] Automated daily data fetching (cron/scheduler)

---

## Tech Stack

- **Python 3.13** with pandas, scikit-learn, scipy, matplotlib, seaborn
- **garminconnect** for Garmin API access
- **Google BigQuery** for data storage
- **Jupyter** for interactive analysis
