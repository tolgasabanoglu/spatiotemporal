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

### Predictive Model Insights
- **Stress baseline carries forward**: Yesterday's stress level influences today, but interventions can break the cycle
- **Running impact is substantial**: Each running session reduces next-day stress by 5.7 points (14% improvement)
- **Sleep is cumulative**: Each additional hour of sleep correlates with measurably lower stress (r = -0.146)
- **Modifiable factors matter**: Running and sleep are the most actionable levers for stress management

---

## Project Structure

```
spatiotemporal/
├── garmin/                       # Data fetching and ETL
│   ├── parse_garmin.py          # Fetch daily metrics from Garmin API
│   ├── activities.py            # Fetch activity data (running, cycling, etc.)
│   ├── load_to_bigquery.py      # Upload raw JSON to BigQuery
│   ├── deploy_views.py          # Deploy SQL transformation views
│   └── sql/
│       └── views.sql            # BigQuery views for data transformation
│
├── dashboard/                    # Visualization
│   ├── streamlit_app.py         # Interactive Streamlit dashboard
│   ├── coffee_recommender.py    # ML-powered café recommendations (RFC)
│   ├── song_recommender.py      # GenAI-powered music suggestions (Gemini API)
│   ├── COFFEE_RECOMMENDATIONS.md # Technical documentation for RFC model
│   ├── deploy_dashboard_views.py # Deploy dashboard-specific views
│   ├── looker_views.sql         # Views optimized for Looker Studio
│   └── README.md                # Dashboard setup instructions
│
├── notebooks/                    # Analysis notebooks
│   ├── analysis.ipynb           # Main analysis: PCA, ML models, feature importance
│   ├── next_day_analysis.ipynb  # Lagged analysis: running/sleep → next-day stress
│   └── algorithms.py            # Reusable ML algorithms
│
├── data/
│   ├── raw/                     # Raw JSON exports from Garmin API
│   │   └── activities/          # Activity-specific data
│   ├── processed/               # Cleaned, merged datasets
│   └── external/                # Environmental data (NDVI, elevation)
│
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Data Pipeline

```
Garmin API → parse_garmin.py → data/raw/*.json
                                    ↓
                         load_to_bigquery.py
                                    ↓
                    garmin_data.garmin_raw_data (raw JSON)
                                    ↓
                          deploy_views.py
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
            v_daily_metrics                 v_dashboard_daily
            (analysis-ready)                (dashboard-ready)
                    ↓                               ↓
              notebooks/                    dashboard/streamlit_app.py
```

### BigQuery Views

| View | Purpose |
|------|---------|
| `v_steps_daily` | Daily step totals, NULL for invalid days (steps=0) |
| `v_body_battery_daily` | Charged/drained metrics |
| `v_heart_rate_daily` | Resting HR, min/max HR |
| `v_stress_daily` | Avg/max stress with categories |
| `v_sleep_daily` | Sleep hours, HRV, body battery change |
| `v_daily_metrics` | Merged daily view for analysis |
| `v_data_quality_summary` | Monthly coverage stats |
| `v_dashboard_*` | Dashboard-optimized views with trends |

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

### 3. Configure BigQuery

Place your Google Cloud service account key as `spatiotemporal-key.json` in the project root.

---

## Usage

### Fetch and Load Data

```bash
source venv/bin/activate

# 1. Fetch daily metrics from Garmin API
python garmin/parse_garmin.py

# 2. Fetch activities (running, cycling, etc.)
python garmin/activities.py

# 3. Upload to BigQuery
python garmin/load_to_bigquery.py

# 4. Deploy transformation views
python garmin/deploy_views.py

# 5. Deploy dashboard views
python dashboard/deploy_dashboard_views.py
```

### Run Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

Open http://localhost:8501 to view the dashboard.

### Run Analysis Notebooks

```bash
jupyter notebook notebooks/
```

---

## Dashboard Features

- **Current Status**: KPI cards for stress, HR, sleep, body battery, steps
- **Coffee Recommendations** (NEW): ML-powered café suggestions based on health metrics and mood
  - Random Forest Classifier trained on 16 LAP Coffee locations
  - Maps health biometrics → mood profiles → environmental features → café predictions
  - Considers proximity, weather, greenness (NDVI), and neighborhood characteristics
  - Detailed explanations with confidence scores
- **Song Recommendations** (NEW): GenAI-powered music suggestions matching mood and weather
  - Gemini API integration with curated fallback playlists
  - Personalized selections for 6 mood profiles (cozy, green, buzz, rainy, recharge, balanced)
- **Stress Trends**: Daily stress with 7-day rolling average
- **Sleep Analysis**: Sleep hours over last 30 days
- **Body Battery**: Charged vs drained visualization
- **Correlations**: Sleep vs stress, body battery vs stress scatter plots
- **Feature Importance**: Random Forest analysis of stress predictors
- **Monthly Summary**: Stress and sleep trends by month

---

## Dashboard Screenshots

### Health Metrics Overview
[Screenshot placeholder - Current Status KPIs showing stress, HR, sleep, battery, steps]

### Coffee Recommendations
[Screenshot placeholder - ML-powered café recommendation card with mood profile, distance, rating, and reasoning]

*The coffee recommendation system uses a Random Forest Classifier trained on environmental data (parks, bars, greenness, nightlight intensity) to predict which LAP Coffee location best matches your current health state and weather conditions.*

### Song Recommendations
[Screenshot placeholder - GenAI-generated playlist with 3 song suggestions matching mood and weather]

*Song recommendations are powered by Google's Gemini API, creating personalized playlists based on your mood profile (derived from health metrics) and current weather.*

### Stress & Sleep Trends
[Screenshot placeholder - Interactive Plotly charts showing daily stress trends with 7-day rolling average and monthly stress/sleep comparison]

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
- Loads data from BigQuery views
- PCA analysis to identify key health dimensions
- Random Forest regression for stress prediction
- Logistic regression for high-stress classification
- Feature importance visualization

### `next_day_analysis.ipynb`
- **Lagged analysis**: How today's actions affect tomorrow
- Running → next-day stress (t-test, correlation)
- Sleep quality → next-day stress
- Multiple regression combining all factors

---

## Tech Stack

- **Python 3.13** with pandas, scikit-learn, scipy, numpy
- **garminconnect** for Garmin API access
- **Google BigQuery** for data storage and transformation
- **Streamlit** for interactive dashboard
- **Plotly** for visualizations
- **Jupyter** for analysis notebooks

---

## Future Work

- [ ] Add NDVI (green space exposure) correlation analysis
- [ ] Cumulative sleep debt effects
- [ ] Running intensity zones vs recovery
- [ ] Seasonal patterns in stress/sleep
- [ ] Automated daily data fetching (cron/scheduler)
- [ ] Deploy dashboard to Cloud Run
