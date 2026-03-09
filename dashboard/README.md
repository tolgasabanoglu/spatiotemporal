# Garmin Health Dashboard

An interactive Streamlit dashboard that visualizes personal health metrics from Garmin, powered by BigQuery and enriched with ML-based recommendations.

---

## Features

- **Health KPIs**: Real-time stress, heart rate, sleep, body battery, and steps
- **Coffee Recommendations**: Random Forest Classifier matching health state to LAP Coffee locations in Berlin
- **Song Recommendations**: Gemini API generating playlists based on mood profile and weather
- **Stress Trends**: Daily stress with 7-day rolling average
- **Sleep Analysis**: Sleep hours over last 30 days
- **Body Battery**: Charged vs drained visualization
- **Correlations**: Sleep vs stress, body battery vs stress scatter plots
- **Feature Importance**: Random Forest analysis of stress predictors
- **Monthly Summary**: Stress and sleep trends by month

---

## Run the Dashboard

```bash
cd spatiotemporal
source venv/bin/activate
streamlit run dashboard/streamlit_app.py
```

Open http://localhost:8501

---

## Data Refresh

Data is fetched and loaded automatically via Apache Airflow (daily at 2 AM). To run manually:

```bash
source venv/bin/activate
python garmin/parse_garmin.py           # Fetch new data from Garmin API
python garmin/load_to_bigquery.py       # Upload to BigQuery
python garmin/deploy_views.py           # Refresh base views
python dashboard/deploy_dashboard_views.py  # Refresh dashboard views
```

---

## BigQuery Views

| View | Purpose |
|------|---------|
| `v_dashboard_daily` | Daily metrics with categories and derived fields |
| `v_dashboard_weekly` | Weekly aggregates (high stress days, avg sleep) |
| `v_dashboard_monthly` | Monthly summaries with coverage stats |
| `v_dashboard_correlations` | Lagged fields for sleep/stress correlation analysis |
| `v_dashboard_trends` | 7-day rolling averages and week-over-week changes |

---

## Looker Studio (Alternative)

Connect directly to BigQuery:

1. Go to [Looker Studio](https://lookerstudio.google.com)
2. Create → Data Source → BigQuery
3. Project: `spatiotemporal-473309`, Dataset: `garmin_data`
4. Select any `v_dashboard_*` view

---

## Screenshots

### Current Status & Recommendations
![Dashboard Overview](screenshots/Screenshot%202026-02-03%20at%2015.09.21.png)
*Health metrics KPIs with ML-powered coffee recommendations and GenAI song suggestions*

### Trends & Analysis
![Stress and Sleep Trends](screenshots/Screenshot%202026-02-03%20at%2015.10.11.png)
*Daily stress trends with 7-day rolling average, sleep hours (last 30 days), body battery*

### Correlations & Feature Importance
![Correlations and ML Insights](screenshots/Screenshot%202026-02-03%20at%2015.10.30.png)
*Sleep vs Stress and Body Battery vs Stress scatter plots with Random Forest feature importance*
