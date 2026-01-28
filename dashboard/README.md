# Garmin Health Dashboard

## Overview

This folder contains resources for visualizing Garmin health metrics using Looker Studio (formerly Google Data Studio) connected to BigQuery.

## Setup

### 1. Deploy Dashboard Views to BigQuery

```bash
cd /Users/tolgasabanoglu/Desktop/github/spatiotemporal
python dashboard/deploy_dashboard_views.py
```

This creates optimized views for dashboards:
- `v_dashboard_daily` - Daily metrics with categories
- `v_dashboard_weekly` - Weekly aggregates
- `v_dashboard_monthly` - Monthly summaries
- `v_dashboard_correlations` - For scatter plots
- `v_dashboard_trends` - 7-day rolling averages

### 2. Create Looker Studio Dashboard

1. Go to [Looker Studio](https://lookerstudio.google.com)
2. Click **Create** → **Data Source**
3. Select **BigQuery** connector
4. Choose project: `spatiotemporal-473309`
5. Choose dataset: `garmin_data`
6. Select view: `v_dashboard_daily` (or others as needed)

### 3. Suggested Dashboard Layout

#### Page 1: Overview
| Widget | Data Source | Metrics |
|--------|-------------|---------|
| Scorecard | v_dashboard_daily | Current avg_stress |
| Scorecard | v_dashboard_daily | Current resting_hr |
| Time Series | v_dashboard_trends | stress_7d_avg over time |
| Pie Chart | v_dashboard_daily | stress_category distribution |

#### Page 2: Sleep Analysis
| Widget | Data Source | Metrics |
|--------|-------------|---------|
| Time Series | v_dashboard_daily | sleep_hours by date |
| Bar Chart | v_dashboard_weekly | avg_sleep_hours by week |
| Scatter | v_dashboard_correlations | sleep_hours vs next_day_stress |

#### Page 3: Recovery
| Widget | Data Source | Metrics |
|--------|-------------|---------|
| Gauge | v_dashboard_daily | net_battery (latest) |
| Time Series | v_dashboard_daily | charged, drained over time |
| Bar Chart | v_dashboard_monthly | avg_net_battery by month |

## Alternative: Streamlit Dashboard

For a local Python dashboard, use:

```bash
pip install streamlit plotly
streamlit run dashboard/streamlit_app.py
```

## Views Reference

### v_dashboard_daily
All daily metrics with derived fields like `stress_level_num`, `recovery_status`, `activity_level`.

### v_dashboard_weekly
Weekly aggregates including `high_stress_days`, `avg_sleep_hours`, `days_with_steps`.

### v_dashboard_monthly
Monthly summaries with `high_stress_pct`, `poor_sleep_days`, `steps_coverage_pct`.

### v_dashboard_correlations
Includes lagged fields (`next_day_stress`, `prev_day_sleep`) for correlation analysis.

### v_dashboard_trends
7-day rolling averages and week-over-week changes for trend analysis.

## Data Refresh

Data is refreshed when you run:
```bash
python garmin/parse_garmin.py      # Fetch new data
python garmin/load_to_bigquery.py  # Upload to BigQuery
python garmin/deploy_views.py      # Refresh base views
python dashboard/deploy_dashboard_views.py  # Refresh dashboard views
```

Looker Studio will automatically pick up new data on each report view.
