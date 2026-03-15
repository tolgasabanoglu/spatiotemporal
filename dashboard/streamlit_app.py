"""
Streamlit Dashboard for Garmin Health Metrics

Run with:
    streamlit run dashboard/streamlit_app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import bigquery
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Config
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREDENTIALS_PATH = os.path.join(PROJECT_ROOT, "spatiotemporal-key.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

st.set_page_config(
    page_title="Spatiotemporal — Health & Environment",
    layout="wide"
)

@st.cache_data(ttl=3600)
def load_data():
    """Load data from BigQuery."""
    client = bigquery.Client()
    query = """
    SELECT * FROM `garmin_data.v_dashboard_daily`
    ORDER BY date
    """
    return client.query(query).to_dataframe()

@st.cache_data(ttl=3600)
def load_trends():
    """Load trend data."""
    client = bigquery.Client()
    query = """
    SELECT * FROM `garmin_data.v_dashboard_trends`
    ORDER BY date
    """
    return client.query(query).to_dataframe()

@st.cache_data(ttl=3600)
def load_monthly():
    """Load monthly summary."""
    client = bigquery.Client()
    query = """
    SELECT * FROM `garmin_data.v_dashboard_monthly`
    ORDER BY month_start
    """
    return client.query(query).to_dataframe()

# Load data
try:
    df = load_data()
    trends = load_trends()
    monthly = load_monthly()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Header
st.title("Spatiotemporal — Health & Environment Dashboard")
st.markdown(f"**Data range:** {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')} ({len(df)} days)")

# Sidebar filters
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Date Range",
    value=(df['date'].min(), df['date'].max()),
    min_value=df['date'].min(),
    max_value=df['date'].max()
)

if len(date_range) == 2:
    mask = (df['date'] >= pd.Timestamp(date_range[0])) & (df['date'] <= pd.Timestamp(date_range[1]))
    df_filtered = df[mask]
    trends_filtered = trends[(trends['date'] >= pd.Timestamp(date_range[0])) & (trends['date'] <= pd.Timestamp(date_range[1]))]
else:
    df_filtered = df
    trends_filtered = trends

# KPI Cards
st.header("Current Status")

# Get latest row with actual data (not all N/A)
latest = None
if len(df_filtered) > 0:
    for i in range(len(df_filtered) - 1, -1, -1):
        row = df_filtered.iloc[i]
        if pd.notna(row.get('avg_stress')):
            latest = row
            break

# Show latest data date (the date of the metrics with actual data)
if latest is not None:
    latest_date = latest['date']
    latest_date_obj = latest_date.date() if hasattr(latest_date, 'date') else latest_date
    today = pd.Timestamp.now().date()
    days_ago = (today - latest_date_obj).days

    if days_ago == 0:
        relative_time = "Today's data"
    elif days_ago == 1:
        relative_time = "Yesterday's data"
    else:
        relative_time = f"{days_ago} days old"

    st.markdown(f"**Latest Metrics:** {latest_date.strftime('%B %d, %Y')} ({relative_time})")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if latest is not None and pd.notna(latest['avg_stress']):
        st.metric("Avg Stress", f"{latest['avg_stress']:.0f}",
                  delta=f"{latest['avg_stress'] - df_filtered['avg_stress'].mean():.1f} vs avg")
    else:
        st.metric("Avg Stress", "N/A")

with col2:
    if latest is not None and pd.notna(latest['resting_hr']):
        st.metric("Resting HR", f"{latest['resting_hr']:.0f} bpm")
    else:
        st.metric("Resting HR", "N/A")

with col3:
    if latest is not None and pd.notna(latest['sleep_hours']):
        st.metric("Sleep", f"{latest['sleep_hours']:.1f} hrs")
    else:
        st.metric("Sleep", "N/A")

with col4:
    if latest is not None and pd.notna(latest['net_battery']):
        st.metric("Net Battery", f"{latest['net_battery']:.0f}",
                  delta="Recovery" if latest['net_battery'] > 0 else "Drain")
    else:
        st.metric("Net Battery", "N/A")

with col5:
    if latest is not None and pd.notna(latest['steps']):
        st.metric("Steps", f"{latest['steps']:,.0f}")
    else:
        st.metric("Steps", "No data")

# Coffee Recommendations
st.header("Today's Coffee Recommendation")

# Info box explaining the methodology
with st.expander("How does this recommendation work?"):
    st.markdown("""
    ### Two-Phase ML Café Recommendation System

    Instead of predicting a specific café directly, the system first learns **what kind of environment you need**
    from how you've been feeling — then finds which LAP Coffee locations currently match that environment.

    ---

    #### Phase 1 — Mood Clustering (Unsupervised)

    K-Means clustering (k=4) was applied to a full year of environmental data across all LAP Coffee locations.
    Each location × date observation was clustered based on dynamic environmental features only
    (static neighborhood features like parks/bars were excluded to avoid location memorisation).

    | Mood | NDVI | Nightlight | Temp | Rain | Description |
    |------|------|------------|------|------|-------------|
    | **summer_green** | 0.34 | low | 21°C | dry | Warm, green, low urban activity |
    | **sunny_urban** | 0.14 | very low | 24°C | dry | Hot, dry, city energy |
    | **winter_cozy** | 0.13 | high | 8°C | light | Cold, dark, sheltered |
    | **rainy_day** | 0.16 | mid | 19°C | heavy | Wet, mild, indoor |

    ---

    #### Phase 2 — Mood Classification (Supervised)

    A classifier is trained to predict your current mood from **lagged Garmin biometrics** —
    not just today's snapshot, but how you've been feeling over the past few days:

    | Feature | Lag | Why |
    |---------|-----|-----|
    | Stress | 1-day, 7-day rolling avg | Accumulated tension |
    | Sleep hours | 1-day, 3-day rolling avg | Sleep debt |
    | REM sleep | 1-day lag | Recovery quality |
    | Body battery | 1-day, 3-day rolling avg | Energy drain over time |
    | Resting HR | 1-day lag | Physiological baseline |

    ---

    #### Recommendation

    Once your mood is predicted, each café is scored against today's actual environmental conditions:

    ```
    final_score = 60% × environment_match + 40% × proximity
    ```

    - **Environment match**: Euclidean distance between today's café conditions and the predicted mood centroid
    - **Proximity**: Distance from current location

    Top 3 cafés are returned.

    ---

    #### Stress Analysis Model Results

    Trained on 384 days of Garmin data (Feb 2025 – Mar 2026):

    | Model | Task | Result |
    |-------|------|--------|
    | PCA | Dimensionality reduction | PC1 explains 59% variance (strain vs recovery axis) |
    | Logistic Regression | Classify high stress days (>50) | Test accuracy = 92.5% |

    **Top protective factors against high stress:**
    - Charged body battery (coefficient −0.82)
    - Body battery change (coefficient −0.56)

    ---
    *Clustering features: NDVI · nightlight · temp_max · temp_min · precip_mm*
    """)

if latest is not None:
    try:
        from coffee_recommender import get_recommendations

        # Calculate 7-day averages for more stable recommendations
        last_7_days = df_filtered.tail(7)

        if len(last_7_days) > 0:
            stress = last_7_days['avg_stress'].mean() if 'avg_stress' in last_7_days.columns else 50
            sleep = last_7_days['sleep_hours'].mean() if 'sleep_hours' in last_7_days.columns else 7
            net_battery = last_7_days['net_battery'].mean() if 'net_battery' in last_7_days.columns else 0
            resting_hr = last_7_days['resting_hr'].mean() if 'resting_hr' in last_7_days.columns else 60

            # Show what period is being used with key metrics
            days_used = len(last_7_days.dropna(subset=['avg_stress']))
            st.caption(f"Based on your last {days_used} days average: Stress {stress:.1f}, Sleep {sleep:.1f}hr, Battery {net_battery:.1f}")
        else:
            # Fallback to latest day
            stress = latest.get('avg_stress', 50)
            sleep = latest.get('sleep_hours', 7)
            net_battery = latest.get('net_battery', 0)
            resting_hr = latest.get('resting_hr', 60)

        # Get recommendations (increased range for more diversity)
        recommendations = get_recommendations(
            stress=stress,
            sleep_hours=sleep,
            net_battery=net_battery,
            resting_hr=resting_hr,
            max_distance_km=None,  # No distance limit — show all cafés
            top_n=3
        )

        # Display top recommendation in a card
        if recommendations and len(recommendations) > 0:
            rec = recommendations[0]

            # Create styled card
            st.markdown(f"""
            <div style="
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #ff6b6b;
            ">
                <h3 style="margin-top: 0;">{rec['cafe_name']}</h3>
                <p style="margin: 5px 0;"><b>Distance:</b> {rec['distance_km']} km from home</p>
                <p style="margin: 5px 0;"><b>Rating:</b> {rec['rating']}/5</p>
                <p style="margin: 5px 0;"><b>Your Stress:</b> {stress:.1f}/100 (7-day avg)</p>
                <p style="margin: 5px 0;"><b>Weather:</b> {rec['weather_temp']:.1f}°C{', ' + str(rec['weather_precip']) + 'mm rain' if rec['weather_precip'] > 0 else ''}</p>
                <p style="margin: 10px 0; font-style: italic;">{rec['reason']}</p>
                <p style="margin: 5px 0; color: #666;"><small>{rec['address']}</small></p>
            </div>
            """, unsafe_allow_html=True)

            # Show alternative recommendations
            if len(recommendations) > 1:
                with st.expander("See alternative recommendations"):
                    for i, alt_rec in enumerate(recommendations[1:], start=2):
                        st.markdown(f"""
                        **{i}. {alt_rec['cafe_name']}** - {alt_rec['distance_km']} km away
                        - {alt_rec['address']}
                        - {alt_rec['rating']}/5 rating
                        """)
        else:
            st.info("No recommendations available at this time.")

    except Exception as e:
        st.warning(f"Coffee recommendations unavailable: {str(e)}")
        import traceback
        with st.expander("Show error details"):
            st.code(traceback.format_exc())
else:
    st.info("Coffee recommendations will appear once health data is available.")

# Song Recommendations
st.header("Today's Song Recommendations")

if latest is not None and 'recommendations' in locals():
    try:
        from song_recommender import get_song_recommendations

        # Get the mood and weather from coffee recommendations
        rec = recommendations[0] if recommendations else None
        if rec:
            st.caption(f"Personalized playlist for your {rec['mood'].replace('_', ' ')} mood")

            # Get song recommendations
            songs = get_song_recommendations(
                mood_profile=rec['mood'],
                stress=stress,
                sleep_hours=sleep,
                weather_temp=rec['weather_temp'],
                weather_precip=rec['weather_precip']
            )

            # Display songs in a styled card
            for i, song in enumerate(songs, 1):
                st.markdown(f"""
                <div style="
                    background-color: #f0f9ff;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #3b82f6;
                    margin-bottom: 10px;
                ">
                    <p style="margin: 0; font-size: 16px;"><b>{i}. {song['title']}</b></p>
                    <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">by {song['artist']}</p>
                    <p style="margin: 8px 0 0 0; font-style: italic; font-size: 13px;">{song['reason']}</p>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"Song recommendations unavailable: {str(e)}")
        with st.expander("Show error details"):
            import traceback
            st.code(traceback.format_exc())
else:
    st.info("Song recommendations will appear once health data is available.")

# Stress Trend
st.header("Stress Trends")

fig_stress = go.Figure()
fig_stress.add_trace(go.Scatter(
    x=df_filtered['date'], y=df_filtered['avg_stress'],
    mode='lines', name='Daily Stress', opacity=0.5
))
fig_stress.add_trace(go.Scatter(
    x=trends_filtered['date'], y=trends_filtered['stress_7d_avg'],
    mode='lines', name='7-day Average', line=dict(width=3)
))
fig_stress.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="High Stress Threshold")
fig_stress.update_layout(
    title="Average Stress Level Over Time",
    xaxis_title="Date", yaxis_title="Stress Level",
    height=400
)
st.plotly_chart(fig_stress, use_container_width=True)

# Two column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sleep Analysis")
    fig_sleep = px.bar(
        df_filtered.tail(30), x='date', y='sleep_hours',
        color='sleep_hours',
        color_continuous_scale='Blues',
        title="Sleep Hours (Last 30 Days)"
    )
    fig_sleep.add_hline(y=7, line_dash="dash", line_color="green", annotation_text="7hr target")
    st.plotly_chart(fig_sleep, use_container_width=True)

with col2:
    st.subheader("Body Battery")
    fig_battery = go.Figure()
    fig_battery.add_trace(go.Bar(
        x=df_filtered.tail(30)['date'],
        y=df_filtered.tail(30)['charged'],
        name='Charged', marker_color='green'
    ))
    fig_battery.add_trace(go.Bar(
        x=df_filtered.tail(30)['date'],
        y=-df_filtered.tail(30)['drained'],
        name='Drained', marker_color='red'
    ))
    fig_battery.update_layout(
        title="Body Battery Charged vs Drained (Last 30 Days)",
        barmode='relative', height=400
    )
    st.plotly_chart(fig_battery, use_container_width=True)

# Relationships
st.header("Relationships")

col1, col2 = st.columns(2)

with col1:
    # Sleep battery recovery vs stress (r = -0.637)
    fig_slp_bat = px.scatter(
        df_filtered.dropna(subset=['sleep_body_battery_change', 'avg_stress']),
        x='sleep_body_battery_change', y='avg_stress',
        trendline='ols',
        labels={'sleep_body_battery_change': 'Battery Recovered During Sleep', 'avg_stress': 'Avg Stress'},
        title="Sleep Recovery vs Stress"
    )
    st.plotly_chart(fig_slp_bat, use_container_width=True)

with col2:
    # Net battery vs stress
    fig_corr2 = px.scatter(
        df_filtered.dropna(subset=['net_battery', 'avg_stress']),
        x='net_battery', y='avg_stress',
        trendline='ols',
        labels={'net_battery': 'Net Body Battery', 'avg_stress': 'Avg Stress'},
        title="Body Battery vs Stress"
    )
    st.plotly_chart(fig_corr2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    # Resting HR rolling avg vs stress rolling avg — dual axis
    trend_df = df_filtered[['date', 'avg_stress', 'resting_hr']].dropna().copy()
    trend_df['stress_7d'] = trend_df['avg_stress'].rolling(7).mean()
    trend_df['hr_7d'] = trend_df['resting_hr'].rolling(7).mean()
    trend_df = trend_df.dropna()
    fig_hr = go.Figure()
    fig_hr.add_trace(go.Scatter(
        x=trend_df['date'], y=trend_df['stress_7d'],
        name='Stress (7d avg)', line=dict(color='coral')
    ))
    fig_hr.add_trace(go.Scatter(
        x=trend_df['date'], y=trend_df['hr_7d'],
        name='Resting HR (7d avg)', line=dict(color='steelblue'),
        yaxis='y2'
    ))
    fig_hr.update_layout(
        title="Resting HR vs Stress Over Time",
        yaxis=dict(title='Stress Level'),
        yaxis2=dict(title='Resting HR (bpm)', overlaying='y', side='right'),
        height=400, legend=dict(orientation='h')
    )
    st.plotly_chart(fig_hr, use_container_width=True)

with col4:
    # Stress by day of week
    dow_df = df_filtered[['date', 'avg_stress']].dropna().copy()
    dow_df['day_of_week'] = pd.to_datetime(dow_df['date']).dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_df['day_of_week'] = pd.Categorical(dow_df['day_of_week'], categories=day_order, ordered=True)
    fig_dow = px.box(
        dow_df.sort_values('day_of_week'),
        x='day_of_week', y='avg_stress',
        labels={'day_of_week': '', 'avg_stress': 'Avg Stress'},
        title="Stress by Day of Week"
    )
    fig_dow.add_hline(y=50, line_dash='dash', line_color='red', opacity=0.4)
    st.plotly_chart(fig_dow, use_container_width=True)

# HRV vs Stress
st.header("HRV vs Stress")
hrv_df = df_filtered.dropna(subset=['avg_overnight_hrv', 'avg_stress'])
if len(hrv_df) > 10:
    col1, col2 = st.columns(2)
    with col1:
        fig_hrv = px.scatter(
            hrv_df, x='avg_overnight_hrv', y='avg_stress',
            trendline='ols',
            labels={'avg_overnight_hrv': 'Overnight HRV (ms)', 'avg_stress': 'Avg Stress'},
            title=f"HRV vs Stress (r = -0.59, n={len(hrv_df)})"
        )
        st.plotly_chart(fig_hrv, use_container_width=True)
    with col2:
        hrv_trend = hrv_df[['date', 'avg_overnight_hrv', 'avg_stress']].copy()
        hrv_trend['hrv_7d'] = hrv_trend['avg_overnight_hrv'].rolling(7).mean()
        hrv_trend['stress_7d'] = hrv_trend['avg_stress'].rolling(7).mean()
        hrv_trend = hrv_trend.dropna()
        fig_hrv_trend = go.Figure()
        fig_hrv_trend.add_trace(go.Scatter(
            x=hrv_trend['date'], y=hrv_trend['stress_7d'],
            name='Stress (7d avg)', line=dict(color='coral')
        ))
        fig_hrv_trend.add_trace(go.Scatter(
            x=hrv_trend['date'], y=hrv_trend['hrv_7d'],
            name='HRV (7d avg)', line=dict(color='mediumseagreen'),
            yaxis='y2'
        ))
        fig_hrv_trend.update_layout(
            title="HRV vs Stress Over Time",
            yaxis=dict(title='Stress Level'),
            yaxis2=dict(title='HRV (ms)', overlaying='y', side='right'),
            height=400, legend=dict(orientation='h')
        )
        st.plotly_chart(fig_hrv_trend, use_container_width=True)
else:
    st.info("Not enough HRV data available yet.")

# Stress Calendar Heatmap
st.header("Stress Calendar")
cal_df = df_filtered[['date', 'avg_stress']].dropna().copy()
cal_df['date'] = pd.to_datetime(cal_df['date'])
cal_df['week'] = cal_df['date'].dt.isocalendar().week.astype(int)
cal_df['year'] = cal_df['date'].dt.isocalendar().year.astype(int)
cal_df['year_week'] = cal_df['year'].astype(str) + '-W' + cal_df['week'].astype(str).str.zfill(2)
cal_df['weekday'] = cal_df['date'].dt.weekday  # 0=Mon, 6=Sun

pivot = cal_df.pivot_table(index='weekday', columns='year_week', values='avg_stress', aggfunc='mean')
pivot = pivot.sort_index()

fig_cal = go.Figure(data=go.Heatmap(
    z=pivot.values,
    x=pivot.columns.tolist(),
    y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    colorscale='RdYlGn_r',
    zmin=20, zmax=70,
    colorbar=dict(title='Stress'),
    hoverongaps=False
))
fig_cal.update_layout(
    title="Daily Stress Heatmap (green = low stress, red = high)",
    xaxis=dict(showticklabels=False),
    height=280,
    margin=dict(l=40, r=40, t=50, b=20)
)
st.plotly_chart(fig_cal, use_container_width=True)

# Model Metrics
st.header("Model Results")

model_features = ['charged', 'drained', 'net_battery', 'resting_hr',
                  'sleep_hours', 'deep_sleep_hours', 'rem_sleep_hours', 'sleep_body_battery_change']
model_data = df_filtered[model_features + ['avg_stress']].dropna()

if len(model_data) > 30:
    X_raw = model_data[model_features]
    y_binary = (model_data['avg_stress'] > 50).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    col1, col2, col3 = st.columns(3)

    # --- Logistic Regression coefficients ---
    with col1:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        lr = LogisticRegression(max_iter=1000, C=0.1)
        lr.fit(X_train, y_train)
        train_acc = lr.score(X_train, y_train)
        test_acc = lr.score(X_test, y_test)

        coef_df = pd.DataFrame({
            'feature': model_features,
            'coefficient': lr.coef_[0]
        }).sort_values('coefficient')

        colors = ['mediumseagreen' if c < 0 else 'coral' for c in coef_df['coefficient']]
        fig_coef = go.Figure(go.Bar(
            x=coef_df['coefficient'], y=coef_df['feature'],
            orientation='h', marker_color=colors
        ))
        fig_coef.update_layout(
            title=f"Logistic Regression Coefficients<br><sub>Test accuracy: {test_acc:.1%} · Train: {train_acc:.1%} · High stress days: {y_binary.sum()}/{len(y_binary)}</sub>",
            xaxis_title="Coefficient (green = protective, red = risk)",
            height=400
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    # --- PCA explained variance ---
    with col2:
        pca = PCA()
        pca.fit(X_scaled)
        evr = pca.explained_variance_ratio_[:6] * 100
        cumulative = np.cumsum(evr)

        fig_pca = go.Figure()
        fig_pca.add_trace(go.Bar(
            x=[f'PC{i+1}' for i in range(len(evr))],
            y=evr, name='Individual', marker_color='steelblue'
        ))
        fig_pca.add_trace(go.Scatter(
            x=[f'PC{i+1}' for i in range(len(evr))],
            y=cumulative, name='Cumulative',
            mode='lines+markers', line=dict(color='coral'), yaxis='y2'
        ))
        fig_pca.update_layout(
            title=f"PCA Explained Variance<br><sub>PC1: {evr[0]:.1f}% · PC1+PC2: {cumulative[1]:.1f}% · 3 PCs: {cumulative[2]:.1f}%</sub>",
            yaxis=dict(title='Variance Explained (%)'),
            yaxis2=dict(title='Cumulative (%)', overlaying='y', side='right'),
            height=400, legend=dict(orientation='h')
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    # --- Feature correlations with stress ---
    with col3:
        corr_vals = [(f, model_data[f].corr(model_data['avg_stress'])) for f in model_features]
        corr_df = pd.DataFrame(corr_vals, columns=['feature', 'correlation']).sort_values('correlation')
        colors_corr = ['mediumseagreen' if c < 0 else 'coral' for c in corr_df['correlation']]
        fig_corr = go.Figure(go.Bar(
            x=corr_df['correlation'], y=corr_df['feature'],
            orientation='h', marker_color=colors_corr
        ))
        fig_corr.update_layout(
            title="Feature Correlation with Stress<br><sub>green = reduces stress · red = increases stress</sub>",
            xaxis_title="Pearson r",
            height=400
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# Monthly Summary
st.header("Monthly Summary")
fig_monthly = go.Figure()
fig_monthly.add_trace(go.Bar(
    x=monthly['month_name'], y=monthly['avg_stress'],
    name='Avg Stress', marker_color='coral'
))
fig_monthly.add_trace(go.Scatter(
    x=monthly['month_name'], y=monthly['avg_sleep_hours'] * 5,  # Scale for visibility
    name='Avg Sleep (x5)', mode='lines+markers', yaxis='y2'
))
fig_monthly.update_layout(
    title="Monthly Stress and Sleep",
    yaxis=dict(title='Stress Level'),
    yaxis2=dict(title='Sleep Hours (scaled)', overlaying='y', side='right'),
    height=400
)
st.plotly_chart(fig_monthly, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*Data from Garmin Connect via BigQuery*")
