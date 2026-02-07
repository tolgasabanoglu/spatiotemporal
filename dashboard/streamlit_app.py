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
from sklearn.ensemble import RandomForestRegressor

# Config
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREDENTIALS_PATH = os.path.join(PROJECT_ROOT, "spatiotemporal-key.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

st.set_page_config(
    page_title="Garmin Health Dashboard",
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
st.title("Garmin Health Dashboard")
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
    ### Machine Learning-Based Café Recommendation System

    This system uses a **Random Forest Classifier (RFC)** trained on full-year environmental data from 15 LAP Coffee locations in Berlin. Here's how it works:

    #### 1. Health-to-Mood Mapping
    Your Garmin biometrics are analyzed to determine your current mood profile:
    - **Stress level** (0-100): High stress suggests need for calm environments
    - **Sleep hours**: Poor sleep indicates recovery needs
    - **Body battery**: Energy levels guide activity preferences
    - **Resting heart rate**: Overall wellness indicator

    **Example mood profiles:**
    - *Cozy Indoor*: High stress + cold weather → need warmth & comfort
    - *Green Nature*: High stress + nice weather → seek restorative greenery
    - *Buzz Urban*: Low stress + energized → want vibrant social atmosphere
    - *Rainy Retreat*: Rainy weather → sheltered cozy space

    #### 2. Mood-to-Environmental Features Translation
    Each mood profile maps to specific environmental characteristics:

    | Feature | Description | Example Values |
    |---------|-------------|----------------|
    | **Parks (1km)** | Number of parks within 1km | Cozy: 8, Buzz: 3 |
    | **Bars (500m)** | Open bars within 500m | Cozy: 2, Buzz: 20 |
    | **NDVI** | Greenness index (0-1, satellite data) | Green: 0.70, Urban: 0.30 |
    | **Nightlight** | Light pollution/urban activity (0-100) | Cozy: 25, Buzz: 65 |
    | **Weather** | Temperature & precipitation | Cold: <5°C, Warm: >20°C |

    #### 3. Random Forest Classification
    The trained model predicts café suitability:
    - **Model type**: Multi-class Random Forest Classifier
    - **Classes**: 15 LAP Coffee locations
    - **Training data**: ~4,288 observations (Full year 2024: Winter, Spring, Summer, Autumn)
    - **Accuracy**: ~98% on test set
    - **Key learnings**:
      - "Cafés with high parks & low bars = quiet residential areas"
      - "Summer: higher NDVI (greenness), outdoor preferences"
      - "Winter: cozy indoor spots preferred"

    **How it predicts:**
    ```
    Input: [parks=8, bars=2, ndvi=0.45, temp=-1°C, ...]
    Output: {
        "LAP Coffee - Kastanienallee": 49% confidence,
        "LAP Coffee - Falckensteinstraße": 41% confidence,
        "LAP Coffee - Akazienstraße": 2% confidence,
        ...
    }
    ```

    #### 4. Location Filtering & Ranking
    - Filter cafés within 5km of your home (Bruchsaler Str. 10715)
    - Rank by model confidence (probability score)
    - Consider café rating and distance
    - Return top 3 recommendations

    #### Model Performance
    ✅ **Trained on complete full-year data** (Dec 2023 - Nov 2024, 3,648 observations)
    - **97.7% test accuracy** across all seasons
    - Robust predictions for all weather conditions (-5°C to 30°C)
    - Captures seasonal variations in NDVI, temperature, and café preferences
    - **Dynamic distance filtering**: Explores farther cafés when you're relaxed and weather is nice

    ---
    *Model feature importance: Parks (34.8%) + Bars (34.4%) + NDVI (11.3%) + Nightlight (10.0%) = 90% of prediction power*
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
            max_distance_km=10,  # Increased from 5km to include more options
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
                <h3 style="margin-top: 0;">☕ {rec['cafe_name']}</h3>
                <p style="margin: 5px 0;"><b>📍 Distance:</b> {rec['distance_km']} km from home</p>
                <p style="margin: 5px 0;"><b>⭐ Rating:</b> {rec['rating']}/5</p>
                <p style="margin: 5px 0;"><b>💪 Your Stress:</b> {stress:.1f}/100 (7-day avg)</p>
                <p style="margin: 5px 0;"><b>🌤️ Weather:</b> {rec['weather_temp']:.1f}°C{', ' + str(rec['weather_precip']) + 'mm rain' if rec['weather_precip'] > 0 else ''}</p>
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
                        - ⭐ {alt_rec['rating']}/5 rating
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
                    <p style="margin: 0; font-size: 16px;"><b>🎵 {i}. {song['title']}</b></p>
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

# Correlation
st.header("Correlations")
col1, col2 = st.columns(2)

with col1:
    fig_corr1 = px.scatter(
        df_filtered.dropna(subset=['sleep_hours', 'avg_stress']),
        x='sleep_hours', y='avg_stress',
        trendline='ols',
        title="Sleep vs Stress"
    )
    st.plotly_chart(fig_corr1, use_container_width=True)

with col2:
    fig_corr2 = px.scatter(
        df_filtered.dropna(subset=['net_battery', 'avg_stress']),
        x='net_battery', y='avg_stress',
        trendline='ols',
        title="Net Battery vs Stress"
    )
    st.plotly_chart(fig_corr2, use_container_width=True)

# Feature Importance
st.header("What Affects Your Stress?")
st.markdown("*Random Forest feature importance - which factors have the biggest impact on stress levels*")

# Prepare data for Random Forest
feature_cols = ['charged', 'drained', 'net_battery', 'resting_hr', 'sleep_hours',
                'deep_sleep_hours', 'rem_sleep_hours', 'steps']
available_features = [c for c in feature_cols if c in df_filtered.columns]

rf_data = df_filtered[available_features + ['avg_stress']].dropna()

if len(rf_data) > 20:
    X = rf_data[available_features]
    y = rf_data['avg_stress']

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X, y)

    # Get feature importance
    importance_df = pd.DataFrame({
        'Feature': available_features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=True)

    # Color by positive/negative correlation with stress
    colors = []
    for feat in importance_df['Feature']:
        corr = rf_data[feat].corr(rf_data['avg_stress'])
        colors.append('red' if corr > 0 else 'green')

    fig_importance = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker_color=colors
    ))
    fig_importance.update_layout(
        title="Feature Importance for Predicting Stress<br><sub>Green = reduces stress | Red = increases stress</sub>",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=400
    )
    st.plotly_chart(fig_importance, use_container_width=True)

    # Show top insights
    top_features = importance_df.tail(3)['Feature'].tolist()
    st.markdown("**Key Insights:**")
    for feat in reversed(top_features):
        corr = rf_data[feat].corr(rf_data['avg_stress'])
        direction = "increases" if corr > 0 else "decreases"
        st.markdown(f"- **{feat}**: Higher values {direction} stress")
else:
    st.warning("Not enough data for feature importance analysis (need at least 20 days)")

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
