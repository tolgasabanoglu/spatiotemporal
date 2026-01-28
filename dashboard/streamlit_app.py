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
