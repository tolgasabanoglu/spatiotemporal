"""
Coffee Recommendation System
Integrates Garmin health metrics with LAP Coffee mood-based recommendations
"""

import os
import sys
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from geopy.distance import geodesic
import joblib

# Path configuration
LAP_PROJECT_PATH = "/Users/tolgasabanoglu/Desktop/github/which-lap-coffee-should-i-visit"
MODEL_PATH = os.path.join(LAP_PROJECT_PATH, "prediction", "rfc_model.joblib")
ENCODER_PATH = os.path.join(LAP_PROJECT_PATH, "prediction", "label_encoder.joblib")
CAFE_DATA_PATH = os.path.join(LAP_PROJECT_PATH, "data", "processed", "lap_locations.gpkg")

# User home location
HOME_LOCATION = {
    "address": "Bruchsaler Strasse, 10715 Berlin",
    "lat": 52.4847,  # Approximate coordinates
    "lon": 13.3247
}

# Weather thresholds
COLD_THRESHOLD = 5  # °C - below this, prefer indoor "Cozy" spots
RAIN_THRESHOLD = 5  # mm - above this, trigger "Rainy Retreat"


def fetch_berlin_weather():
    """Fetch current Berlin weather from Open-Meteo API"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 52.52,  # Berlin center
        "longitude": 13.405,
        "current": "temperature_2m,precipitation,weather_code",
        "timezone": "Europe/Berlin"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        return {
            "temperature": data["current"]["temperature_2m"],
            "precipitation": data["current"]["precipitation"],
            "weather_code": data["current"]["weather_code"],
            "time": data["current"]["time"]
        }
    except Exception as e:
        print(f"Warning: Could not fetch weather data: {e}")
        return {
            "temperature": 10,  # Default fallback
            "precipitation": 0,
            "weather_code": 0,
            "time": datetime.now().isoformat()
        }


def health_to_mood_profile(stress, sleep_hours, net_battery, resting_hr, weather):
    """
    Map health metrics to LAP Coffee mood profile features

    Args:
        stress: Average stress level (0-100)
        sleep_hours: Sleep duration
        net_battery: Body battery net change
        resting_hr: Resting heart rate
        weather: Current weather dict

    Returns:
        dict: Feature dict for model prediction
    """
    temp = weather["temperature"]
    precip = weather["precipitation"]

    # Decision logic based on health metrics
    if stress > 60 and sleep_hours < 7:
        # High stress + poor sleep → Need relaxation
        if temp < COLD_THRESHOLD:
            mood = "cozy_indoor"
            parks = 8
            bars = 2
            nightlight = 25
            ndvi = 0.45
        else:
            mood = "green_nature"
            parks = 15
            bars = 3
            nightlight = 20
            ndvi = 0.70

    elif precip > RAIN_THRESHOLD or temp < COLD_THRESHOLD:
        # Bad weather → Cozy retreat
        mood = "rainy_retreat"
        parks = 8
        bars = 6
        nightlight = 35
        ndvi = 0.50

    elif stress < 40 and sleep_hours > 8 and net_battery > 0:
        # Well-rested + low stress → Social/energetic
        mood = "buzz_urban"
        parks = 3
        bars = 20
        nightlight = 65
        ndvi = 0.30

    elif net_battery < -20:
        # Drained battery → Recovery mode
        mood = "cozy_recharge"
        parks = 10
        bars = 4
        nightlight = 28
        ndvi = 0.55

    else:
        # Balanced state → Moderate activity
        mood = "balanced"
        parks = 8
        bars = 10
        nightlight = 40
        ndvi = 0.50

    # Create feature dict matching model's expected features
    features = {
        "parks_count_1km": parks,
        "open_bars_count_500m": bars,
        "nightlight": nightlight,
        "ndvi": ndvi,
        "lst_celsius_1km": temp,
        "temp_max": temp + 2,
        "temp_min": temp - 3,
        "precip_mm": precip
    }

    return features, mood


def load_cafe_locations():
    """Load LAP Coffee café locations from geopackage"""
    try:
        import geopandas as gpd
        gdf = gpd.read_file(CAFE_DATA_PATH)

        # Extract relevant columns
        cafes = []
        for idx, row in gdf.iterrows():
            # Create full name matching model's expected format: "LAP COFFEE_StreetName"
            address = row.get("address", "")
            # Extract street name from address (remove house number)
            # e.g., "Uhlandstraße 30, 10719 Berlin" -> "Uhlandstraße"
            street_part = address.split(",")[0].strip() if "," in address else address
            # Remove house numbers (digits and common suffixes like 'A', 'B')
            import re
            street_name = re.sub(r'\s+\d+[A-Z]?$', '', street_part).strip()
            # Handle special cases where street name is abbreviated (e.g., "Str." for "Straße")
            if not street_name:
                street_name = f"Location{idx+1}"

            full_name = f"LAP COFFEE_{street_name}"

            cafes.append({
                "name": full_name,
                "display_name": f"LAP Coffee - {street_name}",  # User-friendly name
                "address": address if address else "Berlin",
                "lat": row.geometry.y,
                "lon": row.geometry.x,
                "rating": row.get("rating", 4.5),
                "user_ratings_total": row.get("user_ratings_total", 0)
            })

        return pd.DataFrame(cafes)

    except Exception as e:
        print(f"Warning: Could not load café data: {e}")
        # Return empty dataframe with correct structure
        return pd.DataFrame(columns=["name", "address", "lat", "lon", "rating", "user_ratings_total"])


def calculate_distances(cafes_df, home_lat, home_lon):
    """Calculate distances from home to all cafés"""
    distances = []
    home_coords = (home_lat, home_lon)

    for idx, row in cafes_df.iterrows():
        cafe_coords = (row["lat"], row["lon"])
        dist = geodesic(home_coords, cafe_coords).kilometers
        distances.append(dist)

    cafes_df["distance_km"] = distances
    return cafes_df


def get_recommendations(stress, sleep_hours, net_battery, resting_hr,
                       max_distance_km=5, top_n=3):
    """
    Get coffee shop recommendations based on health metrics

    Args:
        stress: Average stress level
        sleep_hours: Sleep duration
        net_battery: Body battery net change
        resting_hr: Resting heart rate
        max_distance_km: Maximum distance filter
        top_n: Number of recommendations to return

    Returns:
        list: List of recommendation dicts
    """
    try:
        # 1. Fetch current weather
        weather = fetch_berlin_weather()

        # 2. Map health → mood features
        features, mood_name = health_to_mood_profile(
            stress, sleep_hours, net_battery, resting_hr, weather
        )

        # 3. Load café locations
        cafes_df = load_cafe_locations()
        if len(cafes_df) == 0:
            return [{
                "cafe_name": "No cafés available",
                "address": "N/A",
                "distance_km": 0,
                "confidence": 0,
                "mood": mood_name,
                "reason": "Could not load café data"
            }]

        # 4. Calculate distances
        cafes_df = calculate_distances(cafes_df, HOME_LOCATION["lat"], HOME_LOCATION["lon"])

        # 5. Filter by distance
        cafes_nearby = cafes_df[cafes_df["distance_km"] <= max_distance_km].copy()

        if len(cafes_nearby) == 0:
            # If no cafés within distance, return closest ones
            cafes_nearby = cafes_df.nsmallest(top_n, "distance_km")

        # 6. Load model and make predictions
        model = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)

        # Create feature array (must match training order!)
        feature_names = ["parks_count_1km", "open_bars_count_500m", "lst_celsius_1km",
                        "temp_max", "temp_min", "precip_mm", "ndvi", "nightlight"]
        X = pd.DataFrame([features])[feature_names]

        # Get predictions for all cafés
        pred_proba = model.predict_proba(X)[0]

        # Map probabilities to café names
        cafe_scores = {}
        for idx, cafe_class in enumerate(label_encoder.classes_):
            cafe_scores[cafe_class] = pred_proba[idx] * 100  # Convert to percentage

        # Add confidence scores to nearby cafés
        cafes_nearby["confidence"] = cafes_nearby["name"].map(cafe_scores).fillna(0)

        # 7. Rank by confidence, then by distance
        cafes_nearby = cafes_nearby.sort_values(
            by=["confidence", "distance_km"],
            ascending=[False, True]
        )

        # 8. Generate recommendations
        recommendations = []
        for idx, row in cafes_nearby.head(top_n).iterrows():
            reason = generate_reason(stress, sleep_hours, net_battery, mood_name, weather)

            recommendations.append({
                "cafe_name": row.get("display_name", row["name"]),  # Use friendly name
                "address": row["address"],
                "distance_km": round(row["distance_km"], 1),
                "confidence": round(row["confidence"], 1),
                "mood": mood_name,
                "reason": reason,
                "rating": row["rating"],
                "weather_temp": weather["temperature"],
                "weather_precip": weather["precipitation"]
            })

        return recommendations

    except Exception as e:
        print(f"Error generating recommendations: {e}")
        import traceback
        traceback.print_exc()
        return [{
            "cafe_name": "Error",
            "address": "N/A",
            "distance_km": 0,
            "confidence": 0,
            "mood": "unknown",
            "reason": f"Could not generate recommendations: {str(e)}"
        }]


def generate_reason(stress, sleep_hours, net_battery, mood, weather):
    """Generate human-readable explanation for recommendation"""
    temp = weather["temperature"]
    precip = weather["precipitation"]

    reasons = []

    # Health factors
    if stress > 60:
        reasons.append(f"high stress ({stress:.0f})")
    if sleep_hours < 7:
        reasons.append(f"limited sleep ({sleep_hours:.1f}hr)")
    if net_battery < -20:
        reasons.append(f"drained energy ({net_battery:.0f})")
    if stress < 40 and sleep_hours > 8:
        reasons.append("well-rested & relaxed")

    # Weather factors
    if temp < COLD_THRESHOLD:
        reasons.append(f"cold weather ({temp:.1f}°C)")
    if precip > RAIN_THRESHOLD:
        reasons.append(f"rainy ({precip:.1f}mm)")

    # Mood interpretation
    mood_descriptions = {
        "cozy_indoor": "cozy indoor spot for warmth",
        "green_nature": "peaceful green environment",
        "buzz_urban": "vibrant social atmosphere",
        "rainy_retreat": "sheltered retreat from weather",
        "cozy_recharge": "quiet recharge space",
        "balanced": "balanced atmosphere"
    }

    if reasons:
        reason_text = "Based on " + ", ".join(reasons)
    else:
        reason_text = "Based on your current metrics"

    mood_desc = mood_descriptions.get(mood, "suitable environment")
    return f"{reason_text}, we suggest a {mood_desc}"
