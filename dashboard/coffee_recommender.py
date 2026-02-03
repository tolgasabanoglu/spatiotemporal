"""
Coffee Recommendation System
Integrates Garmin health metrics with LAP Coffee mood-based recommendations

## How It Works:

1. **Health-to-Mood Mapping**: Converts Garmin biometrics (stress, sleep, body battery)
   into environmental mood profiles (e.g., "green_nature", "buzz_urban")

2. **Mood-to-Features Translation**: Each mood profile maps to environmental characteristics:
   - parks_count_1km: Number of parks within 1km (0-20)
   - open_bars_count_500m: Bars within 500m (0-25)
   - ndvi: Greenness index from satellite (0-1, higher = greener)
   - nightlight: Light pollution/urban activity (0-100)
   - Weather: Temperature, precipitation from Open-Meteo API

3. **Random Forest Classifier (RFC) Prediction**:
   - Trained model: `/which-lap-coffee-should-i-visit/prediction/rfc_model.joblib`
   - Model type: Multi-class RandomForestClassifier (15 classes = 15 cafés)
   - Training data: ~4,288 observations from full year 2024 (all seasons)
     * Winter (Dec 2023 - Feb 2024): ~720 observations
     * Spring (Mar - May 2024): ~800 observations
     * Summer (Jun - Aug 2024): ~1,312 observations
     * Autumn (Sep - Nov 2024): ~1,456 observations
   - Accuracy: ~98% on test set (improves with full-year data)
   - How it works:
     * Input: Feature vector [parks=15, bars=3, ndvi=0.7, temp=10, ...]
     * Output: Probability distribution over 16 LAP Coffee locations
     * Example: {Kastanienallee: 0.49, Falckensteinstraße: 0.41, ...}
   - The model learned patterns across all seasons:
     * "Cafés with high parks & low bars are in quiet residential areas"
     * "Summer: higher NDVI, outdoor preferences"
     * "Winter: cozy indoor spots preferred"

4. **Location Filtering**: Filters cafés by distance from home (Bruchsaler Str. 10715)
   - Default range: 10km radius (increased for better diversity)

5. **Ranking**: Balanced scoring combining:
   - Model confidence (probability from RFC)
   - Distance penalty (closer is better, but not overwhelming)
   - Slight randomization for variety
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
PARKS_DATA_PATH = os.path.join(LAP_PROJECT_PATH, "data", "processed", "lap_locations_with_park_counts.gpkg")
BARS_DATA_PATH = os.path.join(LAP_PROJECT_PATH, "data", "processed", "lap_locations_with_open_bars.gpkg")

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

    elif precip > RAIN_THRESHOLD:
        # Rainy weather → Sheltered retreat
        mood = "rainy_retreat"
        parks = 8
        bars = 6
        nightlight = 35
        ndvi = 0.50

    elif temp < COLD_THRESHOLD:
        # Just cold (no rain) → Cozy indoor
        mood = "cozy_indoor"
        parks = 8
        bars = 2
        nightlight = 25
        ndvi = 0.45

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
    """Load LAP Coffee café locations with environmental features from geopackages"""
    try:
        import geopandas as gpd
        import re

        # Load base café data
        gdf = gpd.read_file(CAFE_DATA_PATH)

        # Load parks and bars features
        gdf_parks = gpd.read_file(PARKS_DATA_PATH)
        gdf_bars = gpd.read_file(BARS_DATA_PATH)

        # Extract relevant columns
        cafes = []
        for idx, row in gdf.iterrows():
            # Create full name matching model's expected format: "LAP COFFEE_StreetName"
            address = row.get("address", "")
            # Extract street name from address (remove house number)
            street_part = address.split(",")[0].strip() if "," in address else address
            # Remove house numbers (digits and common suffixes like 'A', 'B')
            street_name = re.sub(r'\s+\d+[A-Z]?$', '', street_part).strip()
            if not street_name:
                street_name = f"Location{idx+1}"

            full_name = f"LAP COFFEE_{street_name}"

            # Get parks count for this café (match by address)
            parks_row = gdf_parks[gdf_parks['address'] == address]
            parks_count = parks_row['parks_count_1km'].iloc[0] if len(parks_row) > 0 else 0

            # Get bars count for this café (match by address)
            bars_row = gdf_bars[gdf_bars['address'] == address]
            bars_count = bars_row['open_bars_count_500m'].iloc[0] if len(bars_row) > 0 else 0

            cafes.append({
                "name": full_name,
                "display_name": f"LAP Coffee - {street_name}",
                "address": address if address else "Berlin",
                "lat": row.geometry.y,
                "lon": row.geometry.x,
                "rating": row.get("rating", 4.5),
                "user_ratings_total": row.get("user_ratings_total", 0),
                "parks_count_1km": int(parks_count),
                "open_bars_count_500m": int(bars_count)
            })

        return pd.DataFrame(cafes)

    except Exception as e:
        print(f"Warning: Could not load café data: {e}")
        import traceback
        traceback.print_exc()
        # Return empty dataframe with correct structure
        return pd.DataFrame(columns=["name", "address", "lat", "lon", "rating", "user_ratings_total",
                                     "parks_count_1km", "open_bars_count_500m"])


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
                       max_distance_km=10, top_n=3):
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

        # 7. Rank with balanced scoring (confidence + distance + diversity)
        # Normalize confidence (0-100) and distance (inverse, closer = higher score)
        max_dist = cafes_nearby["distance_km"].max()
        cafes_nearby["distance_score"] = (1 - cafes_nearby["distance_km"] / max_dist) * 100

        # Balanced score: 60% confidence, 40% proximity (favors model, but allows variety)
        cafes_nearby["balanced_score"] = (
            cafes_nearby["confidence"] * 0.6 +
            cafes_nearby["distance_score"] * 0.4
        )

        # Add small random factor for diversity (±2 points)
        # Use daily seed for consistency - same recommendations throughout the day
        date_seed = int(datetime.now().date().toordinal())
        np.random.seed(date_seed)
        cafes_nearby["random_factor"] = np.random.uniform(-2, 2, len(cafes_nearby))
        cafes_nearby["final_score"] = cafes_nearby["balanced_score"] + cafes_nearby["random_factor"]

        # Sort by final score
        cafes_nearby = cafes_nearby.sort_values("final_score", ascending=False)

        # 8. Generate recommendations
        recommendations = []
        for idx, row in cafes_nearby.head(top_n).iterrows():
            # Get actual café features for accurate reasoning
            actual_cafe_features = {
                "parks_count_1km": row.get("parks_count_1km"),
                "open_bars_count_500m": row.get("open_bars_count_500m")
            }

            reason = generate_reason(stress, sleep_hours, net_battery, mood_name,
                                    weather, features, actual_cafe_features)

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


def generate_reason(stress, sleep_hours, net_battery, mood, weather, desired_features, actual_cafe_features):
    """Generate human-readable explanation for recommendation

    Args:
        stress: Stress level
        sleep_hours: Sleep duration
        net_battery: Body battery net change
        mood: Mood profile name
        weather: Weather dict
        desired_features: Features from mood profile (what you're looking for)
        actual_cafe_features: Actual environmental features of the recommended café
    """
    temp = weather["temperature"]
    precip = weather["precipitation"]

    # Health state analysis
    health_state = []
    if stress > 60:
        health_state.append(f"high stress level ({stress:.0f}/100)")
    elif stress < 40:
        health_state.append(f"low stress ({stress:.0f}/100)")

    if sleep_hours < 7:
        health_state.append(f"limited sleep ({sleep_hours:.1f}hr)")
    elif sleep_hours > 8:
        health_state.append(f"well-rested ({sleep_hours:.1f}hr)")

    if net_battery < -15:
        health_state.append(f"drained energy (battery: {net_battery:.0f})")
    elif net_battery > 10:
        health_state.append(f"energized (battery: +{net_battery:.0f})")

    # Weather context
    weather_context = []
    if temp < COLD_THRESHOLD:
        weather_context.append(f"cold ({temp:.1f}°C)")
    elif temp > 20:
        weather_context.append(f"warm ({temp:.1f}°C)")
    if precip > RAIN_THRESHOLD:
        weather_context.append(f"rainy ({precip:.1f}mm)")

    # Get actual café features (with fallbacks)
    actual_parks = actual_cafe_features.get("parks_count_1km")
    actual_bars = actual_cafe_features.get("open_bars_count_500m")

    # Mood profile explanation
    mood_profiles = {
        "cozy_indoor": {"profile": "Cozy Indoor", "reasoning": "Need for warmth and comfort"},
        "green_nature": {"profile": "Green Nature", "reasoning": "Seeking natural, restorative environment"},
        "buzz_urban": {"profile": "Buzz Urban", "reasoning": "Ready for social, energetic atmosphere"},
        "rainy_retreat": {"profile": "Rainy Retreat", "reasoning": "Sheltered comfort during bad weather"},
        "cozy_recharge": {"profile": "Cozy Recharge", "reasoning": "Recovery and energy restoration"},
        "balanced": {"profile": "Balanced", "reasoning": "Moderate, all-around suitable environment"}
    }

    mood_info = mood_profiles.get(mood, {"profile": "Standard", "reasoning": "General recommendation"})

    # Build comprehensive reason
    parts = []

    # Health context
    if health_state:
        parts.append("Your " + " + ".join(health_state))

    # Weather context
    if weather_context:
        parts.append("today's " + " & ".join(weather_context) + " weather")

    # Mood profile
    parts.append(f'suggest a "{mood_info["profile"]}" mood ({mood_info["reasoning"]})')

    reason = ". ".join(parts).capitalize()

    # Add ACTUAL café characteristics (dynamic based on real data)
    if actual_parks is not None and actual_bars is not None:
        # Build detailed feature description with actual numbers
        cafe_details = []

        # Parks description
        cafe_details.append(f"{int(actual_parks)} parks within 1km")

        # Bars description
        cafe_details.append(f"{int(actual_bars)} bars nearby")

        cafe_text = ", ".join(cafe_details)

        # Add environmental conditions being matched
        env_details = []

        # NDVI (greenness from satellite)
        ndvi_val = desired_features.get('ndvi', 0)
        env_details.append(f"NDVI: {ndvi_val:.2f}")

        # Nightlight intensity
        nightlight_val = desired_features.get('nightlight', 0)
        env_details.append(f"nightlight: {nightlight_val:.0f}")

        env_text = ", ".join(env_details)

        full_reason = f"{reason}. Location has: {cafe_text}. Matching conditions: {env_text}."
    else:
        # Fallback if actual features not available
        full_reason = f"{reason}. This café matches your preferred environment."

    return full_reason
