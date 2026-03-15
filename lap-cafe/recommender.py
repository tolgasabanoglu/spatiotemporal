"""
recommender.py

Given today's predicted mood and current environmental conditions,
rank LAP Coffee locations by how well they match the mood cluster.

Pipeline:
    1. Predict mood from today's Garmin biometrics (mood_classifier.py)
    2. Load cluster centroids (from mood_clustering.py)
    3. Fetch today's env conditions per café (or use latest available)
    4. Score each café: distance to predicted cluster centroid + proximity
    5. Return ranked recommendations
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from math import radians, cos, sin, asin, sqrt

CENTROIDS_PATH = Path("data/lap-cafe/cluster_centroids.csv")
LOCATIONS_PATH = Path("data/lap-cafe/locations.csv")
SCALER_PATH = Path("lap-cafe/clustering/scaler.joblib")

CLUSTER_FEATURES = ["ndvi", "nightlight", "temp_max", "temp_min", "precip_mm"]

# Home coordinates (Bruchsaler Str., Berlin)
HOME_LAT = 52.4937
HOME_LON = 13.3418
MAX_DISTANCE_KM = None  # No distance limit — rank all cafés


def haversine(lat1, lon1, lat2, lon2) -> float:
    """Distance in km between two coordinates."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * asin(sqrt(a)) * 6371


def score_cafes(mood: str, today_env: pd.DataFrame) -> pd.DataFrame:
    """
    Score cafés by how closely today's conditions match the predicted mood centroid.

    Args:
        mood:      predicted mood label (e.g., 'green_calm')
        today_env: DataFrame with columns [name, lat, lon] + CLUSTER_FEATURES

    Returns:
        DataFrame with columns [name, address, distance_km, env_score, final_score]
        sorted by final_score descending
    """
    centroids = pd.read_csv(CENTROIDS_PATH, index_col="cluster")
    scaler = joblib.load(SCALER_PATH)

    target_centroid = centroids[centroids["mood"] == mood][CLUSTER_FEATURES].values
    if len(target_centroid) == 0:
        raise ValueError(f"Unknown mood: {mood}")
    target_centroid = target_centroid[0]

    results = []
    for _, cafe in today_env.iterrows():
        cafe_env = cafe[CLUSTER_FEATURES].values.astype(float)

        # Euclidean distance in scaled space (lower = better match)
        cafe_scaled = scaler.transform([cafe_env])[0]
        target_scaled = scaler.transform([target_centroid])[0]
        env_distance = np.linalg.norm(cafe_scaled - target_scaled)
        env_score = max(0, 100 - env_distance * 20)  # normalise to 0-100

        # Proximity score
        distance_km = haversine(HOME_LAT, HOME_LON, cafe["lat"], cafe["lon"])
        proximity_score = max(0, 100 - distance_km * 5)  # ~20km = 0 score

        # Final: 60% environment match, 40% proximity
        final_score = env_score * 0.6 + proximity_score * 0.4

        results.append({
            "name": cafe["name"],
            "address": cafe.get("address", ""),
            "distance_km": round(distance_km, 1),
            "env_score": round(env_score, 1),
            "proximity_score": round(proximity_score, 1),
            "final_score": round(final_score, 1),
        })

    return pd.DataFrame(results).sort_values("final_score", ascending=False)


def recommend(mood: str, today_env: pd.DataFrame, top_n: int = 3) -> list[dict]:
    """Return top N café recommendations for the given mood."""
    ranked = score_cafes(mood, today_env)
    return ranked.head(top_n).to_dict(orient="records")
