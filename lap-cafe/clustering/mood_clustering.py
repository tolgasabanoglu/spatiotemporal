"""
mood_clustering.py

Phase 1: Cluster café × date environmental features into mood types.

Each row (café + date) gets assigned a mood cluster label based on its
environmental conditions (NDVI, weather, nightlight, parks, bars).
Cluster centroids are saved for use at recommendation time.

Steps:
    1. Load env_features.csv
    2. Optionally scale features
    3. Run K-Means (k=4 or 5) — tune with elbow/silhouette
    4. Assign cluster labels + human-readable mood names
    5. Save labelled dataset and cluster centroids

Usage:
    python lap-cafe/clustering/mood_clustering.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

DATA_PATH = Path("data/lap-cafe/env_features_2024.csv")  # update to env_features.csv after re-fetching
OUTPUT_LABELLED = Path("data/lap-cafe/env_features_labelled.csv")
OUTPUT_CENTROIDS = Path("data/lap-cafe/cluster_centroids.csv")
MODEL_PATH = Path("lap-cafe/clustering/kmeans_model.joblib")
SCALER_PATH = Path("lap-cafe/clustering/scaler.joblib")

# Features used for clustering — dynamic features only to avoid static overfitting
CLUSTER_FEATURES = ["ndvi", "nightlight", "temp_max", "temp_min", "precip_mm"]

# Mood names derived from k=4 cluster analysis
# Cluster 0: warm, dry, low NDVI, low nightlight  → sunny_urban
# Cluster 1: cold, dark, high nightlight           → winter_cozy
# Cluster 2: mild, rainy                           → rainy_day
# Cluster 3: warm, high NDVI (0.34), low urban     → summer_green
MOOD_NAMES = {
    0: "sunny_urban",
    1: "winter_cozy",
    2: "rainy_day",
    3: "summer_green",
}


def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 8)):
    """Plot elbow curve and silhouette scores to guide k selection."""
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(k_range, inertias, marker="o")
    ax1.set_title("Elbow Curve")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")
    ax2.plot(k_range, silhouettes, marker="o", color="orange")
    ax2.set_title("Silhouette Score")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Score")
    plt.tight_layout()
    plt.savefig("data/lap-cafe/kmeans_selection.png")
    plt.show()
    print(f"Best k by silhouette: {k_range[np.argmax(silhouettes)]}")


def train(k: int = 4):
    df = pd.read_csv(DATA_PATH)
    X = df[CLUSTER_FEATURES].dropna()
    df = df.loc[X.index].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["mood_cluster"] = km.fit_predict(X_scaled)
    df["mood"] = df["mood_cluster"].map(MOOD_NAMES)

    # Save centroids in original scale for interpretability
    centroids = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=CLUSTER_FEATURES
    )
    centroids.index.name = "cluster"
    centroids["mood"] = centroids.index.map(MOOD_NAMES)

    df.to_csv(OUTPUT_LABELLED, index=False)
    centroids.to_csv(OUTPUT_CENTROIDS)
    joblib.dump(km, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"Clusters saved to {OUTPUT_LABELLED}")
    print(f"\nCluster centroids:\n{centroids.to_string()}")
    print(f"\nCluster distribution:\n{df['mood'].value_counts()}")


if __name__ == "__main__":
    train(k=4)
