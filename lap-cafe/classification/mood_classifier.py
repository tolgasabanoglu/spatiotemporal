"""
mood_classifier.py

Phase 2: Train a classifier to predict mood cluster from Garmin biometrics.

Input:  Garmin daily metrics (stress, sleep, body battery, HR) — lagged
Target: mood cluster label from Phase 1 clustering

The idea: given how you've been feeling over the past few days,
predict which environment type you currently need.

Features (lagged):
    - stress_lag1, stress_roll3, stress_roll7
    - sleep_lag1, sleep_roll3
    - rem_sleep_lag1
    - body_battery_lag1, body_battery_roll3
    - resting_hr_lag1

Usage:
    python lap-cafe/classification/mood_classifier.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

GARMIN_PATH = Path("data/processed/garmin_daily.csv")       # adjust to actual path
ENV_LABELLED_PATH = Path("data/lap-cafe/env_features_labelled.csv")
MODEL_PATH = Path("lap-cafe/classification/mood_classifier.joblib")
ENCODER_PATH = Path("lap-cafe/classification/mood_encoder.joblib")

# Biometric features — lagged to capture accumulated state
BIOMETRIC_FEATURES = [
    "stress_lag1", "stress_roll3", "stress_roll7",
    "sleep_lag1", "sleep_roll3",
    "rem_sleep_lag1",
    "body_battery_lag1", "body_battery_roll3",
    "resting_hr_lag1",
]

TARGET = "mood"


def build_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged and rolling features to Garmin daily dataframe."""
    df = df.sort_values("date").copy()

    df["stress_lag1"] = df["avg_stress"].shift(1)
    df["stress_roll3"] = df["avg_stress"].shift(1).rolling(3).mean()
    df["stress_roll7"] = df["avg_stress"].shift(1).rolling(7).mean()

    df["sleep_lag1"] = df["total_sleep_hours"].shift(1)
    df["sleep_roll3"] = df["total_sleep_hours"].shift(1).rolling(3).mean()
    df["rem_sleep_lag1"] = df["rem_sleep_hours"].shift(1)

    df["body_battery_lag1"] = df["net_body_battery"].shift(1)
    df["body_battery_roll3"] = df["net_body_battery"].shift(1).rolling(3).mean()

    df["resting_hr_lag1"] = df["resting_hr"].shift(1)

    return df.dropna(subset=BIOMETRIC_FEATURES)


def align_with_mood_labels(garmin_df: pd.DataFrame, env_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align Garmin data with mood labels from env clustering.
    Mood label for a given date = most common cluster across all cafés that day.
    """
    daily_mood = (
        env_df.groupby("date")["mood"]
        .agg(lambda x: x.mode()[0])  # most common mood across cafés that day
        .reset_index()
    )
    return garmin_df.merge(daily_mood, on="date", how="inner")


def train():
    garmin_df = pd.read_csv(GARMIN_PATH, parse_dates=["date"])
    env_df = pd.read_csv(ENV_LABELLED_PATH, parse_dates=["date"])

    garmin_df = build_lagged_features(garmin_df)
    df = align_with_mood_labels(garmin_df, env_df)

    print(f"Training dataset: {len(df)} aligned days")
    print(f"Mood distribution:\n{df[TARGET].value_counts()}\n")

    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])
    X = df[BIOMETRIC_FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Compare Logistic Regression vs Random Forest (small dataset → keep it simple)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        print(f"{name} — CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Use Random Forest as final model
    clf = models["Random Forest"]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\nTest accuracy: {clf.score(X_test, y_test):.3f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"Model saved to {MODEL_PATH}")


def predict(garmin_today: dict) -> str:
    """
    Predict mood from today's (lagged) biometric features.

    Args:
        garmin_today: dict with keys matching BIOMETRIC_FEATURES

    Returns:
        mood label string (e.g., 'green_calm')
    """
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)

    X = pd.DataFrame([garmin_today])[BIOMETRIC_FEATURES]
    mood_idx = clf.predict(X)[0]
    return le.inverse_transform([mood_idx])[0]


if __name__ == "__main__":
    train()
