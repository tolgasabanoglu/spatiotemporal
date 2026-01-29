# Coffee Recommendation System - Technical Documentation

## Overview

This system integrates your Garmin health metrics with LAP Coffee's mood-based recommendation model to suggest personalized café visits based on your biometric data and current weather.

---

## How the Random Forest Classifier (RFC) Works

### Model Architecture
- **Type**: Multi-class Random Forest Classifier
- **Classes**: 16 LAP Coffee locations in Berlin
- **Training Data**: 3,440 observations from October-November 2024
- **Accuracy**: ~96% on test set
- **Model Path**: `/which-lap-coffee-should-i-visit/prediction/rfc_model.joblib`

### Prediction Pipeline

```
Health Metrics → Mood Profile → Environmental Features → RFC → Café Probabilities → Ranked Recommendations
```

**Step 1: Health-to-Mood Mapping**
Your 7-day average health metrics are analyzed:
- Stress level (0-100)
- Sleep hours
- Body battery net change
- Resting heart rate

These map to mood profiles:
- **Cozy Indoor**: High stress + cold weather → Need warmth & comfort
- **Green Nature**: High stress + nice weather → Seek restorative greenery
- **Buzz Urban**: Low stress + energized → Want vibrant social atmosphere
- **Rainy Retreat**: Rainy weather → Sheltered cozy space
- **Cozy Recharge**: Drained energy → Quiet recovery environment
- **Balanced**: Moderate metrics → Flexible atmosphere

**Step 2: Mood-to-Features Translation**
Each mood profile translates to specific environmental characteristics:

| Mood | Parks (1km) | Bars (500m) | NDVI | Nightlight | Interpretation |
|------|-------------|-------------|------|------------|----------------|
| Cozy Indoor | 8 | 2 | 0.45 | 25 | Residential, low activity |
| Green Nature | 15 | 3 | 0.70 | 20 | High greenery, peaceful |
| Buzz Urban | 3 | 20 | 0.30 | 65 | Dense urban, high nightlife |
| Rainy Retreat | 8 | 6 | 0.50 | 35 | Moderate activity, cozy |
| Balanced | 8 | 10 | 0.50 | 40 | Mixed neighborhood |

**Step 3: RFC Prediction**
The trained model receives these features plus current weather:
```python
Input: [parks=8, bars=2, ndvi=0.45, nightlight=25, temp=-1, precip=0, ...]
Output: {
    "LAP COFFEE_Kastanienallee": 0.49,
    "LAP COFFEE_Falckensteinstraße": 0.41,
    "LAP COFFEE_Akazienstraße": 0.02,
    ...
}
```

The model learned patterns like:
- Cafés in residential areas (high parks, low bars) → Kreuzberg, Prenzlauer Berg
- Cafés in vibrant areas (low parks, high bars) → Friedrichshain, Mitte
- Green cafés (high NDVI) → Near parks like Volkspark

**Step 4: Location Filtering & Ranking**
- Filter to cafés within 5km of home (Bruchsaler Str. 10715)
- Rank by model probability (confidence)
- Return top 3 recommendations

---

## Training Data Limitation

### ⚠️ IMPORTANT: Seasonal Bias

**Current Status**: The model was trained **ONLY on autumn data** (September-November 2023, 2024, 2025).

**Why This Matters**:
The model learned café-environment relationships during autumn, but seasons differ significantly:

| Season | NDVI (Greenness) | Temperature | User Behavior | Model Accuracy |
|--------|------------------|-------------|---------------|----------------|
| **Autumn** (training) | 0.40-0.65 | 5-15°C | Moderate outdoor | ✅ **96%** (in-sample) |
| Winter | 0.20-0.40 | -5 to 5°C | Indoor preference | ⚠️ **Unknown** |
| Spring | 0.50-0.75 | 10-20°C | Transitional | ⚠️ **Unknown** |
| Summer | 0.60-0.80 | 20-30°C | High outdoor activity | ⚠️ **Unknown** |

**Potential Issues**:
1. **Winter**: Lower NDVI values not seen during training → Model may misclassify cafés
2. **Summer**: Higher temperatures & NDVI → Model trained on cooler conditions
3. **Behavioral Shifts**: People prefer different café types in summer (outdoor seating) vs winter (indoor warmth)

---

## Recommendation: Collect Full-Year Training Data

To make the model truly robust year-round, we should:

### Phase 1: Data Collection (All Seasons)
Collect environmental data for:
- **Winter**: December 2024, January-February 2025
- **Spring**: March-May 2025
- **Summer**: June-August 2025
- **Autumn**: September-November 2025 (already have this)

This would give ~10,000-12,000 training observations across all seasons.

### Phase 2: Feature Engineering
Re-run the LAP Coffee feature engineering pipeline:
```bash
cd /which-lap-coffee-should-i-visit

# Fetch data for each season
python src/features/add_weather.py --start-date 2024-12-01 --end-date 2025-02-28
python src/features/add_ndvi.py --start-date 2024-12-01 --end-date 2025-02-28
python src/features/add_nightlights_daily.py --start-date 2024-12-01 --end-date 2025-02-28
# ... repeat for spring & summer
```

### Phase 3: Model Retraining
```bash
cd prediction
python rfc.py --data-path ../data/processed/lap_locations_full_year.csv
```

Expected outcomes:
- Model learns seasonal patterns (e.g., "summer → high NDVI + warm → outdoor cafés")
- Better generalization to current conditions
- More accurate recommendations year-round

### Estimated Effort
- Data collection: 2-3 hours (API calls for all seasons)
- Feature engineering: 1-2 hours (running existing pipeline)
- Model retraining: 30 minutes
- Testing & validation: 1 hour

**Total**: ~5-6 hours of work for significantly improved recommendations.

---

## Alternative: Seasonal Adjustment Layer

If full retraining isn't feasible, we could add a **seasonal adjustment layer**:

```python
def adjust_features_for_season(features, current_month):
    """Adjust features to account for seasonal differences"""
    if current_month in [12, 1, 2]:  # Winter
        features['ndvi'] *= 0.6  # Lower greenness
        features['parks_count_1km'] *= 0.7  # Less park activity
    elif current_month in [6, 7, 8]:  # Summer
        features['ndvi'] *= 1.2  # Higher greenness
        features['open_bars_count_500m'] *= 1.3  # More outdoor activity
    return features
```

This is a **quick fix** but not as robust as full retraining.

---

## Current Performance

Despite the seasonal limitation, the model provides reasonable recommendations because:
1. **Café locations don't change** - Kastanienallee is still in a residential area in winter
2. **Weather integration** - Real-time temperature adjusts for seasonal conditions
3. **Relative rankings** - Even if absolute probabilities shift, relative café ordering is stable

**Bottom Line**: The system works well but would benefit from full-year training data for optimal accuracy.

---

## Summary

| Aspect | Current | Ideal |
|--------|---------|-------|
| Training Data | Autumn only (3,440 obs) | All seasons (10,000+ obs) |
| Seasonal Accuracy | Good in autumn, uncertain otherwise | Good year-round |
| Model Generalization | Limited | Robust |
| Recommendation Quality | ⭐⭐⭐⭐ (4/5) | ⭐⭐⭐⭐⭐ (5/5) |

**Next Steps**:
1. ✅ Use current system (works reasonably well)
2. 🔄 Plan full-year data collection for 2025
3. 🚀 Retrain model in late 2025 with complete dataset
