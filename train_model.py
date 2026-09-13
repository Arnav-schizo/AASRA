import pandas as pd
import numpy as np
import sqlite3
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("=" * 70)
print("AASTRA — HAZARD INTELLIGENCE")
print("=" * 70)

# ============================================================
# LOAD
# ============================================================

rain = pd.read_csv("hazard_rainfall_features.csv")
river = pd.read_csv("hazard_river_features.csv")
flood = pd.read_csv("hazard_flood_features.csv")
landslide = pd.read_csv("hazard_landslide_features.csv")

conn = sqlite3.connect("disaster.db")

historical = pd.read_sql(
    "SELECT * FROM historical_disasters",
    conn
)

conn.close()

print("\nDATA LOADED")
print("Rainfall:", len(rain))
print("River:", len(river))
print("Flood:", len(flood))
print("Landslide:", len(landslide))
print("Historical disasters:", len(historical))


# ============================================================
# DISTRICT CLEANING
# ============================================================

def clean_district(x):

    if pd.isna(x):
        return np.nan

    x = str(x).strip().lower()

    x = x.replace("uttar kashi", "uttarkashi")
    x = x.replace("uttar-kashi", "uttarkashi")

    return x


# ============================================================
# RAINFALL BASE
# ============================================================

rain["date"] = pd.to_datetime(
    rain["date"],
    errors="coerce"
)

rain["year"] = rain["date"].dt.year

rain["district_model"] = (
    rain["district"]
    .apply(clean_district)
)

rain = rain.dropna(
    subset=["date", "district_model"]
).copy()


# ============================================================
# FLOOD EVENTS
# ============================================================

flood["start_date"] = pd.to_datetime(
    flood["start_date"],
    errors="coerce"
)

flood["end_date"] = pd.to_datetime(
    flood["end_date"],
    errors="coerce"
)

flood["end_date"] = (
    flood["end_date"]
    .fillna(flood["start_date"])
)

flood["district_text"] = (
    flood["districts"]
    .astype("string")
    .str.lower()
)

districts = [
    "almora",
    "bageshwar",
    "chamoli",
    "champawat",
    "dehradun",
    "haridwar",
    "nainital",
    "pauri garhwal",
    "pithoragarh",
    "rudraprayag",
    "tehri garhwal",
    "udham singh nagar",
    "uttarkashi"
]

flood_events = []

for _, row in flood.iterrows():

    text = row["district_text"]

    if pd.isna(text):
        continue

    for district in districts:

        if district in str(text):

            flood_events.append({
                "district_model": district,
                "start_date": row["start_date"],
                "end_date": row["end_date"]
            })

flood_events = pd.DataFrame(
    flood_events
)

print("\nFlood district-event matches:", len(flood_events))


# ============================================================
# LANDSLIDE EVENTS
# ============================================================

landslide["district_model"] = (
    landslide["landslide_district_standard"]
    .apply(clean_district)
)

landslide["landslide_year"] = pd.to_numeric(
    landslide["landslide_history_year"],
    errors="coerce"
)

landslide = landslide.dropna(
    subset=[
        "district_model",
        "landslide_year"
    ]
).copy()

landslide["landslide_year"] = (
    landslide["landslide_year"]
    .astype(int)
)

# District/year historical landslide count
landslide_history = (
    landslide
    .groupby(
        [
            "district_model",
            "landslide_year"
        ]
    )
    .size()
    .reset_index(
        name="previous_landslide_count"
    )
)


# ============================================================
# HISTORICAL DISASTERS
# ============================================================

historical["historical_date"] = pd.to_datetime(
    historical["date_ymd"],
    errors="coerce"
)

historical["historical_year"] = (
    historical["historical_date"]
    .dt.year
)

if "level_1" in historical.columns:

    historical["district_raw"] = (
        historical["level_1"]
    )

elif "level_2" in historical.columns:

    historical["district_raw"] = (
        historical["level_2"]
    )

else:

    historical["district_raw"] = (
        historical["location"]
    )

historical["district_model"] = (
    historical["district_raw"]
    .apply(clean_district)
)

historical = historical.dropna(
    subset=[
        "district_model",
        "historical_year"
    ]
).copy()

historical["historical_year"] = (
    historical["historical_year"]
    .astype(int)
)

historical_year = (
    historical
    .groupby(
        [
            "district_model",
            "historical_year"
        ]
    )
    .size()
    .reset_index(
        name="historical_disaster_count"
    )
)

# Shift by one year so historical information is genuinely prior
historical_year["year"] = (
    historical_year["historical_year"] + 1
)

historical_year = historical_year[
    [
        "district_model",
        "year",
        "historical_disaster_count"
    ]
]


# ============================================================
# CREATE HAZARD LABEL
# ============================================================

def get_hazard(row):

    district = row["district_model"]
    date = row["date"]
    year = row["year"]

    # --------------------------------------------------------
    # Flood: rainfall date falls inside a recorded flood event
    # --------------------------------------------------------

    if len(flood_events) > 0:

        matching_flood = flood_events[
            (flood_events["district_model"] == district)
            &
            (flood_events["start_date"] <= date)
            &
            (flood_events["end_date"] >= date)
        ]

        if len(matching_flood) > 0:

            return "FLOOD"

    # --------------------------------------------------------
    # Landslide: historical landslide occurred in district/year
    # --------------------------------------------------------

    matching_landslide = landslide[
        (landslide["district_model"] == district)
        &
        (landslide["landslide_year"] == year)
    ]

    if len(matching_landslide) > 0:

        return "LANDSLIDE"

    return "NORMAL"


print("\nCreating hazard labels...")

rain["hazard_type"] = rain.apply(
    get_hazard,
    axis=1
)

print("\n" + "=" * 70)
print("🔥 HAZARD LABEL DISTRIBUTION")
print("=" * 70)

print(
    rain["hazard_type"]
    .value_counts()
    .to_string()
)


# ============================================================
# ADD PREVIOUS HISTORICAL DISASTER CONTEXT
# ============================================================

rain = rain.merge(
    historical_year,
    on=[
        "district_model",
        "year"
    ],
    how="left"
)

rain["historical_disaster_count"] = (
    rain["historical_disaster_count"]
    .fillna(0)
)


# ============================================================
# ADD PREVIOUS LANDSLIDE COUNT
# ============================================================

landslide_lag = (
    landslide_history
    .copy()
)

landslide_lag["year"] = (
    landslide_lag["landslide_year"] + 1
)

landslide_lag = landslide_lag[
    [
        "district_model",
        "year",
        "previous_landslide_count"
    ]
]

rain = rain.merge(
    landslide_lag,
    on=[
        "district_model",
        "year"
    ],
    how="left"
)

rain["previous_landslide_count"] = (
    rain["previous_landslide_count"]
    .fillna(0)
)


# ============================================================
# RIVER FEATURES
# ============================================================

river["date"] = pd.to_datetime(
    river["date"],
    errors="coerce"
)

river["district_model"] = (
    river["district"]
    .apply(clean_district)
)

river_numeric = [
    "mean_discharge_m3_sec",
    "max_discharge_m3_sec",
    "min_discharge_m3_sec",
    "discharge_range",
    "mean_discharge_7day",
    "mean_discharge_14day",
    "mean_discharge_30day",
    "max_discharge_7day",
    "max_discharge_14day",
    "max_discharge_30day",
    "discharge_std_7day",
    "discharge_std_14day",
    "discharge_std_30day",
    "discharge_change",
    "discharge_change_pct"
]

river_numeric = [
    c for c in river_numeric
    if c in river.columns
]

# District/year river context
river["year"] = river["date"].dt.year

annual_river = (
    river
    .groupby(
        [
            "district_model",
            "year"
        ]
    )[river_numeric]
    .mean()
    .reset_index()
)

rain = rain.merge(
    annual_river,
    on=[
        "district_model",
        "year"
    ],
    how="left"
)


# ============================================================
# FEATURES
# ============================================================

rain_features = [
    "rainfall_mean",
    "rainfall_min",
    "rainfall_max",
    "rainfall_std",
    "rainfall_sum",
    "hourly_rainfall_max",

    "rainfall_mean_3day",
    "rainfall_mean_7day",
    "rainfall_mean_14day",
    "rainfall_mean_30day",

    "rainfall_accumulation_3day",
    "rainfall_accumulation_7day",
    "rainfall_accumulation_14day",
    "rainfall_accumulation_30day",

    "rainfall_max_3day",
    "rainfall_max_7day",
    "rainfall_max_14day",
    "rainfall_max_30day",

    "hourly_rainfall_max_7day",
    "hourly_rainfall_max_30day",

    "rainy_days_7day",
    "rainy_days_30day"
]

feature_columns = (
    rain_features
    + river_numeric
    + [
        "historical_disaster_count",
        "previous_landslide_count"
    ]
)

feature_columns = [
    c for c in feature_columns
    if c in rain.columns
]

X = rain[feature_columns].copy()

for col in X.columns:

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median(numeric_only=True)
)

X = X.fillna(0)

y = rain["hazard_type"]


# ============================================================
# TIME-BASED SPLIT
# ============================================================

rain = rain.sort_values(
    "date"
).reset_index(drop=True)

X = X.loc[rain.index]
y = y.loc[rain.index]

split = int(
    len(rain) * 0.80
)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("\n" + "=" * 70)
print("TRAIN / TEST")
print("=" * 70)

print("Training:", len(X_train))
print("Testing:", len(X_test))

print("\nTraining classes:")
print(y_train.value_counts().to_string())

print("\nTesting classes:")
print(y_test.value_counts().to_string())


# ============================================================
# TRAIN
# ============================================================

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train,
    y_train
)

print("Training complete!")


# ============================================================
# PREDICTION
# ============================================================

pred = rf.predict(
    X_test
)

prob = rf.predict_proba(
    X_test
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("🔥 FINAL MODEL RESULTS")
print("=" * 70)

print(
    "\nAccuracy:",
    round(
        accuracy_score(
            y_test,
            pred
        ),
        4
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        pred
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": rf.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\n" + "=" * 70)
print("🔥 TOP 15 IMPORTANT FACTORS")
print("=" * 70)

print(
    importance
    .head(15)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# HAZARD SCORE
# ============================================================

results = rain.iloc[
    split:
].copy()

results["predicted_hazard"] = pred

results["confidence"] = (
    prob.max(axis=1)
)

results["hazard_score"] = (
    results["confidence"] * 100
).round(2)

def severity(score):

    if score <= 20:
        return "LOW"

    elif score <= 40:
        return "MODERATE"

    elif score <= 60:
        return "HIGH"

    elif score <= 80:
        return "VERY HIGH"

    return "EXTREME"


results["severity"] = (
    results["hazard_score"]
    .apply(severity)
)


# ============================================================
# SHOW ACTUAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("🚨 HAZARD PREDICTIONS")
print("=" * 70)

print(
    results[
        [
            "date",
            "district_model",
            "hazard_type",
            "predicted_hazard",
            "hazard_score",
            "severity"
        ]
    ]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    "hazard_predictions.csv",
    index=False
)

importance.to_csv(
    "hazard_feature_importance.csv",
    index=False
)

rain.to_csv(
    "hazard_model_dataset.csv",
    index=False
)

joblib.dump(
    {
        "model": rf,
        "features": feature_columns
    },
    "hazard_model.pkl"
)

print("\n" + "=" * 70)
print("✅ HAZARD ML COMPLETE")
print("=" * 70)

print("hazard_predictions.csv")
print("hazard_feature_importance.csv")
print("hazard_model_dataset.csv")
print("hazard_model.pkl")