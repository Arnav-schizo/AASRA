import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ============================================================
# 1. LOAD ENGINEERED DATA
# ============================================================

rain = pd.read_csv("hazard_rainfall_features.csv")
river = pd.read_csv("hazard_river_features.csv")
flood = pd.read_csv("hazard_flood_features.csv")
landslide = pd.read_csv("hazard_landslide_features.csv")

print("=" * 70)
print("LOADED DATA")
print("=" * 70)

print("Rainfall:", len(rain))
print("River:", len(river))
print("Flood:", len(flood))
print("Landslide:", len(landslide))


# ============================================================
# 2. PREPARE RAINFALL MODEL DATA
# ============================================================

rain["date"] = pd.to_datetime(
    rain["date"],
    errors="coerce"
)

rain["district"] = (
    rain["district"]
    .astype("string")
    .str.strip()
)

rain["year"] = rain["date"].dt.year


# ============================================================
# 3. PREPARE FLOOD HISTORY
# ============================================================

flood["start_date"] = pd.to_datetime(
    flood["start_date"],
    errors="coerce"
)

flood["flood_year"] = (
    flood["start_date"].dt.year
)

# Use districts field for historical district matching
if "districts" in flood.columns:

    flood["flood_district"] = (
        flood["districts"]
        .astype("string")
        .str.strip()
    )

else:

    flood["flood_district"] = pd.Series(
        pd.NA,
        index=flood.index,
        dtype="string"
    )


# ============================================================
# 4. PREPARE LANDSLIDE HISTORY
# ============================================================

landslide["landslide_district"] = (
    landslide["landslide_district_standard"]
    .astype("string")
    .str.strip()
)

landslide["landslide_year"] = pd.to_numeric(
    landslide["landslide_history_year"],
    errors="coerce"
)


# ============================================================
# 5. NORMALIZE DISTRICT NAMES
# ============================================================

def clean_district(x):

    if pd.isna(x):
        return np.nan

    x = str(x).strip().lower()

    replacements = {
        "uttar kashi": "uttarkashi",
        "uttar-kashi": "uttarkashi",
        "u.s. nagar": "udham singh nagar",
        "udham singh nagar": "udham singh nagar",
        "pithoragarh": "pithoragarh",
        "chamoli": "chamoli",
        "nainital": "nainital",
        "dehradun": "dehradun",
        "almora": "almora",
        "bageshwar": "bageshwar",
        "tehri garhwal": "tehri garhwal",
        "pauri garhwal": "pauri garhwal",
        "rudraprayag": "rudraprayag",
        "haridwar": "haridwar",
        "champawat": "champawat"
    }

    return replacements.get(x, x)


rain["district_model"] = rain["district"].apply(
    clean_district
)

landslide["district_model"] = (
    landslide["landslide_district"]
    .apply(clean_district)
)

# Flood district strings can contain multiple districts.
# We will use substring matching later rather than
# destroying the original flood information.


# ============================================================
# 6. CREATE LANDSLIDE HISTORICAL COUNTS
# ============================================================

landslide_district_counts = (
    landslide
    .groupby("district_model")
    .size()
    .rename("historical_landslide_count")
    .reset_index()
)

landslide_year_counts = (
    landslide
    .dropna(subset=["landslide_year"])
    .groupby(
        ["district_model", "landslide_year"]
    )
    .size()
    .rename("landslides_same_year")
    .reset_index()
)


# ============================================================
# 7. CREATE FLOOD DISTRICT-YEAR COUNTS
# ============================================================

flood_rows = []

for _, row in flood.iterrows():

    district_text = row.get("flood_district", np.nan)
    year = row.get("flood_year", np.nan)

    if pd.isna(district_text):
        continue

    if pd.isna(year):
        continue

    district_text = str(district_text).lower()

    for district in [
        "chamoli",
        "uttarkashi",
        "dehradun",
        "almora",
        "nainital",
        "tehri garhwal",
        "pauri garhwal",
        "rudraprayag",
        "pithoragarh",
        "bageshwar",
        "haridwar",
        "champawat",
        "udham singh nagar"
    ]:

        if district in district_text:

            flood_rows.append({
                "district_model": district,
                "flood_year": int(year)
            })


flood_events = pd.DataFrame(
    flood_rows
)

if len(flood_events) > 0:

    flood_year_counts = (
        flood_events
        .groupby(
            ["district_model", "flood_year"]
        )
        .size()
        .rename("floods_same_year")
        .reset_index()
    )

else:

    flood_year_counts = pd.DataFrame(
        columns=[
            "district_model",
            "flood_year",
            "floods_same_year"
        ]
    )


# ============================================================
# 8. BUILD MODEL DATASET
# ============================================================

model = rain.copy()

# Historical landslide district count
model = model.merge(
    landslide_district_counts,
    on="district_model",
    how="left"
)

# Historical landslides in same year
model = model.merge(
    landslide_year_counts,
    left_on=[
        "district_model",
        "year"
    ],
    right_on=[
        "district_model",
        "landslide_year"
    ],
    how="left"
)

model.drop(
    columns=["landslide_year"],
    inplace=True,
    errors="ignore"
)

# Historical floods in same year
model = model.merge(
    flood_year_counts,
    left_on=[
        "district_model",
        "year"
    ],
    right_on=[
        "district_model",
        "flood_year"
    ],
    how="left"
)

model.drop(
    columns=["flood_year"],
    inplace=True,
    errors="ignore"
)

model["historical_landslide_count"] = (
    model["historical_landslide_count"]
    .fillna(0)
)

model["landslides_same_year"] = (
    model["landslides_same_year"]
    .fillna(0)
)

model["floods_same_year"] = (
    model["floods_same_year"]
    .fillna(0)
)


# ============================================================
# 9. CREATE PROTOTYPE TARGET
# ============================================================

# Priority:
# FLOOD if historical flood occurred that district/year
# LANDSLIDE if historical landslide occurred that district/year
# NORMAL otherwise.
#
# This is a prototype historical-event label, NOT an official
# ground-truth hazard classification.

model["hazard_type"] = "NORMAL"

model.loc[
    model["landslides_same_year"] > 0,
    "hazard_type"
] = "LANDSLIDE"

model.loc[
    model["floods_same_year"] > 0,
    "hazard_type"
] = "FLOOD"


# ============================================================
# 10. CHECK TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(
    model["hazard_type"]
    .value_counts()
    .to_string()
)


# ============================================================
# 11. SELECT FEATURES
# ============================================================

candidate_features = [
    "rainfall_mean",
    "rainfall_min",
    "rainfall_max",
    "rainfall_std",
    "rainfall_sum",
    "max_hourly_rainfall_mm",
    "hourly_rainfall_mean",
    "hourly_rainfall_min",
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

    "rainy_days_7day",
    "rainy_days_30day",

    "historical_landslide_count",
    "landslides_same_year",
    "floods_same_year"
]

features = [
    c for c in candidate_features
    if c in model.columns
]

print("\nFEATURES USED:")
print(features)


# ============================================================
# 12. CLEAN NUMERIC DATA
# ============================================================

X = model[features].copy()

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

y = model["hazard_type"]


# ============================================================
# 13. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

model = model.sort_values(
    "date"
).reset_index(drop=True)

X = X.loc[model.index]
y = y.loc[model.index]

split_index = int(
    len(model) * 0.80
)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("\n" + "=" * 70)
print("TRAIN / TEST")
print("=" * 70)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining target:")
print(y_train.value_counts().to_string())

print("\nTesting target:")
print(y_test.value_counts().to_string())


# ============================================================
# 14. TRAIN RANDOM FOREST
# ============================================================

model_rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model_rf.fit(
    X_train,
    y_train
)


# ============================================================
# 15. PREDICTIONS
# ============================================================

predictions = model_rf.predict(
    X_test
)

probabilities = model_rf.predict_proba(
    X_test
)


# ============================================================
# 16. EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL RESULTS")
print("=" * 70)

print(
    "\nAccuracy:",
    round(
        accuracy_score(
            y_test,
            predictions
        ),
        4
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": features,
    "importance": model_rf.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\n" + "=" * 70)
print("TOP FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 18. SAMPLE PREDICTIONS
# ============================================================

result = model.loc[
    model.index[split_index:]
].copy()

result["predicted_hazard"] = predictions

# Maximum probability
result["prediction_confidence"] = (
    probabilities.max(axis=1)
)

print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

sample_columns = [
    c for c in [
        "date",
        "district",
        "rainfall_max_3day",
        "rainfall_max_7day",
        "rainfall_accumulation_7day",
        "historical_landslide_count",
        "landslides_same_year",
        "floods_same_year",
        "hazard_type",
        "predicted_hazard",
        "prediction_confidence"
    ]
    if c in result.columns
]

print(
    result[
        sample_columns
    ]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# 19. SAVE MODEL RESULTS
# ============================================================

result.to_csv(
    "hazard_predictions.csv",
    index=False
)

importance.to_csv(
    "hazard_feature_importance.csv",
    index=False
)

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print("hazard_predictions.csv")
print("hazard_feature_importance.csv")
# ============================================================
# RAINFALL FEATURES — SAVE FINAL CSV
# ============================================================

import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("disaster.db")
rain = pd.read_sql("SELECT * FROM rainfall_history", conn)
conn.close()

rain["date"] = pd.to_datetime(rain["date"], errors="coerce")

rain = rain.sort_values(
    ["district", "station", "date", "rainfall_id"]
).reset_index(drop=True)

# Time features
rain["year"] = rain["date"].dt.year
rain["month"] = rain["date"].dt.month
rain["day_of_year"] = rain["date"].dt.dayofyear

# ------------------------------------------------------------
# Same station + date statistical profile
# ------------------------------------------------------------

daily = (
    rain.groupby(
        ["district", "station", "date"],
        dropna=False
    )
    .agg(
        rainfall_mean=("daily_rainfall_mm", "mean"),
        rainfall_min=("daily_rainfall_mm", "min"),
        rainfall_max=("daily_rainfall_mm", "max"),
        rainfall_std=("daily_rainfall_mm", "std"),
        rainfall_sum=("daily_rainfall_mm", "sum"),

        hourly_rainfall_mean=("max_hourly_rainfall_mm", "mean"),
        hourly_rainfall_min=("max_hourly_rainfall_mm", "min"),
        hourly_rainfall_max=("max_hourly_rainfall_mm", "max"),

        readings_sum=("rainfall_readings", "sum"),
        readings_mean=("rainfall_readings", "mean")
    )
    .reset_index()
)

daily["rainfall_std"] = daily["rainfall_std"].fillna(0)

# ------------------------------------------------------------
# Calendar-based rolling features
# ------------------------------------------------------------

feature_groups = []

for (district, station), group in daily.groupby(
    ["district", "station"]
):

    group = group.sort_values("date").copy()

    indexed = group.set_index("date")

    for window in ["3D", "7D", "14D", "30D"]:

        days = window.replace("D", "")

        indexed[f"rainfall_mean_{days}day"] = (
            indexed["rainfall_mean"]
            .rolling(window, min_periods=1)
            .mean()
        )

        indexed[f"rainfall_accumulation_{days}day"] = (
            indexed["rainfall_mean"]
            .rolling(window, min_periods=1)
            .sum()
        )

        indexed[f"rainfall_max_{days}day"] = (
            indexed["rainfall_max"]
            .rolling(window, min_periods=1)
            .max()
        )

    indexed["hourly_rainfall_max_7day"] = (
        indexed["hourly_rainfall_max"]
        .rolling("7D", min_periods=1)
        .max()
    )

    indexed["hourly_rainfall_max_30day"] = (
        indexed["hourly_rainfall_max"]
        .rolling("30D", min_periods=1)
        .max()
    )

    indexed["rainfall_std_7day"] = (
        indexed["rainfall_mean"]
        .rolling("7D", min_periods=2)
        .std()
        .fillna(0)
    )

    indexed["rainfall_std_30day"] = (
        indexed["rainfall_mean"]
        .rolling("30D", min_periods=2)
        .std()
        .fillna(0)
    )

    indexed["rainy_days_7day"] = (
        indexed["rainfall_mean"]
        .gt(0)
        .rolling("7D", min_periods=1)
        .sum()
    )

    indexed["rainy_days_30day"] = (
        indexed["rainfall_mean"]
        .gt(0)
        .rolling("30D", min_periods=1)
        .sum()
    )

    feature_groups.append(
        indexed.reset_index()
    )

daily_features = pd.concat(
    feature_groups,
    ignore_index=True
)

# ------------------------------------------------------------
# Merge features back to EVERY original rainfall record
# ------------------------------------------------------------

feature_columns = [
    "district",
    "station",
    "date",
    "rainfall_mean",
    "rainfall_min",
    "rainfall_max",
    "rainfall_std",
    "rainfall_sum",
    "hourly_rainfall_mean",
    "hourly_rainfall_min",
    "hourly_rainfall_max",
    "readings_sum",
    "readings_mean",
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
    "rainfall_std_7day",
    "rainfall_std_30day",
    "rainy_days_7day",
    "rainy_days_30day"
]

rain_features = rain.merge(
    daily_features[feature_columns],
    on=["district", "station", "date"],
    how="left"
)

# Save
rain_features.to_csv(
    "hazard_rainfall_features.csv",
    index=False
)

print("\n" + "=" * 70)
print("RAINFALL FEATURES CREATED")
print("=" * 70)

print("Original rainfall records:", len(rain))
print("Feature rainfall records:", len(rain_features))
print("Unique rainfall IDs:", rain_features["rainfall_id"].nunique())
print("Feature columns:", len(rain_features.columns))

print("\nSAVED:")
print("hazard_rainfall_features.csv")