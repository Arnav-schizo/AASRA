import sqlite3
import pandas as pd
import numpy as np

# ============================================================
# LOAD ORIGINAL RAINFALL DATA
# ============================================================

conn = sqlite3.connect("disaster.db")

rain = pd.read_sql(
    "SELECT * FROM rainfall_history",
    conn
)

conn.close()

print("Original rainfall records:", len(rain))

# ============================================================
# DATE
# ============================================================

rain["date"] = pd.to_datetime(
    rain["date"],
    errors="coerce"
)

rain = rain.sort_values(
    ["district", "station", "date", "rainfall_id"]
).reset_index(drop=True)

rain["year"] = rain["date"].dt.year
rain["month"] = rain["date"].dt.month
rain["day_of_year"] = rain["date"].dt.dayofyear

# ============================================================
# SAME-DAY STATISTICS
# ============================================================

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

# ============================================================
# ROLLING FEATURES
# ============================================================

groups = []

for (district, station), group in daily.groupby(
    ["district", "station"]
):

    group = group.sort_values("date").copy()

    x = group.set_index("date")

    for days in [3, 7, 14, 30]:

        window = f"{days}D"

        x[f"rainfall_mean_{days}day"] = (
            x["rainfall_mean"]
            .rolling(window, min_periods=1)
            .mean()
        )

        x[f"rainfall_accumulation_{days}day"] = (
            x["rainfall_mean"]
            .rolling(window, min_periods=1)
            .sum()
        )

        x[f"rainfall_max_{days}day"] = (
            x["rainfall_max"]
            .rolling(window, min_periods=1)
            .max()
        )

    x["hourly_rainfall_max_7day"] = (
        x["hourly_rainfall_max"]
        .rolling("7D", min_periods=1)
        .max()
    )

    x["hourly_rainfall_max_30day"] = (
        x["hourly_rainfall_max"]
        .rolling("30D", min_periods=1)
        .max()
    )

    x["rainfall_std_7day"] = (
        x["rainfall_mean"]
        .rolling("7D", min_periods=2)
        .std()
        .fillna(0)
    )

    x["rainfall_std_30day"] = (
        x["rainfall_mean"]
        .rolling("30D", min_periods=2)
        .std()
        .fillna(0)
    )

    x["rainy_days_7day"] = (
        x["rainfall_mean"]
        .gt(0)
        .rolling("7D", min_periods=1)
        .sum()
    )

    x["rainy_days_30day"] = (
        x["rainfall_mean"]
        .gt(0)
        .rolling("30D", min_periods=1)
        .sum()
    )

    groups.append(x.reset_index())

daily_features = pd.concat(
    groups,
    ignore_index=True
)

# ============================================================
# MERGE BACK TO ALL ORIGINAL RECORDS
# ============================================================

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

# ============================================================
# VALIDATION
# ============================================================

print("\nFEATURE RECORDS:", len(rain_features))
print(
    "UNIQUE RAINFALL IDs:",
    rain_features["rainfall_id"].nunique()
)

print(
    "FEATURE COLUMNS:",
    len(rain_features.columns)
)

# ============================================================
# SAVE
# ============================================================

rain_features.to_csv(
    "hazard_rainfall_features.csv",
    index=False
)

print("\nSUCCESS!")
print("Created: hazard_rainfall_features.csv")