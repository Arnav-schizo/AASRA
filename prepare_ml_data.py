import sqlite3
import pandas as pd

# Connect to existing database
conn = sqlite3.connect("disaster.db")

# Load household data
query = """
SELECT
    household_id,
    population,
    children,
    elderly,
    disabled,
    low_income,
    hazard_exposure,
    previous_disaster_exposure,
    flood_exposure,
    river_exposure,
    landslide_exposure,
    rainfall_exposure,
    distance_to_nearest_hospital_km,
    vulnerability_level
FROM households
"""

df = pd.read_sql_query(query, conn)

conn.close()


# --------------------------------------------------
# Step 1: Display original data
# --------------------------------------------------

print("Original data:")
print(df)

print("\nShape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())


# --------------------------------------------------
# Step 2: Separate input features and target
# --------------------------------------------------

# household_id is only an identifier.
# It must NOT be used as an ML feature.

X = df.drop(
    ["household_id", "vulnerability_level"],
    axis=1
)

y = df["vulnerability_level"]

print("\nFeatures (X):")
print(X.columns.tolist())

print("\nTarget (y):")
print(y.value_counts())


# --------------------------------------------------
# Step 3: Split data into training and testing data
# --------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# --------------------------------------------------
# Step 4: Create and train Random Forest model
# --------------------------------------------------

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

# Train the model
model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")


# --------------------------------------------------
# Step 5: Evaluate the model
# --------------------------------------------------

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Predict the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nPredictions:")
print(y_pred)

print("\nActual values:")
print(y_test.values)

print("\nAccuracy:")
print(accuracy)

print("\nAccuracy percentage:")
print(accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------
# Step 5.1: Feature importance
# --------------------------------------------------

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)


# --------------------------------------------------
# Step 5.2: Predict vulnerability for all households
# --------------------------------------------------

all_predictions = model.predict(X)

# Add ML predictions to dataframe
df["ml_vulnerability"] = all_predictions

print("\nML predictions for all households:")

print(
    df[
        [
            "household_id",
            "vulnerability_level",
            "ml_vulnerability"
        ]
    ]
)


# --------------------------------------------------
# Step 5.3: Compare original and ML predictions
# --------------------------------------------------

correct = (
    df["vulnerability_level"] ==
    df["ml_vulnerability"]
).sum()

total = len(df)

print("\nCorrect predictions:")
print(correct, "out of", total)

print("\nOverall agreement:")
print((correct / total) * 100, "%")


# --------------------------------------------------
# Step 6: Convert ML vulnerability level into score
# --------------------------------------------------

vulnerability_score_map = {
    "HIGH": 100,
    "MEDIUM": 60,
    "LOW": 30
}

df["ml_vulnerability_score"] = df[
    "ml_vulnerability"
].map(vulnerability_score_map)

print("\nML Vulnerability Scores:")

print(
    df[
        [
            "household_id",
            "vulnerability_level",
            "ml_vulnerability",
            "ml_vulnerability_score"
        ]
    ]
)


# --------------------------------------------------
# Step 7: Export ML results
# --------------------------------------------------

output = df[
    [
        "household_id",
        "ml_vulnerability",
        "ml_vulnerability_score"
    ]
]

output.to_csv(
    "member2_vulnerability_output.csv",
    index=False
)

print("\n----------------------------------------")
print("Output file created successfully!")
print("File: member2_vulnerability_output.csv")
print("----------------------------------------")

print("\nFinal ML output:")
print(output)

import joblib

joblib.dump(model, "vulnerability_model.pkl")

print("\nML model saved successfully!")
print("File: vulnerability_model.pkl")