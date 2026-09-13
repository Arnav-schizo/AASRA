import sqlite3

conn = sqlite3.connect("disaster.db")
cursor = conn.cursor()

cursor.execute("""
SELECT
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
    in_historical_flood_zone,
    distance_to_nearest_river_km,
    distance_to_nearest_hospital_km,
    elevation_meters,
    slope_degrees,
    vulnerability_level
FROM households
LIMIT 10
""")

columns = [
    "population",
    "children",
    "elderly",
    "disabled",
    "low_income",
    "hazard_exposure",
    "previous_disaster_exposure",
    "flood_exposure",
    "river_exposure",
    "landslide_exposure",
    "rainfall_exposure",
    "in_historical_flood_zone",
    "distance_to_nearest_river_km",
    "distance_to_nearest_hospital_km",
    "elevation_meters",
    "slope_degrees",
    "vulnerability_level"
]

print("ML DATA")
print("=" * 80)

for row in cursor.fetchall():
    for column, value in zip(columns, row):
        print(f"{column}: {value}")
    print("-" * 80)

conn.close()