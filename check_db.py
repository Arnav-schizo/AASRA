import sqlite3

conn = sqlite3.connect("disaster.db")
cursor = conn.cursor()

cursor.execute("""
SELECT
    household_id,
    population,
    children,
    elderly,
    disabled,
    low_income,
    hazard_exposure,
    previous_disaster_exposure,
    vulnerability_score,
    vulnerability_level,
    priority_level,
    risk_score,
    risk_level
FROM households
LIMIT 10
""")

for row in cursor.fetchall():
    print(row)

conn.close()