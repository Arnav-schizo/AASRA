import pandas as pd
import webbrowser

df = pd.read_csv("hazard_predictions.csv")

# Keep useful output columns
cols = [
    "date",
    "district_model",
    "hazard_type",
    "predicted_hazard",
    "hazard_score",
    "severity"
]

cols = [c for c in cols if c in df.columns]

table = df[cols].head(100).to_html(
    index=False,
    classes="results"
)

html = f"""
<!DOCTYPE html>
<html>
<head>
<title>AASRA Hazard Intelligence</title>

<style>
body {{
    font-family: Arial;
    margin: 40px;
    background: #f5f5f5;
}}

h1 {{
    text-align: center;
}}

.results {{
    width: 100%;
    border-collapse: collapse;
    background: white;
}}

.results th {{
    background: #222;
    color: white;
    padding: 12px;
}}

.results td {{
    padding: 10px;
    border: 1px solid #ddd;
    text-align: center;
}}

.results tr:hover {{
    background: #eee;
}}

.score {{
    font-weight: bold;
}}
</style>
</head>

<body>

<h1>🚨 AASRA — Uttarakhand Hazard Intelligence</h1>

<p>
Showing ML-generated hazard predictions from rainfall,
river and historical disaster data.
</p>

{table}

</body>
</html>
"""

with open(
    "hazard_results.html",
    "w",
    encoding="utf-8"
) as f:
    f.write(html)

webbrowser.open(
    "hazard_results.html"
)

print("Dashboard created:")
print("hazard_results.html")