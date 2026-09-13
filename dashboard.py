import pandas as pd
import webbrowser

# ============================================================
# LOAD ML RESULTS
# ============================================================

df = pd.read_csv("hazard_predictions.csv")

# Pick the highest-risk record for the main card
# Ignore NORMAL predictions when selecting
# the main hazard assessment.
hazard_df = df[
    df["predicted_hazard"].str.upper() != "NORMAL"
].copy()

if len(hazard_df) > 0:
    hazard_df = hazard_df.sort_values(
        "hazard_score",
        ascending=False
    )
    row = hazard_df.iloc[0]
else:
    row = df.sort_values(
        "hazard_score",
        ascending=False
    ).iloc[0]

district = str(row["district_model"]).title()
hazard = str(row["predicted_hazard"])
score = float(row["hazard_score"])
severity = str(row["severity"])
date = str(row["date"])

# ============================================================
# SEVERITY DISPLAY
# ============================================================

if severity == "EXTREME":
    badge = "🔴"
elif severity == "VERY HIGH":
    badge = "🟠"
elif severity == "HIGH":
    badge = "🟡"
elif severity == "MODERATE":
    badge = "🟢"
else:
    badge = "🟢"

# ============================================================
# ACTION
# ============================================================

if hazard == "LANDSLIDE":
    action = (
        "Prepare evacuation for vulnerable locations, "
        "avoid unstable slopes and restrict movement "
        "through landslide-prone areas."
    )
elif hazard == "FLOOD":
    action = (
        "Prepare evacuation for flood-prone locations, "
        "avoid flooded roads and move vulnerable residents "
        "towards designated safe areas."
    )
else:
    action = (
        "Continue monitoring rainfall, river conditions "
        "and historical hazard indicators."
    )

# ============================================================
# TOP FACTORS
# ============================================================

try:
    importance = pd.read_csv(
        "hazard_feature_importance.csv"
    )

    top_factors = importance.sort_values(
        "importance",
        ascending=False
    ).head(5)

    factors_html = ""

    for _, f in top_factors.iterrows():

        name = str(f["feature"]).replace("_", " ").title()
        value = float(f["importance"])

        factors_html += f"""
        <div class="factor">
            <span>{name}</span>
            <b>{value:.3f}</b>
        </div>
        """

except Exception:

    factors_html = """
    <div class="factor">
        Model feature information unavailable
    </div>
    """

# ============================================================
# HTML
# ============================================================

html = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>AASRA Hazard Intelligence</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    font-family: Arial, sans-serif;
    background: #eef2f5;
    color: #17202a;
}}

.header {{
    background: #111827;
    color: white;
    padding: 22px 40px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}

.logo {{
    font-size: 28px;
    font-weight: bold;
}}

.status {{
    background: #16a34a;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 14px;
}}

.container {{
    max-width: 1250px;
    margin: 30px auto;
    padding: 0 20px;
}}

.grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
}}

.card {{
    background: white;
    border-radius: 18px;
    padding: 28px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.08);
}}

.location {{
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 8px;
}}

.date {{
    color: #6b7280;
    margin-bottom: 25px;
}}

.score {{
    text-align: center;
    padding: 20px;
}}

.score-number {{
    font-size: 76px;
    font-weight: bold;
}}

.score-label {{
    font-size: 22px;
    font-weight: bold;
    margin-top: 5px;
}}

.hazard {{
    font-size: 30px;
    font-weight: bold;
    text-align: center;
    margin: 20px 0;
}}

.progress {{
    height: 16px;
    background: #e5e7eb;
    border-radius: 10px;
    overflow: hidden;
}}

.progress-bar {{
    height: 100%;
    width: {score}%;
    background: #dc2626;
}}

.section-title {{
    font-size: 19px;
    font-weight: bold;
    margin: 25px 0 15px;
}}

.condition {{
    display: flex;
    justify-content: space-between;
    padding: 14px;
    margin: 8px 0;
    background: #f8fafc;
    border-radius: 10px;
}}

.factor {{
    display: flex;
    justify-content: space-between;
    padding: 12px;
    border-bottom: 1px solid #e5e7eb;
}}

.action {{
    background: #fff7ed;
    border-left: 5px solid #f97316;
    padding: 18px;
    border-radius: 8px;
    line-height: 1.6;
}}

.map {{
    height: 360px;
    background: #dbeafe;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    color: #1e3a8a;
}}

.footer {{
    text-align: center;
    color: #6b7280;
    padding: 30px;
}}

@media(max-width: 800px) {{
    .grid {{
        grid-template-columns: 1fr;
    }}
}}

</style>

</head>

<body>

<div class="header">

<div class="logo">
🚨 AASRA — Disaster Preparedness
</div>

<div class="status">
● ML ENGINE ACTIVE
</div>

</div>


<div class="container">

<div class="grid">

<!-- SCORE CARD -->

<div class="card">

<div class="location">
📍 {district}
</div>

<div class="date">
Assessment date: {date}
</div>

<div class="score">

<div class="score-number">
{score:.2f}
</div>

<div class="score-label">
{badge} {severity}
</div>

</div>

<div class="progress">
<div class="progress-bar"></div>
</div>

<div class="hazard">
{hazard}
</div>

<div class="section-title">
Hazard Assessment
</div>

<div class="condition">
<span>ML Hazard Probability</span>
<b>{score:.2f}%</b>
</div>

<div class="condition">
<span>Risk Category</span>
<b>{severity}</b>
</div>

<div class="condition">
<span>Prediction Source</span>
<b>Random Forest</b>
</div>

</div>


<!-- CONDITIONS -->

<div class="card">

<div class="section-title">
🌧 Current / Historical Indicators
</div>

<div class="condition">
<span>Rainfall indicators</span>
<b>ANALYZED</b>
</div>

<div class="condition">
<span>River discharge</span>
<b>ANALYZED</b>
</div>

<div class="condition">
<span>Flood history</span>
<b>ANALYZED</b>
</div>

<div class="condition">
<span>Landslide history</span>
<b>ANALYZED</b>
</div>

<div class="condition">
<span>Historical disasters</span>
<b>ANALYZED</b>
</div>

<div class="section-title">
🔎 Important Factors
</div>

{factors_html}

</div>


<!-- ACTION -->

<div class="card">

<div class="section-title">
⚠️ Recommended Action
</div>

<div class="action">
{action}
</div>

<div class="section-title">
🚑 Decision Support
</div>

<div class="condition">
<span>Hazard status</span>
<b>{hazard}</b>
</div>

<div class="condition">
<span>Priority</span>
<b>{severity}</b>
</div>

</div>


<!-- MAP PLACEHOLDER -->

<div class="card">

<div class="section-title">
🗺 Uttarakhand Hazard Map
</div>

<div class="map">
Uttarakhand Hazard Map
<br>
<br>
📍 {district}
</div>

</div>

</div>

</div>

<div class="footer">
AASRA • ML-assisted disaster risk intelligence • Uttarakhand
</div>

</body>

</html>
"""

# ============================================================
# SAVE + OPEN
# ============================================================

with open(
    "aasra_scorecard.html",
    "w",
    encoding="utf-8"
) as f:

    f.write(html)

webbrowser.open(
    "aasra_scorecard.html"
)

print("========================================")
print("AASRA SCORECARD CREATED")
print("========================================")
print("District:", district)
print("Hazard:", hazard)
print("Score:", round(score, 2))
print("Severity:", severity)
print("")
print("Opened: aasra_scorecard.html")