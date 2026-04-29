import pandas as pd
import os
from sklearn.metrics import accuracy_score

# ---------------- PATHS ----------------
GROUND_TRUTH = "data/test_labels.csv"
SUBMISSION_FOLDER = "submissions/"
LEADERBOARD_FILE = "evaluation/leaderboard.csv"
HTML_FILE = "evaluation/leaderboard.html"

# ---------------- LOAD GROUND TRUTH ----------------
gt = pd.read_csv(GROUND_TRUTH)
gt.columns = gt.columns.str.strip()

print("GT columns:", gt.columns.tolist())

results = []

# ---------------- PROCESS SUBMISSIONS ----------------
for file in os.listdir(SUBMISSION_FOLDER):
    if file.endswith(".csv"):
        path = os.path.join(SUBMISSION_FOLDER, file)

        try:
            sub = pd.read_csv(path)

            # Debug
            print(file, "columns:", sub.columns.tolist())

            # Clean column names
            sub.columns = sub.columns.str.strip()

            # Check format
            if "id" not in sub.columns or "prediction" not in sub.columns:
                print(f"{file} skipped (wrong format)")
                continue

            # Check row count
            if len(sub) != len(gt):
                print(f"{file} skipped (row mismatch)")
                continue

            # Merge on ID
            merged = pd.merge(gt, sub, on="id")

            # Safety check
            if "risk" not in merged.columns:
                print(f"{file} skipped (risk column missing)")
                continue

            # Calculate accuracy
            score = accuracy_score(merged["risk"], merged["prediction"])

            results.append({
                "team": file.replace(".csv", ""),
                "accuracy": round(score, 4)
            })

        except Exception as e:
            print(f"Error in {file}: {e}")

# ---------------- CREATE LEADERBOARD ----------------
# ---------------- CREATE LEADERBOARD ----------------
leaderboard = pd.DataFrame(results)

if not leaderboard.empty:
    leaderboard = leaderboard.sort_values(by="accuracy", ascending=False)
    leaderboard.to_csv(LEADERBOARD_FILE, index=False)

# ✅ ALWAYS READ FROM CSV (THIS IS THE FIX)
if os.path.exists(LEADERBOARD_FILE):
    leaderboard = pd.read_csv(LEADERBOARD_FILE)
else:
    leaderboard = pd.DataFrame(columns=["team", "accuracy"])

# ---------------- CREATE HTML ----------------
html_table = leaderboard.to_html(index=False)

with open("evaluation/leaderboard.html", "w", encoding="utf-8") as f:
    f.write(f"""
    <html>
    <head>
        <title>Leaderboard</title>
        <style>
            body {{
                font-family: Arial;
                text-align: center;
                background-color: #f4f4f4;
            }}
            h1 {{
                color: #333;
            }}
            table {{
                border-collapse: collapse;
                margin: auto;
                background: white;
                box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
            }}
        </style>
    </head>
    <body>
        <h1>Leaderboard</h1>
        {html_table}
    </body>
    </html>
    """)

print("✅ Leaderboard updated (CSV → HTML linked)!")