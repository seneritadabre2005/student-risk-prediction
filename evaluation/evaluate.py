import pandas as pd
import os
from sklearn.metrics import accuracy_score

GROUND_TRUTH = "data/test_labels.csv"
SUBMISSION_FOLDER = "submissions/"
LEADERBOARD_FILE = "evaluation/leaderboard.csv"

# Load ground truth
gt = pd.read_csv(GROUND_TRUTH)
gt.columns = gt.columns.str.strip()

print("GT columns:", gt.columns.tolist())

results = []

for file in os.listdir(SUBMISSION_FOLDER):
    if file.endswith(".csv"):
        path = os.path.join(SUBMISSION_FOLDER, file)

        try:
            sub = pd.read_csv(path)

            # DEBUG print
            print(file, "columns:", sub.columns.tolist())

            # Clean column names
            sub.columns = sub.columns.str.strip()

            # Check required columns
            if "id" not in sub.columns or "prediction" not in sub.columns:
                print(f"{file} skipped (wrong format)")
                continue

            # Check row count
            if len(sub) != len(gt):
                print(f"{file} skipped (row mismatch)")
                continue

            # Merge on ID
            merged = pd.merge(gt, sub, on="id")

            # Final safety check
            if "risk" not in merged.columns:
                print(f"{file} skipped (risk column missing in GT)")
                continue

            # Calculate accuracy
            score = accuracy_score(merged["risk"], merged["prediction"])

            results.append({
                "team": file.replace(".csv", ""),
                "accuracy": round(score, 4)
            })

        except Exception as e:
            print(f"Error in {file}: {e}")

# Create leaderboard
leaderboard = pd.DataFrame(results)

if not leaderboard.empty:
    leaderboard = leaderboard.sort_values(by="accuracy", ascending=False)

leaderboard.to_csv(LEADERBOARD_FILE, index=False)

print("✅ Leaderboard updated!")