import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

# 📌 Get project root directory (IMPORTANT FIX)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 📌 Correct file paths
TRAIN_PATH = os.path.join(BASE_DIR, "data", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "test.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "baseline", "baseline_predictions.csv")

# Load data
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

LABEL_COL = "risk"

# Split features and label
X_train = train.drop(LABEL_COL, axis=1)
y_train = train[LABEL_COL]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Training performance
y_train_pred = model.predict(X_train)

print("\nTraining Performance:")
print(classification_report(y_train, y_train_pred))
print("Macro F1:", f1_score(y_train, y_train_pred, average="macro"))

# Predict on test
y_pred = model.predict(test)

# Create submission
submission = pd.DataFrame({
    "id": range(1, len(y_pred) + 1),
    "prediction": y_pred
})

# Save output
submission.to_csv(OUTPUT_PATH, index=False)

print("✅ Baseline predictions saved at:", OUTPUT_PATH)