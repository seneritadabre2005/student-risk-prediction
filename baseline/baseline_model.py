import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
OUTPUT_PATH = "baseline/baseline_predictions.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

LABEL_COL = "risk"

X_train = train.drop(LABEL_COL, axis=1)
y_train = train[LABEL_COL]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

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

submission.to_csv(OUTPUT_PATH, index=False)

print("✅ Baseline predictions saved!")