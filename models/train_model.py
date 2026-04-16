import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate dataset
np.random.seed(42)
rows = 300

data = pd.DataFrame({
    "Attendance": np.random.randint(30, 100, rows),
    "Internal_Marks": np.random.randint(20, 100, rows),
    "Assignment_Score": np.random.randint(20, 100, rows),
    "Study_Hours": np.random.randint(1, 10, rows),
    "Sleep_Hours": np.random.randint(4, 10, rows)
})

# Create Risk column
data["Risk"] = np.where(
    (data["Attendance"] < 60) |
    (data["Internal_Marks"] < 45) |
    (data["Study_Hours"] < 3),
    "Yes", "No"
)

# Encode
le = LabelEncoder()
data["Risk"] = le.fit_transform(data["Risk"])

X = data.drop("Risk", axis=1)
y = data["Risk"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model
joblib.dump(model, "student_model.pkl")

# Save dataset
data.to_csv("data/student_data.csv", index=False)

print("Model & dataset saved!")