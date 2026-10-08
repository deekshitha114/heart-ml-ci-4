import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
data = pd.read_csv("heart.csv")

# Separate features and target
X = data.drop("target", axis=1)
y = data["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "training_records": len(X_train),
    "testing_records": len(X_test)
}

# Save model
joblib.dump(model, "heart_model.pkl")

# Save metrics
with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

# Save predictions
predictions = X_test.copy()
predictions["actual"] = y_test.values
predictions["predicted"] = y_pred

predictions.to_csv("heart_predictions.csv", index=False)

print("Model trained successfully")
print(metrics)
