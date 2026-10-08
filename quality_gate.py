import json

with open("metrics.json", "r") as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]

print("Model Accuracy:", accuracy)

if accuracy >= 0.80:
    print("ML Quality Gate PASSED")
else:
    print("ML Quality Gate FAILED")
    raise SystemExit(1)
