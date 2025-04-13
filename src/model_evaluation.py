import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix

# Get project root (where main.py is located)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(PROJECT_ROOT)

# Load trained model from the main project directory
model_path = os.path.join(PROJECT_ROOT, "model.pkl")
loaded_obj = pickle.load(open(model_path, "rb"))

# Unpack the tuple if necessary
if isinstance(loaded_obj, tuple):
    xgb_model, scaler = loaded_obj
else:
    xgb_model = loaded_obj

# Load test data from the correct path
test_data_path = os.path.join(PROJECT_ROOT, "data", "features", "test_features.csv")
test_data = pd.read_csv(test_data_path)

X_test = test_data.iloc[:, :-1].values  # Features
y_test = test_data.iloc[:, -1].values   # Target

# Make predictions
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]  # Probabilities for positive class

# Display first 10 actual vs. predicted values
df_comparison = pd.DataFrame({'Actual': y_test[:10], 'Predicted': y_pred[:10]})
print("📊 First 10 Predictions:\n", df_comparison)

# Calculate evaluation metrics
metrics_dict = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "auc": roc_auc_score(y_test, y_pred_proba),
}

# Save metrics to JSON in the project root
metrics_path = os.path.join(PROJECT_ROOT, "metrics.json")
with open(metrics_path, "w") as file:
    json.dump(metrics_dict, file, indent=4)

# Print detailed evaluation metrics
print("\n📊 Model Evaluation Metrics:")
print("Accuracy:", metrics_dict["accuracy"])
print("Precision:", metrics_dict["precision"])
print("Recall:", metrics_dict["recall"])
print("AUC:", metrics_dict["auc"])
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\n✅ Model evaluation complete. Metrics saved in:", metrics_path)

