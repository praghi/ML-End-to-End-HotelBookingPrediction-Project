import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import os

# Load processed training features
# Ensure absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the current script's directory
ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Root directory
DATA_PATH = os.path.join(BASE_DIR, "../data/features/train_features.csv")  # Adjust path
print(DATA_PATH)
# Load training data
train_data = pd.read_csv(DATA_PATH) 
print(train_data.head(5))
# Split features (X) and target (y)
X_train = train_data.iloc[:, :-1].values  # All columns except the last
y_train = train_data.iloc[:, -1].values   # Last column is the target


# Define and train the XGBoost model
xgb_model = XGBClassifier(n_estimators=200, max_depth=9, learning_rate=0.2, random_state=42)
xgb_model.fit(X_train, y_train)

# Save both model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

# Save the trained scaler in the main directory

# Save the trained model in the main directory
model_path = os.path.join(ROOT_DIR, "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(xgb_model, f)


print(f"✅ Model saved at {model_path}")

print("✅ Model and scaler saved as model_with_scaler.pkl")
