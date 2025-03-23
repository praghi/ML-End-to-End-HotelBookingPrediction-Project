import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
import os

# Load processed training features
# Ensure absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the current script's directory
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

# Save model
pickle.dump(xgb_model, open("model.pkl", "wb"))

print("✅ Model training complete. Model saved as models/model.pkl")
