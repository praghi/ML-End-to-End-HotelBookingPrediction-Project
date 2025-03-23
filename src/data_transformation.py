import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Automatically detect the project's root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

def load_data(data_path=DATA_DIR) -> tuple:
    """Loads train and test data from CSV files using the default data directory."""
    train_path = os.path.join(data_path, "train.csv")
    test_path = os.path.join(data_path, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Error: Train/Test data files are missing in {data_path}!")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df

def transform_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Standardizes numerical features and encodes categorical columns."""
    scaler = StandardScaler()

    for df in [train_df, test_df]:
        df[['lead_time', 'avg_price_per_room']] = scaler.fit_transform(df[['lead_time', 'avg_price_per_room']])
        df['booking_status'] = df['booking_status'].map({'Not_Canceled': 0, 'Canceled': 1})

    return train_df, test_df

def save_transformed_data(train_df: pd.DataFrame, test_df: pd.DataFrame, output_path=PROCESSED_DIR):
    """Saves transformed data in the processed folder."""
    os.makedirs(output_path, exist_ok=True)

    train_df.to_csv(os.path.join(output_path, "train_transformed.csv"), index=False)
    test_df.to_csv(os.path.join(output_path, "test_transformed.csv"), index=False)
    
    print("✅ Transformed data saved successfully!")

def main():
    """Runs the data transformation pipeline."""
    try:
        train_df, test_df = load_data()  # ✅ No need to pass `data_path` manually
        train_transformed, test_transformed = transform_data(train_df, test_df)
        save_transformed_data(train_transformed, test_transformed)
        print("✅ Data transformation completed successfully!")
    except Exception as e:
        print(f"❌ Error in data transformation: {e}")

if __name__ == "__main__":
    main()
