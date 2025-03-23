import os
import pandas as pd
# Automatically detect project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")

def load_transformed_data(data_path=PROCESSED_DIR):
    """Loads transformed train and test data from processed directory."""
    train_path = os.path.join(data_path, "train_transformed.csv")
    test_path = os.path.join(data_path, "test_transformed.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Error: Transformed data files are missing!")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df

def feature_engineering(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Performs feature selection and engineering."""
    
    # Drop irrelevant columns
    cols_to_drop = ['booking_id', 'arrival_year', 'arrival_month', 'arrival_date']
    train_df = train_df.drop(columns=[col for col in cols_to_drop if col in train_df.columns], errors='ignore')
    test_df = test_df.drop(columns=[col for col in cols_to_drop if col in test_df.columns], errors='ignore')

    # Convert categorical features to strings (ensuring uniformity)
    categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
    for col in categorical_cols:
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)

    # One-Hot Encoding for categorical features
    train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
    test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

    # Ensure train and test have the same columns
    missing_cols = set(train_df.columns) - set(test_df.columns)
    for col in missing_cols:
        test_df[col] = 0  # Add missing columns in test_df with 0 values

    test_df = test_df[train_df.columns]  # Reorder columns to match train_df

    # Handle missing values after encoding
    train_df.fillna(train_df.median(), inplace=True)
    test_df.fillna(test_df.median(), inplace=True)

    return train_df, test_df


def save_features(train_df: pd.DataFrame, test_df: pd.DataFrame, output_path=FEATURES_DIR):
    """Saves feature-selected data in the features directory."""
    os.makedirs(output_path, exist_ok=True)

    train_df.to_csv(os.path.join(output_path, "train_features.csv"), index=False)
    test_df.to_csv(os.path.join(output_path, "test_features.csv"), index=False)
    
    print("✅ Feature-selected data saved successfully!")

def main():
    """Runs the feature selection pipeline."""
    try:
        train_df, test_df = load_transformed_data()
        train_features, test_features = feature_engineering(train_df, test_df)
        save_features(train_features, test_features)
        print("✅ Feature engineering completed successfully!")
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")

if __name__ == "__main__":
    main()


