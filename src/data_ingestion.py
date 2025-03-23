import numpy as np
import pandas as pd
import os 
from sklearn.model_selection import train_test_split
import yaml

def load_data(filepath:str)-> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        return df 
    except pd.errors.ParserError as e:
        print(f"Error: Failed to parse the CSV file from {filepath}.")
        print(e)
        raise
    except Exception as e:
        print(f"Error: An unexpected error occured while loading the data.")
        print(e)
        raise

def preprocess_data(df:pd.DataFrame)-> pd.DataFrame:
    try: 
        print("Number of rows and columns:", df.shape)

       # Display the first few rows of the dataset to get an overview
        print("First few rows of the dataset:")
        print(df.head(5))
        df = df
        return df
    except KeyError as e:
        print(f"Error:Missing column {e} in the dataframe.")
        raise
    except Exception as e:
        print(f"Error: An unexpected error occured during preprocessing")
        print(e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        # Get the absolute path of the main project directory (parent of 'src')
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Set the data path to the main directory
        data_path = os.path.join(project_dir, data_path, 'raw')
        os.makedirs(data_path, exist_ok=True)

        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)

        print(f"Data saved successfully at: {data_path}")

    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the data")
        print(e)
        raise

def main():
    try:
        df = load_data(filepath="src/hotel_reservations.csv")
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df,test_size=0.20,random_state=42)
        save_data(train_data,test_data,data_path='data')
        print('Data Ingestion part completed successfully')
    except Exception as e:
        print(f"Error:{e}")
        print("Failed to complete the data ingestion")


if __name__=='__main__':
    main()