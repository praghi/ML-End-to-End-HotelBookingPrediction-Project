import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle 
from sklearn.preprocessing import LabelEncoder , StandardScaler
#from pymongo.mongo_client import MongoClient
#from pymongo.server_api import ServerApi
#from urllib.parse import quote_plus
import os 
from dotenv import load_dotenv 


def load_model():
    # Get the parent directory (main directory)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one level
    model_path = os.path.join(base_dir, "model.pkl")  # Use "model.pkl" in the main directory
    scaler_path = os.path.join(base_dir,"scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"🚨 Model file '{model_path}' not found!")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"🚨 Scaler file '{scaler_path}' not found!")

    # Load model
    with open(model_path, 'rb') as f_model:
        xgb_model = pickle.load(f_model)

    # Load scaler
    with open(scaler_path, 'rb') as f_scaler:
        scaler = pickle.load(f_scaler)

    return xgb_model, scaler


def preprocessing_input_data(data, scaler):
    categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
    
    df = pd.DataFrame([data])
    
    # Scale numerical columns
    df[['lead_time', 'avg_price_per_room']] = scaler.transform(df[['lead_time', 'avg_price_per_room']])
    
    # One-hot encoding
    df = pd.get_dummies(df, columns=categorical_cols)

    # 5️⃣ Expected columns (24 in total)
    expected_columns = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 
                    'required_car_parking_space', 'lead_time', 'repeated_guest', 'no_of_previous_cancellations', 
                    'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests', 
                    'type_of_meal_plan_Meal Plan 2', 'type_of_meal_plan_Meal Plan 3', 'type_of_meal_plan_Not Selected', 
                    'room_type_reserved_Room_Type 2', 'room_type_reserved_Room_Type 3', 'room_type_reserved_Room_Type 4', 
                    'room_type_reserved_Room_Type 5', 'room_type_reserved_Room_Type 6', 'room_type_reserved_Room_Type 7', 
                    'market_segment_type_Complementary', 'market_segment_type_Corporate', 'market_segment_type_Offline', 
                    'market_segment_type_Online']

# 6️⃣ Add missing columns with 0 if they are not present in the user data
    missing_cols = set(expected_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0

# 7️⃣ Reorder the columns to match the expected order
    df = df[expected_columns]
# Set pandas to display all rows if needed
    pd.set_option('display.max_columns', None)  # Show all columns 

    return df


def predict_data(data):
    xgb_model, scaler = load_model()
    processed_data = preprocessing_input_data(data, scaler)
    print(processed_data.head(5))
    prediction = xgb_model.predict(processed_data)
    if prediction[0] == 0 :
        return 'Not Cancelled'
    else:
        return 'Cancelled'
           
def main(): 
    st.title("🔮 Housing Booking Prediction")
    st.write('Enter your data to get a prediction for housing booking.')

    # User inputs
    no_of_adults = st.number_input("No Of Adults", min_value=1, max_value=10, value=1)
    no_of_children = st.number_input("No Of Children", min_value=0, max_value=15, value=0)
    no_of_weekend_nights = st.number_input("No Of Weekend Nights", min_value=0, max_value=30, value=2)
    no_of_week_nights = st.number_input("No Of Week Nights", min_value=0, max_value=30, value=3)
    type_of_meal_plan = st.selectbox("Type Of Meal Plan", ['Meal Plan 1', 'Not Selected', 'Meal Plan 2', 'Meal Plan 3'])
    required_car_parking_space = st.number_input("Required Car Parking Space", min_value=0, max_value=5, value=0)
    room_type_reserved = st.selectbox("Room Type Reserved", ['Room_Type 1', 'Room_Type 4', 'Room_Type 2', 'Room_Type 6', 'Room_Type 5', 'Room_Type 7', 'Room_Type 3'])
    lead_time = st.number_input("Lead Time", min_value=0, max_value=800, value=90)
    market_segment_type = st.selectbox("Market Segment Type", ['Offline', 'Online', 'Corporate', 'Aviation', 'Complementary'])
    repeated_guest = st.number_input("Repeated Guest", min_value=0, max_value=1, value=0)
    no_of_previous_cancellations = st.number_input("No Of Previous Cancellations", min_value=0, max_value=20, value=0)
    no_of_previous_bookings_not_canceled = st.number_input("No Of Previous Bookings Not Canceled", min_value=0, max_value=60, value=0)
    avg_price_per_room= st.number_input("Avg Price Per Room", min_value=0, max_value=800, value=200)
    no_of_special_requests = st.number_input("No Of Special Requests", min_value=0, max_value=10, value=0)

    # Prediction button
    if st.button("🔮 Predict Your Score"):
        user_data = {
            'no_of_adults': no_of_adults, 
            'no_of_children': no_of_children, 
            'no_of_weekend_nights': no_of_weekend_nights, 
            'no_of_week_nights': no_of_week_nights,
            'type_of_meal_plan': type_of_meal_plan, 
            'required_car_parking_space': required_car_parking_space,
            'room_type_reserved': room_type_reserved, 
            'lead_time': lead_time, 
            'market_segment_type': market_segment_type, 
            'repeated_guest': repeated_guest, 
            'no_of_previous_cancellations': no_of_previous_cancellations,
            'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled, 
            'avg_price_per_room': avg_price_per_room, 
            'no_of_special_requests': no_of_special_requests
        }
        
        prediction = predict_data(user_data)
        st.success(f'🏠 Prediction Result: {prediction}')

# Run the app
if __name__ == "__main__":
    main()
