import pickle
import pandas as pd

# Load model and scaler
def load_model_and_scaler():
    with open(r"D:\Python\Eron Learning\ML Projects\ML-End-to-End-HotelBookingPrediction-Project\model.pkl", "rb") as file:
        xgb_model = pickle.load(file)
    
    with open(r"D:\Python\Eron Learning\ML Projects\ML-End-to-End-HotelBookingPrediction-Project\scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    
    return xgb_model, scaler

# Preprocessing input data
def preprocess_data(data, scaler):
    df = pd.DataFrame([data])
    df[['lead_time', 'avg_price_per_room']] = scaler.transform(df[['lead_time', 'avg_price_per_room']])
    df = pd.get_dummies(df, columns=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type'])
    return df

# Predict using the model
def predict(data):
    xgb_model, scaler = load_model_and_scaler()
    processed_data = preprocess_data(data, scaler)
    print(processed_data)
    return xgb_model.predict(processed_data)

# Test input
user_data = {
    'no_of_adults': 2, 
    'no_of_children': 1, 
    'no_of_weekend_nights': 2, 
    'no_of_week_nights': 3,
    'type_of_meal_plan': 'Meal Plan 1', 
    'required_car_parking_space': 1,
    'room_type_reserved': 'Room_Type 2', 
    'lead_time': 50, 
    'market_segment_type': 'Online', 
    'repeated_guest': 1, 
    'no_of_previous_cancellations': 1,
    'no_of_previous_bookings_not_canceled': 2, 
    'avg_price_per_room': 150, 
    'no_of_special_requests': 1
}

# Get prediction
prediction = predict(user_data)
print(f"Prediction: {prediction}")
