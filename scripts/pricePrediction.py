import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from keras.models import load_model

# Define file paths for the trained model, scaler, and one-hot encoders
MODEL_PATH = '../models/trained_model.h5'
SCALER_PATH = '../models/trained_scaler.pkl'
OHE_FEATURES_PATH = '../models/ohe_features.pkl'
OHE_FUEL_PATH = '../models/ohe_fuel.pkl'

# Load the trained Keras model, scaler, and one-hot encoded features and fuel data
loaded_model = load_model(MODEL_PATH)
loaded_scaler = joblib.load(SCALER_PATH)
ohe_features_df = joblib.load(OHE_FEATURES_PATH)
ohe_fuel_df = joblib.load(OHE_FUEL_PATH)
MODEL_OHE_COLUMNS = [
    column for column in ohe_features_df.columns
    if column.startswith('model_') and column != 'model_year'
]
FUEL_OHE_COLUMNS = [
    column for column in ohe_fuel_df.columns
    if column.startswith('fuel_')
]

# Function to predict car price
def predict_car_price(data):
    current_year = int(datetime.now().strftime("%Y"))

    # Calculate the age of the car
    age = current_year - float(data['year']) + 1

    # Scale the kilometers driven using the loaded scaler
    scaled_km = loaded_scaler.transform(pd.DataFrame({'km': [float(data['km'])]}))

    # Encode transmission 
    encoded_transmission = 1 if data['transmission'].lower() == 'automatic' else 0

    # Get the brand rank from the one-hot encoded features
    brand_rank = ohe_features_df.query(f"brand == '{data['brand']}'")['brand_rank'].unique()

    # Get the one-hot encoded model and fuel data
    model_ohe = (
        ohe_features_df
        .query(f"brand == '{data['brand']}' and model == '{data['model']}'")
        .drop_duplicates(subset='model')[MODEL_OHE_COLUMNS]
        .values
    )
    fuel_ohe = (
        ohe_fuel_df
        .query(f"fuel == '{data['fuel'].lower()}'")[FUEL_OHE_COLUMNS]
        .values
    )

    if brand_rank.size == 0 or model_ohe.size == 0 or fuel_ohe.size == 0:
        raise ValueError("Not enough encoded data to build a prediction for the requested vehicle.")

    # Prepare the feature input for the model
    features = np.array([[age, scaled_km[0][0], encoded_transmission]])
    model_input = np.hstack((features, brand_rank.reshape(1, -1), model_ohe, fuel_ohe))
    model_input = model_input.astype('float32')

    # Make the car price prediction using the loaded model
    predicted_price = loaded_model.predict(model_input)

    # Convert the predicted price back to the original scale
    return str(np.expm1(predicted_price)[0][0])
