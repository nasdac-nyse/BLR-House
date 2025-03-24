import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
file_path = "bengaluru_house_prices.csv"
df = pd.read_csv(file_path)

# Drop unnecessary columns
df = df.drop(columns=['area_type', 'availability', 'society'])

# Handle missing values
df = df.dropna(subset=['location'])
df['bath'].fillna(df['bath'].median(), inplace=True)
df['balcony'].fillna(df['balcony'].median(), inplace=True)
df['BHK'] = df['size'].str.extract('(\d+)').astype(float)
df = df.drop(columns=['size'])

def convert_sqft_to_num(sqft):
    try:
        if '-' in sqft:
            vals = list(map(float, sqft.split('-')))
            return (vals[0] + vals[1]) / 2
        return float(sqft)
    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
df = df.dropna(subset=['total_sqft'])
df['total_sqft'] = df['total_sqft'].astype(float)

# Feature Engineering: Create price per sqft
df['price_per_sqft'] = df['price'] / df['total_sqft']

# Remove extreme outliers (keeping data within 1st and 99th percentile)
low, high = df['price_per_sqft'].quantile([0.01, 0.99])
df = df[(df['price_per_sqft'] > low) & (df['price_per_sqft'] < high)]

# Encode 'location' using Label Encoding
label_encoder = LabelEncoder()
df['location'] = label_encoder.fit_transform(df['location'])

# Define features (X) and target (y)
X = df.drop(columns=['price'])
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train[['total_sqft', 'bath', 'balcony', 'BHK', 'location', 'price_per_sqft']] = scaler.fit_transform(X_train[['total_sqft', 'bath', 'balcony', 'BHK', 'location', 'price_per_sqft']])
X_test[['total_sqft', 'bath', 'balcony', 'BHK', 'location', 'price_per_sqft']] = scaler.transform(X_test[['total_sqft', 'bath', 'balcony', 'BHK', 'location', 'price_per_sqft']])

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Function to predict price based on user input
def predict_price(total_sqft, bath, balcony, BHK, location):
    if location in label_encoder.classes_:
        location_encoded = label_encoder.transform([location])[0]
    else:
        location_encoded = -1  # Assign an unknown location value
    
    input_data = np.array([[total_sqft, bath, balcony, BHK, location_encoded, total_sqft / BHK]])
    input_scaled = scaler.transform(input_data)
    predicted_price = model.predict(input_scaled)[0]
    return f"Predicted House Price: {predicted_price:.2f} Lakhs"

# Streamlit UI
st.title("Bangalore House Price Prediction")

# User Inputs
total_sqft = st.number_input("Total Square Feet", min_value=300, max_value=10000, value=1200)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
balcony = st.number_input("Number of Balconies", min_value=0, max_value=5, value=1)
BHK = st.number_input("Number of BHK", min_value=1, max_value=10, value=2)
location = st.text_input("Enter Location")

# Predict button
if st.button("Predict Price"):
    price = predict_price(total_sqft, bath, balcony, BHK, location)
    st.success(price)
