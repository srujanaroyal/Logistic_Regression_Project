import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------
# Load Dataset
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "linear_regression_datasest.csv")

df = pd.read_csv(csv_path)

# ----------------------------
# Features & Target
# ----------------------------
X = df[['Feature_1', 'Feature_2']]
y = df['Target']

# ----------------------------
# Train Model
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# Evaluation
# ----------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Linear Regression App")

st.title("📈 Linear Regression Prediction App")

st.write("Model Performance")
st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# ----------------------------
# User Input
# ----------------------------
feature1 = st.number_input(
    "Enter Feature 1",
    value=5.0
)

feature2 = st.number_input(
    "Enter Feature 2",
    value=10.0
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict"):

    input_data = pd.DataFrame({
        "Feature_1": [feature1],
        "Feature_2": [feature2]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Target Value: {prediction[0]:.2f}"
    )