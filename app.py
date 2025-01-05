# Gerekli kütüphaneler
import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükleme
st.title("California Housing Price Prediction")
st.write("Welcome to the Housing Price Prediction App!")

data_load_state = st.text("Loading dataset...")
housing = fetch_california_housing(as_frame=True)
data = housing.data
data['target'] = housing.target
data_load_state.text("Dataset loaded successfully!")

# Step 1: Explore the Dataset
st.header("1. Explore the Dataset")
if st.checkbox("Show raw data"):
    st.write(data.head())

# Step 2: Analyze the Data
st.header("2. Analyze the Data")
st.subheader("Dataset Dimensions")
st.write(f"Number of rows: {data.shape[0]}")
st.write(f"Number of columns: {data.shape[1]}")

if st.checkbox("Show summary statistics"):
    st.subheader("Summary Statistics")
    st.write(data.describe())

if st.checkbox("Show correlation matrix"):
    st.subheader("Correlation Matrix")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Step 3: Train the Model
st.header("3. Train the Model")
st.write("Train a Linear Regression model to predict median house values.")

# Özellikler ve hedef değişkeni ayırma
X = data.drop("target", axis=1)
y = data["target"]

test_size = st.slider("Select test size (percentage):", 10, 50, 20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

st.write(f"Training set size: {X_train.shape[0]} rows")
st.write(f"Test set size: {X_test.shape[0]} rows")

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**R-squared (R²):** {r2:.2f}")

# Step 4: Make Predictions
st.header("4. Make a Prediction")
st.write("Provide inputs for each feature to predict the median house value:")

input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"Enter value for {col}:", value=float(data[col].mean()))

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    st.subheader("Predicted Median House Value")
    st.write(f"The predicted median house value is **${prediction[0] * 1000:,.2f}**.")


