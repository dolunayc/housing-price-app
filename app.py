import streamlit as st

st.title("California Housing Price Prediction")
st.write("Welcome to the Housing Price Prediction App!")

# Kullanıcıdan veri alma
user_input = st.number_input("Enter a value for testing:", value=0)
st.write(f"You entered: {user_input}")

import pandas as pd
from sklearn.datasets import fetch_california_housing
import ssl

# SSL doğrulamasını devre dışı bırak
ssl._create_default_https_context = ssl._create_unverified_context

# Veri setini yükleme
st.header("Step 1: Load the Dataset")
data_load_state = st.text("Loading dataset...")
housing = fetch_california_housing(as_frame=True)  # Veri setini yükle
data = housing.data
data['target'] = housing.target  # Hedef değişkeni ekle
data_load_state.text("Dataset loaded successfully!")  # Yükleme mesajını güncelle

# Veri setini görüntüleme
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(data.head())  # İlk 5 satırı göster

import seaborn as sns
import matplotlib.pyplot as plt

# Korelasyon matrisi
st.subheader("Correlation Matrix")
if st.checkbox("Show correlation matrix"):
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Hedef değişken analizi
st.subheader("Feature vs Target Relationship")
feature = st.selectbox("Select a feature to compare with target:", data.columns[:-1])  # Hedef hariç sütunlar
fig, ax = plt.subplots()
sns.scatterplot(x=data[feature], y=data["target"], ax=ax)
ax.set_xlabel(feature)
ax.set_ylabel("Target (Median House Value)")
ax.set_title(f"{feature} vs Target")
st.pyplot(fig)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Veri ön işleme
st.header("Step 2: Data Preprocessing")
X = data.drop("target", axis=1)  # Özellikler
y = data["target"]  # Hedef değişken
test_size = st.slider("Select test size (percentage):", 10, 50, 20)  # Test seti oranı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
st.write(f"Training set size: {X_train.shape[0]} rows")
st.write(f"Test set size: {X_test.shape[0]} rows")

# Model eğitimi
st.header("Step 3: Train a Linear Regression Model")
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R-squared (R²): {r2:.2f}")

# Tahmin yapma
st.header("Step 4: Make Predictions")
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"Enter value for {col}:", value=float(data[col].mean()))

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    st.write(f"Predicted Median House Value: {prediction[0]:.2f}")

