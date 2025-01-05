# Gerekli kütüphaneleri yükleyelim
import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk

# Veri setini yükleyelim
st.title("California Housing Price Prediction")
st.write("Welcome to the Housing Price Prediction App!")

data_load_state = st.text("Loading dataset...")
housing = fetch_california_housing(as_frame=True)
data = housing.data
data['target'] = housing.target

# Sütun isimlerini düzenleme
data.rename(columns={
    'MedInc': 'median_income',
    'HouseAge': 'housing_median_age',
    'AveRooms': 'total_rooms',
    'AveBedrms': 'total_bedrooms',
    'AveOccup': 'average_occupancy',
    'Latitude': 'latitude',
    'Longitude': 'longitude',
    'Population': 'population'
}, inplace=True)

data_load_state.text("Dataset loaded successfully!")

# Veri setini kontrol edelim
if st.checkbox("Show raw data"):
    st.write(data.head())

# Kullanıcının kriterlerine göre mahalle öneri sistemi
st.subheader("Find the Best Neighborhood for You")
budget = st.number_input("Enter your budget ($):", value=300000)
min_rooms = st.slider("Minimum number of rooms:", 1, 10, 3)
max_population = st.slider("Maximum population:", 100, 5000, 1000)

# Kullanıcı kriterlerine göre filtreleme
if st.button("Find Neighborhoods"):
    filtered_data = data[
        (data['target'] * 1000 <= budget) & 
        (data['total_rooms'] >= min_rooms) & 
        (data['population'] <= max_population)
    ]
    
    if not filtered_data.empty:
        st.write("Recommended Neighborhoods:")
        st.write(filtered_data[['longitude', 'latitude', 'target', 'total_rooms', 'population']].head(10))
        
        # Filtrelenmiş mahalleleri haritada gösterme
        map_data = pd.DataFrame({
            'lat': filtered_data['latitude'],
            'lon': filtered_data['longitude'],
            'target': filtered_data['target']
        })
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=map_data['lat'].mean(),
                longitude=map_data['lon'].mean(),
                zoom=6,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    'ScatterplotLayer',
                    data=map_data,
                    get_position='[lon, lat]',
                    get_color='[200, 30, 0, 160]',
                    get_radius=1000,
                ),
            ],
        ))
    else:
        st.write("No neighborhoods match your criteria. Try adjusting the filters!")

# Veri analizi yapalım
if st.checkbox("Show summary statistics"):
    st.write(data.describe())

if st.checkbox("Show correlation matrix"):
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Model eğitimi
st.header("Train and Predict")
X = data.drop("target", axis=1)
y = data["target"]

# Veri setini ayıralım
test_size = st.slider("Select test size (percentage):", 10, 50, 20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

# Kullanıcı model seçimi
model_choice = st.selectbox("Choose a model:", ["Linear Regression", "Random Forest"])
if model_choice == "Linear Regression":
    model = LinearRegression()
else:
    model = RandomForestRegressor()

model.fit(X_train, y_train)

# Performans değerlendirme
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.subheader("Model Performance")
st.write(f"**Model:** {model_choice}")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**R-squared (R²):** {r2:.2f}")

# Kullanıcı girdisi ve tahmin
st.subheader("Make a Prediction")
st.write("Provide values for the following features to predict the median house value:")

input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"Enter value for {col}:", value=float(data[col].mean()))

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    st.write(f"The predicted median house value is **${prediction[0] * 1000:,.2f}**.")

# Gelişmiş görselleştirmeler
st.subheader("Advanced Visualizations")

# Görselleştirme: Prediction vs Actual
if st.checkbox("Visualize Prediction vs Actual Data"):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
    ax.set_xlabel("Actual Median House Value")
    ax.set_ylabel("Predicted Median House Value")
    ax.set_title("Prediction vs Actual")
    st.pyplot(fig)

# Özellik önemi (Feature Importance)
if model_choice == "Random Forest" and st.checkbox("Show Feature Importance"):
    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(feature_importance)

# Coğrafi görselleştirme
if 'latitude' in data.columns and 'longitude' in data.columns:
    if st.checkbox("Geographical View of Neighborhoods"):
        map_data = pd.DataFrame({
            'lat': data['latitude'],
            'lon': data['longitude'],
            'target': data['target']
        })
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=map_data['lat'].mean(),
                longitude=map_data['lon'].mean(),
                zoom=6,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    'ScatterplotLayer',
                    data=map_data,
                    get_position='[lon, lat]',
                    get_color='[200, 30, 0, 160]',
                    get_radius=1000,
                ),
            ],
        ))
else:
    st.error("Geographical columns ('latitude' and 'longitude') are missing!")

