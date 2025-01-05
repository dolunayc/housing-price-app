import streamlit as st

st.title("California Housing Price Prediction")
st.write("Welcome to the Housing Price Prediction App!")

# Kullanıcıdan veri alma
user_input = st.number_input("Enter a value for testing:", value=0)
st.write(f"You entered: {user_input}")


