"""Car Price Prediction: Streamlit app.

Run locally:   streamlit run app.py
The model and the dropdown options are made by notebooks/car_price.ipynb (models/ folder).
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from car_price import REF_YEAR  # also makes the pickled pipeline's helpers importable

HERE = Path(__file__).parent
MODEL_PATH = HERE / "models" / "car_price_model.joblib"
OPTIONS_PATH = HERE / "models" / "app_options.json"

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_options():
    return json.loads(OPTIONS_PATH.read_text(encoding="utf-8"))


model = load_model()
opt = load_options()
rng = opt["ranges"]

st.title("🚗 Used Car Price Prediction")
st.write(
    "Estimate the price of a used car in India from its details. "
    f"The model was trained on {opt['n_train']:,} cars listed up to {REF_YEAR - 1}; "
    "prices are in **lakh** (1 lakh = ₹100,000)."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("The car")
    brand = st.selectbox("Brand", opt["brands"], index=opt["brands"].index(opt["default"]["Brand"]))
    models_for_brand = opt["models_by_brand"][brand]
    default_model = opt["default"]["Model"] if brand == opt["default"]["Brand"] else models_for_brand[0]
    car_model = st.selectbox("Model", models_for_brand, index=models_for_brand.index(default_model))
    year = st.slider("Year of manufacture", rng["Year"][0], rng["Year"][1], opt["default"]["Year"])
    fuel_type = st.selectbox("Fuel type", opt["fuel_types"], index=opt["fuel_types"].index(opt["default"]["Fuel_Type"]))
    transmission = st.radio("Transmission", opt["transmissions"], horizontal=True,
                            index=opt["transmissions"].index(opt["default"]["Transmission"]))

with col2:
    st.subheader("Engine")
    engine = st.number_input("Engine capacity (cc)", rng["Engine"][0], rng["Engine"][1], opt["default"]["Engine"], step=50)
    power = st.number_input("Power (bhp)", rng["Power"][0], rng["Power"][1], opt["default"]["Power"], step=5.0)
    mileage = st.number_input("Mileage (km per litre of petrol)", rng["Mileage"][0], rng["Mileage"][1],
                              opt["default"]["Mileage"], step=0.5,
                              help="For CNG/LPG cars, enter the petrol-equivalent value (km/kg × 0.69).")

with col3:
    st.subheader("History")
    location = st.selectbox("City", opt["locations"], index=opt["locations"].index(opt["default"]["Location"]))
    km_driven = st.number_input("Kilometers driven", rng["Kilometers_Driven"][0], rng["Kilometers_Driven"][1],
                                opt["default"]["Kilometers_Driven"], step=1000)
    owner_type = st.selectbox("Owner", opt["owner_types"], index=0)

# Same columns the model was trained on: raw inputs, before feature engineering
car = pd.DataFrame([{
    "Location": location, "Year": year, "Kilometers_Driven": km_driven, "Fuel_Type": fuel_type,
    "Transmission": transmission, "Owner_Type": owner_type, "Mileage": mileage, "Engine": engine,
    "Power": power, "Brand": brand, "Model": car_model,
}])[opt["input_columns"]]

st.divider()
if st.button("Predict price", type="primary", use_container_width=True):
    price = float(model.predict(car)[0])
    mae = opt["test_mae"]
    low, high = max(price - mae, 0.0), price + mae
    st.metric("Estimated price", f"₹ {price:,.2f} lakh", help="1 lakh = ₹100,000")
    st.write(f"≈ ₹ {price * 100_000:,.0f}. On unseen test cars the model is off by about "
             f"₹ {mae:.2f} lakh on average (MAE), so a fair range is **₹ {low:,.2f} – {high:,.2f} lakh**.")
    with st.expander("What the model received"):
        st.dataframe(car, hide_index=True)

st.caption("Zero Grad · Machine Learning course project · "
           "data: Kaggle Used Cars Price Prediction (India)")
