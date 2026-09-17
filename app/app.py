import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sydney Housing Price Predictor", page_icon="\U0001f3e0")

st.title("Sydney Housing Price Predictor")
st.write(
    "Enter a property's features below to get a predicted sale price, "
    "based on a machine learning model trained on 165 sold properties in "
    "Mosman, Parramatta and Blacktown."
)


@st.cache_resource
def load_model():
    return joblib.load("trained_model.joblib")


try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.warning(
        "No trained model found yet. Run the notebook through Part 6 first "
        "to generate `trained_model.joblib` in this folder."
    )

STANDALONE_FEATURES = [
    ("feat_built_in_wardrobes", "Built-in wardrobes"),
    ("feat_close_to_shops", "Close to shops"),
    ("feat_close_to_schools", "Close to schools"),
    ("feat_close_to_transport", "Close to transport"),
    ("feat_air_conditioning", "Air conditioning"),
    ("feat_secure_parking", "Secure parking"),
    ("feat_dishwasher", "Dishwasher"),
    ("feat_intercom", "Intercom"),
    ("feat_floorboards", "Floorboards"),
    ("feat_balcony", "Balcony"),
]

COMPOSITE_FEATURES = [
    "feat_has_view",
    "feat_has_extra_bathroom_comfort",
    "feat_has_outdoor_living",
    "feat_has_extra_climate_control",
    "feat_has_study_or_extra_room",
    "feat_has_security_features",
    "feat_has_extra_parking",
    "feat_has_leisure_amenities",
    "feat_has_lifestyle_extras",
    "feat_has_utility_features",
    "feat_fully_fenced",
    "feat_ensuite",
    "feat_split_system_air_con",
    "feat_outdoor_entertainment",
]

SUBURB_CBD_KM = {"Mosman": 8, "Parramatta": 23, "Blacktown": 34}

with st.form("property_form"):
    col1, col2 = st.columns(2)

    with col1:
        suburb = st.selectbox("Suburb", ["Mosman", "Parramatta", "Blacktown"])
        property_type = st.selectbox(
            "Property type",
            [
                "House",
                "Apartment / Unit / Flat",
                "Townhouse",
                "Semi-Detached",
                "Studio",
            ],
        )
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)
        carspaces = st.number_input("Car spaces", min_value=0, max_value=10, value=1)

    with col2:
        distance_to_cbd_km = st.number_input(
            "Distance to CBD (km)",
            min_value=0.0,
            value=float(SUBURB_CBD_KM["Mosman"]),
            help="Defaults update per suburb below - override if you know the exact distance.",
        )
        st.caption("Select property features present:")
        feature_values = {}
        for col, label in STANDALONE_FEATURES:
            feature_values[col] = st.checkbox(label)

    submitted = st.form_submit_button("Predict price")

if submitted:
    if not model_loaded:
        st.error(
            "Cannot predict — train and save the model first (see notebook Part 6)."
        )
    else:
        row = {
            "suburb": suburb,
            "property_type": property_type,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "carspaces": carspaces,
            "distance_to_cbd_km": distance_to_cbd_km,
        }
        row.update({col: int(val) for col, val in feature_values.items()})
        row.update({col: 0 for col in COMPOSITE_FEATURES})

        input_df = pd.DataFrame([row])
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted sale price: ${prediction:,.0f}")
        st.caption(
            "This is a model estimate based on limited training data — treat it as a "
            "guide, not a valuation."
        )

st.divider()
st.caption(
    "Built for the Sydney Housing Price Prediction ML mini project. "
    "Model: see notebooks/sydney_housing_project.ipynb for training details."
)
