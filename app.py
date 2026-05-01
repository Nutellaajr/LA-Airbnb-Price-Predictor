from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st


st.set_page_config(
    page_title="LA Airbnb Price Predictor",
    layout="wide",
)


BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "feature_and_modeling"
MODEL_PATH = MODEL_DIR / "final_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"
FEATURE_DATA_PATH = MODEL_DIR / "data_with_features.csv"
RAW_DATA_PATH = MODEL_DIR / "listings_clean.csv"

REFERENCE_DATE = pd.to_datetime("2024-09-04")

LANDMARKS = {
    "hollywood": (34.0928, -118.3287),
    "beverly_hills": (34.0736, -118.4004),
    "santa_monica_beach": (34.0195, -118.4912),
    "downtown_la": (34.0522, -118.2437),
    "lax_airport": (33.9416, -118.4085),
    "venice_beach": (33.9850, -118.4695),
}

NUMERICAL_FEATURES = [
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "review_scores_rating",
    "host_response_rate",
    "latitude",
    "longitude",
    "beds_per_bedroom",
    "bath_per_person",
    "room_density",
    "host_tenure_days",
    "host_tenure_years",
    "host_join_year",
    "host_join_month",
    "dist_to_hollywood",
    "dist_to_beverly_hills",
    "dist_to_santa_monica_beach",
    "dist_to_downtown_la",
    "dist_to_lax_airport",
    "dist_to_venice_beach",
    "min_dist_to_landmark",
]

CATEGORICAL_FEATURES = [
    "host_response_time",
    "neighbourhood_cleansed",
    "neighbourhood_group_cleansed",
    "room_type",
    "property_type_grouped",
    "license_status",
]

BOOLEAN_FEATURES = [
    "host_is_superhost",
    "instant_bookable",
    "host_never_responded",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES


@st.cache_resource
def load_model_assets():
    missing = [path for path in [MODEL_PATH, PREPROCESSOR_PATH] if not path.exists()]
    if missing:
        return None, None, missing

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor, []


@st.cache_data
def load_listing_data():
    data_path = FEATURE_DATA_PATH if FEATURE_DATA_PATH.exists() else RAW_DATA_PATH
    if not data_path.exists():
        return pd.DataFrame()

    data = pd.read_csv(data_path)
    for col in ["price", "latitude", "longitude", "bedrooms", "bathrooms", "accommodates"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


@st.cache_data
def load_feature_importance():
    if not FEATURE_IMPORTANCE_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


def clean_options(series):
    values = series.dropna().astype(str).sort_values().unique().tolist()
    return values if values else ["Unknown"]


def most_common_value(data, column, fallback):
    if column not in data.columns or data[column].dropna().empty:
        return fallback
    return data[column].mode(dropna=True).iloc[0]


def median_value(data, column, fallback):
    if column not in data.columns:
        return fallback
    value = pd.to_numeric(data[column], errors="coerce").median()
    return fallback if pd.isna(value) else value


def haversine(lat1, lon1, lat2, lon2):
    radius_km = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(a))


def known_grouped_property_types(data):
    if "property_type_grouped" in data.columns:
        return set(data["property_type_grouped"].dropna().astype(str).unique())
    if "property_type" in data.columns:
        type_counts = data["property_type"].dropna().astype(str).value_counts()
        return set(type_counts[type_counts >= 20].index)
    return set()


def engineer_listing_features(raw_input, grouped_property_types):
    features = pd.DataFrame([raw_input])

    features["host_since"] = pd.to_datetime(features["host_since"], errors="coerce")
    features["beds_per_bedroom"] = features["beds"] / features["bedrooms"].replace(0, 1)
    features["bath_per_person"] = features["bathrooms"] / features["accommodates"].replace(0, np.nan)
    features["room_density"] = features["accommodates"] / features["beds"].replace(0, np.nan)

    features["host_tenure_days"] = (REFERENCE_DATE - features["host_since"]).dt.days
    features["host_tenure_years"] = features["host_tenure_days"] / 365
    features["host_join_year"] = features["host_since"].dt.year
    features["host_join_month"] = features["host_since"].dt.month

    property_type = str(raw_input["property_type"])
    if property_type in grouped_property_types:
        features["property_type_grouped"] = property_type
    else:
        features["property_type_grouped"] = "Other"

    dist_cols = []
    for name, (lat, lon) in LANDMARKS.items():
        col = f"dist_to_{name}"
        features[col] = haversine(features["latitude"], features["longitude"], lat, lon)
        dist_cols.append(col)
    features["min_dist_to_landmark"] = features[dist_cols].min(axis=1)

    for col in BOOLEAN_FEATURES:
        features[col] = features[col].astype(float)

    return features[ALL_FEATURES]


def dollar_prediction(model, preprocessor, listing_features):
    processed = preprocessor.transform(listing_features)
    predicted_log_price = model.predict(processed)[0]
    return max(0, float(np.expm1(predicted_log_price)))


def prepare_map_data(data, selected_room_types, price_range, selected_groups, bedroom_range, max_points):
    required = ["latitude", "longitude", "price", "room_type", "neighbourhood_cleansed"]
    if data.empty or any(col not in data.columns for col in required):
        return pd.DataFrame()

    map_data = data.dropna(subset=["latitude", "longitude", "price"]).copy()

    if selected_room_types:
        map_data = map_data[map_data["room_type"].isin(selected_room_types)]
    map_data = map_data[map_data["price"].between(price_range[0], price_range[1])]

    if "neighbourhood_group_cleansed" in map_data.columns and selected_groups:
        map_data = map_data[map_data["neighbourhood_group_cleansed"].isin(selected_groups)]

    if "bedrooms" in map_data.columns:
        map_data = map_data[map_data["bedrooms"].between(bedroom_range[0], bedroom_range[1])]

    if len(map_data) > max_points:
        map_data = map_data.sample(max_points, random_state=42)

    log_price = np.log1p(map_data["price"].clip(lower=0))
    denom = log_price.max() - log_price.min()
    if pd.notna(denom) and denom != 0:
        normalized = (log_price - log_price.min()) / denom
    else:
        normalized = pd.Series(0.5, index=map_data.index)
    map_data["color"] = normalized.apply(
        lambda x: [int(60 + 190 * x), int(150 - 70 * x), int(220 - 160 * x), 155]
    )
    map_data["price_label"] = map_data["price"].map(lambda x: f"${x:,.0f}")

    for col in ["bedrooms", "bathrooms", "accommodates"]:
        if col not in map_data.columns:
            map_data[col] = np.nan

    return map_data


def render_map(map_data):
    if map_data.empty:
        st.info("No listings match the selected map filters.")
        return

    center_lat = float(map_data["latitude"].median())
    center_lon = float(map_data["longitude"].median())
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position="[longitude, latitude]",
        get_fill_color="color",
        get_radius=65,
        pickable=True,
        opacity=0.75,
    )

    tooltip = {
        "html": """
            <b>{price_label}</b><br/>
            {room_type}<br/>
            {neighbourhood_cleansed}<br/>
            Bedrooms: {bedrooms}<br/>
            Bathrooms: {bathrooms}<br/>
            Accommodates: {accommodates}
        """,
        "style": {"backgroundColor": "#111827", "color": "white"},
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=9.4,
            pitch=0,
        ),
        map_style=None,
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True)


def show_missing_model_message(missing):
    missing_names = ", ".join(str(path) for path in missing)
    st.error(f"Missing required file(s): {missing_names}")
    if MODEL_PATH in missing:
        st.info(
            'In `feature_and_modeling/feature_and_modeling_code.ipynb`, add '
            '`joblib.dump(gb_tuned, "feature_and_modeling/final_model.pkl")` '
            "immediately after the cell that creates and evaluates `gb_tuned`, before the final model comparison is saved. "
            'If you run the notebook from inside `feature_and_modeling/`, use `joblib.dump(gb_tuned, "final_model.pkl")` instead.'
        )


def main():
    st.title("LA Airbnb Price Predictor")
    st.caption("Predict nightly prices and explore how listing prices vary across Los Angeles.")

    listings = load_listing_data()
    model, preprocessor, missing = load_model_assets()
    feature_importance = load_feature_importance()

    if missing:
        show_missing_model_message(missing)
        st.stop()

    if listings.empty:
        st.warning("Listing data was not found, so form defaults and the map will be limited.")

    predict_tab, map_tab, importance_tab = st.tabs(
        ["Price Predictor", "LA Price Map", "Feature Importances"]
    )

    with predict_tab:
        st.subheader("Estimate a Nightly Price")

        room_types = clean_options(listings["room_type"]) if "room_type" in listings else ["Entire home/apt"]
        neighbourhoods = (
            clean_options(listings["neighbourhood_cleansed"])
            if "neighbourhood_cleansed" in listings
            else ["West Los Angeles"]
        )
        neighbourhood_groups = (
            clean_options(listings["neighbourhood_group_cleansed"])
            if "neighbourhood_group_cleansed" in listings
            else ["City of Los Angeles"]
        )
        property_types = (
            clean_options(listings["property_type"])
            if "property_type" in listings
            else ["Entire rental unit", "Entire home", "Private room in home"]
        )
        license_statuses = (
            clean_options(listings["license_status"])
            if "license_status" in listings
            else ["none"]
        )
        response_times = (
            clean_options(listings["host_response_time"])
            if "host_response_time" in listings
            else ["within a few hours"]
        )

        with st.form("prediction_form"):
            left, middle, right = st.columns(3)

            with left:
                st.markdown("**Listing Basics**")
                room_type = st.selectbox(
                    "Room type",
                    room_types,
                    index=room_types.index("Entire home/apt") if "Entire home/apt" in room_types else 0,
                )
                property_type = st.selectbox(
                    "Property type",
                    property_types,
                    index=property_types.index(most_common_value(listings, "property_type", property_types[0]))
                    if most_common_value(listings, "property_type", property_types[0]) in property_types
                    else 0,
                )
                neighbourhood = st.selectbox(
                    "Neighbourhood",
                    neighbourhoods,
                    index=neighbourhoods.index(most_common_value(listings, "neighbourhood_cleansed", neighbourhoods[0]))
                    if most_common_value(listings, "neighbourhood_cleansed", neighbourhoods[0]) in neighbourhoods
                    else 0,
                )
                neighbourhood_group = st.selectbox(
                    "Neighbourhood group",
                    neighbourhood_groups,
                    index=neighbourhood_groups.index(
                        most_common_value(listings, "neighbourhood_group_cleansed", neighbourhood_groups[0])
                    )
                    if most_common_value(listings, "neighbourhood_group_cleansed", neighbourhood_groups[0])
                    in neighbourhood_groups
                    else 0,
                )
                license_status = st.selectbox("License status", license_statuses)

            with middle:
                st.markdown("**Capacity and Location**")
                accommodates = st.number_input("Accommodates", min_value=1, max_value=20, value=4, step=1)
                bedrooms = st.number_input("Bedrooms", min_value=0.0, max_value=12.0, value=2.0, step=0.5)
                bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=12.0, value=1.5, step=0.5)
                beds = st.number_input("Beds", min_value=1.0, max_value=20.0, value=2.0, step=0.5)
                latitude = st.number_input(
                    "Latitude",
                    min_value=33.60,
                    max_value=34.40,
                    value=float(median_value(listings, "latitude", 34.0522)),
                    step=0.001,
                    format="%.5f",
                )
                longitude = st.number_input(
                    "Longitude",
                    min_value=-118.80,
                    max_value=-117.90,
                    value=float(median_value(listings, "longitude", -118.2437)),
                    step=0.001,
                    format="%.5f",
                )

            with right:
                st.markdown("**Host and Booking Details**")
                minimum_nights = st.number_input(
                    "Minimum nights",
                    min_value=1,
                    max_value=365,
                    value=int(median_value(listings, "minimum_nights", 2)),
                    step=1,
                )
                availability_365 = st.slider(
                    "Availability over next 365 days",
                    min_value=0,
                    max_value=365,
                    value=int(median_value(listings, "availability_365", 180)),
                )
                number_of_reviews = st.number_input(
                    "Number of reviews",
                    min_value=0,
                    max_value=2000,
                    value=int(median_value(listings, "number_of_reviews", 20)),
                    step=1,
                )
                review_scores_rating = st.slider(
                    "Review score rating",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(median_value(listings, "review_scores_rating", 4.8)),
                    step=0.05,
                )
                host_response_time = st.selectbox("Host response time", response_times)
                host_response_rate = st.slider(
                    "Host response rate",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(median_value(listings, "host_response_rate", 1.0)),
                    step=0.01,
                )
                host_since = st.date_input("Host since", value=pd.to_datetime("2018-01-01"))
                host_is_superhost = st.checkbox("Superhost")
                instant_bookable = st.checkbox("Instant bookable")
                host_never_responded = st.checkbox("Host never responded")

            submitted = st.form_submit_button("Predict nightly price", type="primary")

        if submitted:
            raw_input = {
                "host_since": host_since,
                "host_response_time": host_response_time,
                "host_response_rate": host_response_rate,
                "host_is_superhost": host_is_superhost,
                "neighbourhood_cleansed": neighbourhood,
                "neighbourhood_group_cleansed": neighbourhood_group,
                "latitude": latitude,
                "longitude": longitude,
                "property_type": property_type,
                "room_type": room_type,
                "accommodates": accommodates,
                "bathrooms": bathrooms,
                "bedrooms": bedrooms,
                "beds": beds,
                "minimum_nights": minimum_nights,
                "availability_365": availability_365,
                "number_of_reviews": number_of_reviews,
                "review_scores_rating": review_scores_rating,
                "instant_bookable": instant_bookable,
                "host_never_responded": host_never_responded,
                "license_status": license_status,
            }
            listing_features = engineer_listing_features(raw_input, known_grouped_property_types(listings))
            prediction = dollar_prediction(model, preprocessor, listing_features)
            st.metric("Predicted nightly price", f"${prediction:,.0f}")
            st.caption("The model predicts `log_price = log(1 + price)`; this app converts it back with `np.expm1()`.")

    with map_tab:
        st.subheader("Explore LA Airbnb Prices")
        if listings.empty:
            st.info("Add `feature_and_modeling/data_with_features.csv` or `feature_and_modeling/listings_clean.csv` to enable the map.")
        else:
            filters, map_area = st.columns([1, 3])
            with filters:
                st.markdown("**Filters**")
                room_options = clean_options(listings["room_type"]) if "room_type" in listings else []
                selected_room_types = st.multiselect("Room type", room_options, default=room_options)

                price_min = int(np.nanpercentile(listings["price"], 1)) if "price" in listings else 0
                price_max = int(np.nanpercentile(listings["price"], 99)) if "price" in listings else 1000
                price_range = st.slider("Price range", price_min, price_max, (price_min, price_max), step=10)

                group_options = (
                    clean_options(listings["neighbourhood_group_cleansed"])
                    if "neighbourhood_group_cleansed" in listings
                    else []
                )
                selected_groups = st.multiselect(
                    "Neighbourhood group",
                    group_options,
                    default=group_options,
                    disabled=not bool(group_options),
                )

                max_bedrooms = int(min(12, np.nanmax(listings["bedrooms"]))) if "bedrooms" in listings else 8
                bedroom_range = st.slider("Bedrooms", 0, max_bedrooms, (0, max_bedrooms))
                max_points = st.slider("Maximum map points", 1000, 15000, 6000, step=1000)

            filtered_map_data = prepare_map_data(
                listings,
                selected_room_types,
                price_range,
                selected_groups,
                bedroom_range,
                max_points,
            )
            with map_area:
                st.caption(f"Showing {len(filtered_map_data):,} listings after filters.")
                render_map(filtered_map_data)

    with importance_tab:
        st.subheader("Top Feature Importances")
        if feature_importance.empty:
            st.info("Add `feature_and_modeling/feature_importance.csv` to show model importances.")
        else:
            top_n = st.slider("Number of features", min_value=5, max_value=30, value=15)
            top_features = feature_importance.head(top_n).set_index("feature")[["importance"]]
            st.bar_chart(top_features, use_container_width=True)
            with st.expander("View raw importance table"):
                st.dataframe(feature_importance, use_container_width=True)


if __name__ == "__main__":
    main()
