import streamlit as st
import pandas as pd

# Ye function data ko cache karega (baar baar load nahi karega)
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

@st.cache_resource
def load_model():
    # Agar aap koi ML model load kar rahe hain toh yahan cache karein
    # jaise: model = joblib.load('model.pkl')
    # return model
    pass

import os
import pickle

import streamlit as st
import pandas as pd

import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Thermal Source Detection",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SIH26162 Thermal Source Detection",
    page_icon="🔥",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "classified_fire_data.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "trained_model.pkl"
)


# ============================================================
# LOAD CLASSIFIED DATA
# ============================================================

@st.cache_data
def load_fire_data():
    return pd.read_csv(DATA_PATH)


try:
    fires = load_fire_data()

except FileNotFoundError:
    st.error(
        "❌ classified_fire_data.csv not found.\n\n"
        f"Expected location:\n{DATA_PATH}"
    )
    st.stop()

except Exception as e:
    st.error(f"❌ Error loading classified data: {e}")
    st.stop()


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    return (
        model_data["model"],
        model_data["industry_encoder"],
        model_data["target_encoder"],
        model_data["features"]
    )


rf_available = False

if os.path.exists(MODEL_PATH):

    try:

        (
            rf_model,
            industry_encoder,
            target_encoder,
            trained_features
        ) = load_model()

        rf_available = True

    except Exception as e:

        st.sidebar.warning(
            f"⚠️ Random Forest model could not be loaded: {e}"
        )

else:

    st.sidebar.warning(
        "⚠️ trained_model.pkl not found."
    )


# ============================================================
# TITLE
# ============================================================

st.title("🔥 Thermal Source Detection Dashboard")

st.markdown(
    """
    **SIH26162 — AI & GIS based Thermal Anomaly Classification**

    This dashboard combines thermal observations, industrial
    proximity, persistence and environmental information to
    classify detected thermal sources.
    """
)


# ============================================================
# SIDEBAR — CLASSIFICATION MODE
# ============================================================

st.sidebar.header("🤖 Classification Mode")

classification_mode = st.sidebar.radio(
    "Select Classification Method",
    [
        "Rule-Based Classification",
        "Random Forest ML"
    ]
)


# ============================================================
# RULE-BASED DASHBOARD
# ============================================================

if classification_mode == "Rule-Based Classification":

    st.header("📌 Rule-Based Classification")

    st.caption(
        "6000 thermal anomaly classification and GIS dashboard"
    )


    # ========================================================
    # SIDEBAR FILTERS
    # ========================================================

    st.sidebar.header("🔎 Filters")

    classes = sorted(
        fires["source_class"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_classes = st.sidebar.multiselect(
        "Thermal Source Type",
        classes,
        default=classes
    )


    # ========================================================
    # FILTER DATA
    # ========================================================

    filtered = fires[
        fires["source_class"].isin(selected_classes)
    ].copy()


    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Detections",
            len(filtered)
        )

    with col2:

        industrial_count = filtered[
            filtered["source_class"].str.contains(
                "Industrial",
                case=False,
                na=False
            )
        ].shape[0]

        st.metric(
            "Industrial Sources",
            industrial_count
        )

    with col3:

        persistent_count = filtered[
            filtered["source_class"].str.contains(
                "Persistent",
                case=False,
                na=False
            )
        ].shape[0]

        st.metric(
            "Persistent Sources",
            persistent_count
        )

    with col4:

        if len(filtered) > 0:
            avg_frp = filtered["frp"].mean()
        else:
            avg_frp = 0

        st.metric(
            "Average FRP",
            f"{avg_frp:.2f}"
        )


    # ========================================================
    # CLASSIFICATION COUNTS
    # ========================================================

    st.subheader("📊 Thermal Source Classification")

    classification_counts = (
        filtered["source_class"]
        .value_counts()
    )

    st.bar_chart(classification_counts)


    # ========================================================
    # GIS MAP
    # ========================================================

    st.subheader("🗺️ Interactive Thermal Anomaly Map")

    if len(filtered) > 0:

        center_lat = filtered["latitude"].mean()
        center_lon = filtered["longitude"].mean()

        m = folium.Map(
            location=[
                center_lat,
                center_lon
            ],
            zoom_start=6,
            tiles="OpenStreetMap"
        )


        # ----------------------------------------------------
        # Marker Colors — CURRENT 5 CLASSES
        # ----------------------------------------------------

        marker_colors = {

            "Industrial Fire":
                "red",

            "Persistent Industrial Thermal Source":
                "darkred",

            "Wildfire":
                "green",

            "Agricultural Burning":
                "orange",

            "Other Thermal Source":
                "blue"
        }


        # ----------------------------------------------------
        # Marker Cluster
        # ----------------------------------------------------

        marker_cluster = MarkerCluster(
            name="Thermal Sources"
        ).add_to(m)


        # ----------------------------------------------------
        # Add Thermal Markers
        # ----------------------------------------------------

        for _, row in filtered.iterrows():

            source_class = row["source_class"]

            marker_color = marker_colors.get(
                source_class,
                "blue"
            )

            land_cover = row.get(
                "land_cover",
                "N/A"
            )

            nearest_industry = row.get(
                "nearest_industry",
                "N/A"
            )

            industry_type = row.get(
                "industry_type",
                "N/A"
            )

            distance = row.get(
                "distance_to_industry_km",
                0
            )

            persistence = row.get(
                "persistence_count",
                0
            )

            risk_score = row.get(
                "risk_score",
                "N/A"
            )


            # ------------------------------------------------
            # Distance Formatting
            # ------------------------------------------------

            try:

                distance_text = (
                    f"{float(distance):.2f}"
                )

            except:

                distance_text = str(distance)


            # ------------------------------------------------
            # Popup
            # ------------------------------------------------

            popup_text = f"""
            <div style="font-size:14px;">

            <b>🔥 Thermal Source:</b>
            {source_class}

            <br><br>

            <b>Brightness:</b>
            {row.get("brightness", "N/A")}

            <br>

            <b>FRP:</b>
            {row.get("frp", "N/A")}

            <br>

            <b>Confidence:</b>
            {row.get("confidence", "N/A")}

            <br>

            <b>Land Cover:</b>
            {land_cover}

            <br>

            <b>Nearest Industry:</b>
            {nearest_industry}

            <br>

            <b>Industry Type:</b>
            {industry_type}

            <br>

            <b>Distance to Industry:</b>
            {distance_text} km

            <br>

            <b>Persistence:</b>
            {persistence}

            <br>

            <b>Risk Score:</b>
            {risk_score}

            </div>
            """


            folium.Marker(

                location=[
                    row["latitude"],
                    row["longitude"]
                ],

                popup=folium.Popup(
                    popup_text,
                    max_width=500
                ),

                tooltip=source_class,

                icon=folium.Icon(
                    color=marker_color,
                    icon="fire",
                    prefix="fa"
                )

            ).add_to(marker_cluster)


        # ----------------------------------------------------
        # Map Legend
        # ----------------------------------------------------

        legend_html = """
        <div style="
            position: fixed;
            bottom: 30px;
            left: 30px;
            width: 280px;
            z-index: 9999;
            background-color: white;
            border: 2px solid grey;
            padding: 10px;
            font-size: 13px;
            border-radius: 8px;
        ">

        <b>🔥 Thermal Source Legend</b>

        <br><br>

        <span style="color:red;">●</span>
        Industrial Fire

        <br>

        <span style="color:darkred;">●</span>
        Persistent Industrial Thermal Source

        <br>

        <span style="color:green;">●</span>
        Wildfire

        <br>

        <span style="color:orange;">●</span>
        Agricultural Burning

        <br>

        <span style="color:blue;">●</span>
        Other Thermal Source

        </div>
        """

        m.get_root().html.add_child(
            folium.Element(legend_html)
        )


        # ----------------------------------------------------
        # Layer Control
        # ----------------------------------------------------

        folium.LayerControl().add_to(m)


        # ----------------------------------------------------
        # Display Map
        # ----------------------------------------------------

        st_folium(
            m,
            width=None,
            height=600
        )

    else:

        st.warning(
            "No thermal detections match the selected filters."
        )


    # ========================================================
    # DATA TABLE
    # ========================================================

    st.subheader("📋 Detection Details")

    display_columns = [

        "latitude",
        "longitude",
        "brightness",
        "confidence",
        "frp",
        "land_cover",
        "nearest_industry",
        "industry_type",
        "distance_to_industry_km",
        "persistence_count",
        "source_class",
        "confidence_percent",
        "risk_score"

    ]

    available_columns = [

        col
        for col in display_columns
        if col in filtered.columns

    ]

    st.dataframe(
        filtered[available_columns],
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.subheader("⬇️ Download Classified Data")

    csv_data = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="classified_fire_data.csv",
        mime="text/csv"
    )


# ============================================================
# RANDOM FOREST ML DASHBOARD
# ============================================================

else:

    st.header("🤖 Random Forest ML Classification")

    st.markdown(
        """
        Enter the thermal anomaly features below.

        The trained Random Forest model will predict the
        **Thermal Source**.
        """
    )


    # ========================================================
    # MODEL CHECK
    # ========================================================

    if not rf_available:

        st.error(
            """
            ❌ Random Forest model is not available.

            Please make sure this file exists:

            models/trained_model.pkl
            """
        )

        st.stop()


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.success(
        "✅ Trained Random Forest model loaded successfully."
    )

    st.info(
        "The saved Random Forest model is used directly. "
        "The model is NOT trained again during prediction."
    )


    # ========================================================
    # MODEL CLASSES
    # ========================================================

    st.subheader("🎯 Supported Thermal Sources")

    for i, cls in enumerate(
        target_encoder.classes_
    ):

        st.write(
            f"**{i} — {cls}**"
        )


    # ========================================================
    # INPUT SECTION
    # ========================================================

    st.subheader("🔥 Thermal Event Input")

    col1, col2 = st.columns(2)


    # ========================================================
    # LEFT COLUMN
    # ========================================================

    with col1:

        brightness = st.number_input(
            "Brightness",
            min_value=0.0,
            value=350.0,
            step=1.0,
            help="Thermal brightness value"
        )

        confidence = st.number_input(
            "Confidence (%)",
            min_value=0.0,
            max_value=100.0,
            value=90.0,
            step=1.0,
            help="Satellite detection confidence"
        )

        frp = st.number_input(
            "FRP",
            min_value=0.0,
            value=120.0,
            step=1.0,
            help="Fire Radiative Power"
        )

        distance_km = st.number_input(
            "Distance to Industry (km)",
            min_value=0.0,
            value=1.5,
            step=0.1,
            help="Distance from thermal anomaly to nearest industrial site"
        )


    # ========================================================
    # RIGHT COLUMN
    # ========================================================

    with col2:

        industry_types = list(
            industry_encoder.classes_
        )

        industry_type = st.selectbox(
            "Nearest Industry Type",
            industry_types
        )

        persistence_count = st.number_input(
            "Persistence / Detection Count",
            min_value=1,
            value=5,
            step=1,
            help="Number of repeated thermal detections at the location"
        )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.markdown("---")

    st.subheader("📋 Input Summary")

    input_summary = pd.DataFrame({

        "Feature": [

            "Brightness",
            "Confidence",
            "FRP",
            "Industry Type",
            "Distance (km)",
            "Persistence / Detection Count"

        ],

        "Value": [

            brightness,
            confidence,
            frp,
            industry_type,
            distance_km,
            persistence_count

        ]

    })

    st.dataframe(
        input_summary,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    st.markdown("---")

    predict_button = st.button(
        "🔍 Predict Thermal Source",
        type="primary",
        use_container_width=True
    )


    # ========================================================
    # RANDOM FOREST PREDICTION
    # ========================================================

    if predict_button:

        try:

            # ------------------------------------------------
            # Encode Industry Type
            # ------------------------------------------------

            industry_type_encoded = (
                industry_encoder
                .transform([industry_type])[0]
            )


            # ------------------------------------------------
            # Create Input DataFrame
            # ------------------------------------------------

            input_data = pd.DataFrame([{

                "brightness":
                    brightness,

                "confidence":
                    confidence,

                "frp":
                    frp,

                "industry_type_encoded":
                    industry_type_encoded,

                "distance_to_industry_km":
                    distance_km,

                "persistence_count":
                    persistence_count

            }])


            # ------------------------------------------------
            # Match Training Feature Order
            # ------------------------------------------------

            input_data = input_data[
                trained_features
            ]


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            prediction = rf_model.predict(
                input_data
            )[0]


            # ------------------------------------------------
            # Convert Encoded Class
            # ------------------------------------------------

            predicted_class = (
                target_encoder
                .inverse_transform(
                    [prediction]
                )[0]
            )


            # ------------------------------------------------
            # Probability
            # ------------------------------------------------

            probabilities = (
                rf_model
                .predict_proba(
                    input_data
                )[0]
            )


            # ------------------------------------------------
            # Prediction Confidence
            # ------------------------------------------------

            prediction_confidence = (
                probabilities[prediction] * 100
            )


            # =================================================
            # RESULT
            # =================================================

            st.markdown("---")

            st.subheader("🎯 Prediction Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.success(
                    f"🔥 Thermal Source Detected: "
                    f"**{predicted_class}**"
                )

            with result_col2:

                st.metric(
                    "Prediction Confidence",
                    f"{prediction_confidence:.2f}%"
                )


            # =================================================
            # SOURCE EXPLANATION
            # =================================================

            source_explanations = {

                "Agricultural Burning":
                    "The model classified this thermal event as agricultural burning.",

                "Industrial Fire":
                    "The model classified this thermal event as an industrial fire.",

                "Other Thermal Source":
                    "The model classified this event as another thermal source.",

                "Persistent Industrial Thermal Source":
                    "The model classified this as a persistent thermal source associated with an industrial area.",

                "Wildfire":
                    "The model classified this thermal event as a wildfire."

            }

            explanation = source_explanations.get(
                predicted_class,
                "The Random Forest model generated this thermal source classification."
            )

            st.info(
                f"**Classification Meaning:** {explanation}"
            )


            # =================================================
            # CLASS PROBABILITIES
            # =================================================

            st.subheader(
                "📊 Thermal Source Probabilities"
            )

            probability_data = pd.DataFrame({

                "Thermal Source":
                    target_encoder.classes_,

                "Probability (%)":
                    probabilities * 100

            })


            probability_data = (

                probability_data
                .sort_values(
                    "Probability (%)",
                    ascending=False
                )
                .reset_index(
                    drop=True
                )

            )


            probability_data[
                "Probability (%)"
            ] = (

                probability_data[
                    "Probability (%)"
                ].round(2)

            )


            st.dataframe(
                probability_data,
                use_container_width=True,
                hide_index=True
            )


            # =================================================
            # PROBABILITY CHART
            # =================================================

            st.subheader(
                "📈 Prediction Probability"
            )

            st.bar_chart(
                probability_data.set_index(
                    "Thermal Source"
                )[
                    "Probability (%)"
                ]
            )


            # =================================================
            # FINAL RESULT
            # =================================================

            st.markdown("---")

            st.subheader(
                "📋 Final ML Classification"
            )

            final_result = pd.DataFrame({

                "Parameter": [

                    "Brightness",
                    "Confidence",
                    "FRP",
                    "Industry Type",
                    "Distance to Industry",
                    "Persistence / Detection Count",
                    "🔥 Thermal Source",
                    "🎯 Prediction Confidence"

                ],

                "Value": [

                    brightness,
                    f"{confidence}%",
                    frp,
                    industry_type,
                    f"{distance_km} km",
                    persistence_count,
                    predicted_class,
                    f"{prediction_confidence:.2f}%"

                ]

            })

            st.dataframe(
                final_result,
                use_container_width=True,
                hide_index=True
            )


            # =================================================
            # ML EXPLANATION
            # =================================================

            st.markdown("---")

            st.subheader(
                "ℹ️ Random Forest Classification"
            )

            st.write(
                """
                The Random Forest model was trained using six
                features:

                **Brightness, Confidence, FRP, Industry Type,
                Distance to Industry and Persistence/Detection Count.**

                The trained model predicts the **Thermal Source**
                from these input features.

                The displayed confidence represents the Random
                Forest predicted probability of the selected class.

                The model is loaded from the saved `.pkl` file and
                is not retrained when the user makes a prediction.
                """
            )


        except Exception as e:

            st.error(
                f"❌ Prediction error: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "SIH26162 | Thermal Anomaly Detection and GIS Classification Prototype"
)