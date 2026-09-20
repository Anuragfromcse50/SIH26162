
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SIH26162 Thermal Source Detection",
    page_icon="🔥",
    layout="wide"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

DATA_PATH = "data/classified_fire_data.csv"

try:
    fires = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(
        "classified_fire_data.csv not found. "
        "Please run the notebook classification steps first."
    )
    st.stop()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🔥 Thermal Source Detection Dashboard")

st.markdown(
    """
    **SIH26162 — AI & GIS based Thermal Anomaly Classification**

    This dashboard combines thermal observations, land-cover information,
    industrial proximity and persistence to classify detected thermal sources.
    """
)


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("🔎 Filters")

classes = sorted(
    fires["source_class"].dropna().unique().tolist()
)

selected_classes = st.sidebar.multiselect(
    "Thermal Source Type",
    classes,
    default=classes
)


# Filter data
filtered = fires[
    fires["source_class"].isin(selected_classes)
].copy()


# --------------------------------------------------
# METRICS
# --------------------------------------------------

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
    avg_frp = filtered["frp"].mean()

    st.metric(
        "Average FRP",
        f"{avg_frp:.2f}"
    )


# --------------------------------------------------
# CLASSIFICATION DISTRIBUTION
# --------------------------------------------------

st.subheader("📊 Thermal Source Classification")

classification_counts = (
    filtered["source_class"]
    .value_counts()
)

st.bar_chart(classification_counts)


# --------------------------------------------------
# GIS MAP
# --------------------------------------------------

st.subheader("🗺️ Interactive Thermal Anomaly Map")


if len(filtered) > 0:

    center_lat = filtered["latitude"].mean()
    center_lon = filtered["longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="OpenStreetMap"
    )

    # Marker colors
    marker_colors = {
        "Industrial Fire": "red",
        "Persistent Industrial Thermal Source": "darkred",
        "Wildfire": "green",
        "Agricultural Burning": "orange",
        "Other Thermal Source": "blue"
    }

    for _, row in filtered.iterrows():

        source_class = row["source_class"]

        marker_color = marker_colors.get(
            source_class,
            "blue"
        )

        # Popup with nearest_industry and industry_type ADDED
        popup_text = f"""
        <b>Classification:</b> {source_class}<br>
        <b>Brightness:</b> {row['brightness']}<br>
        <b>FRP:</b> {row['frp']}<br>
        <b>Confidence:</b> {row['confidence']}<br>
        <b>Land Cover:</b> {row['land_cover']}<br>
        <b>Nearest Industry:</b> {row.get('nearest_industry', 'N/A')}<br>
        <b>Industry Type:</b> {row.get('industry_type', 'N/A')}<br>
        <b>Distance to Industry:</b> {row['distance_to_industry_km']:.2f} km<br>
        <b>Persistence:</b> {row['persistence_count']}<br>
        <b>Risk Score:</b> {row['risk_score']}
        """

        folium.Marker(
            location=[
                row["latitude"],
                row["longitude"]
            ],
            popup=folium.Popup(
                popup_text,
                max_width=500,
                max_height=400
            ),
            tooltip=source_class,
            icon=folium.Icon(
                color=marker_color,
                icon="fire",
                prefix="fa"
            )
        ).add_to(m)

    st_folium(
        m,
        width=None,
        height=600
    )

else:

    st.warning(
        "No thermal detections match the selected filters."
    )


# --------------------------------------------------
# DATA TABLE
# --------------------------------------------------

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
    col for col in display_columns
    if col in filtered.columns
]

st.dataframe(
    filtered[available_columns],
    use_container_width=True
)


# --------------------------------------------------
# DOWNLOAD
# --------------------------------------------------

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


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "SIH26162 | Thermal Anomaly Detection and GIS Classification Prototype"
)