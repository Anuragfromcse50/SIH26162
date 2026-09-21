# 🔥 SIH26162 — Thermal Anomaly & Industrial Fire Classification System

## 📌 Overview

**SIH26162** is a GIS and Machine Learning based system designed to analyze thermal anomalies/fire events and classify their possible sources.

The system combines thermal/fire detection data with:

* 🔥 Thermal anomaly features
* 🏭 Nearest industrial facility information
* 📍 Distance from industrial facilities
* 🌱 Land-cover information
* 🔁 Persistence/detection count
* 🤖 Machine Learning classification
* 🗺️ Interactive GIS visualization

The project provides both **Rule-Based Classification** and **Random Forest Machine Learning Classification** through an interactive Streamlit dashboard.

---

## 🎯 Objectives

The main objectives of the project are:

1. Detect and analyze thermal anomalies.
2. Identify the nearest industrial facility to a thermal event.
3. Calculate the distance between thermal events and industrial facilities.
4. Analyze persistence of thermal anomalies.
5. Classify thermal events into different source categories.
6. Visualize detected events on an interactive GIS map.
7. Provide a Machine Learning based prediction system.

---

## 🧠 Classification Categories

The current trained Random Forest model classifies events into five categories:

| Class                                   | Description                                                          |
| --------------------------------------- | -------------------------------------------------------------------- |
| 🌾 Agricultural Burning                 | Thermal events associated with agricultural burning                  |
| 🏭 Industrial Fire                      | Fire events associated with industrial facilities                    |
| 🔥 Other Thermal Source                 | Thermal anomalies that do not clearly belong to the other categories |
| 🏭 Persistent Industrial Thermal Source | Repeated thermal activity near industrial facilities                 |
| 🌲 Wildfire                             | Thermal events associated with forest/wildland fires                 |

---

## ⚙️ System Workflow

```text
Thermal Anomaly Data
        ↓
Data Cleaning
        ↓
Land Cover Integration
        ↓
Industrial Facility Integration
        ↓
Nearest Industry Detection
        ↓
Distance Calculation
        ↓
Persistence Analysis
        ↓
Feature Engineering
        ↓
 ┌───────────────────────┐
 │                       │
 ▼                       ▼
Rule-Based           Random Forest
Classification        Classification
 │                       │
 └───────────┬───────────┘
             ↓
      Streamlit Dashboard
             ↓
      Interactive GIS Map
```

---

## 🤖 Machine Learning

The project uses a **Random Forest Classifier**.

### Features Used

The current model uses:

```text
Brightness
Confidence
FRP
Industry Type
Distance to Industry
Persistence Count
```

`Industry Type` is encoded using `LabelEncoder`.

The target variable is:

```text
source_class
```

### Train/Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

using stratified sampling.

### Model Configuration

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
```

### Current Model Accuracy

The current trained model achieved approximately:

**56.42% test accuracy**

The model also provides:

* Classification report
* Class-wise precision
* Recall
* F1-score
* Prediction probability/confidence

---

## 🗺️ GIS Dashboard

The Streamlit dashboard provides an interactive map containing thermal anomaly locations.

The map includes:

* 📍 Thermal event markers
* 🏭 Industrial information
* 🔥 Classification information
* 📊 Event details
* 🔎 Filtering
* 🗂️ Marker clustering
* 🗺️ OpenStreetMap base map
* 📥 Classified data download

Users can inspect individual thermal events through map popups.

---

## 📊 Dashboard Modes

### 1. Rule-Based Classification

Uses predefined conditions based on:

* Industrial proximity
* Persistence
* Land cover
* Thermal characteristics

This mode is useful for transparent, rule-based analysis.

### 2. Random Forest ML

Users can manually provide input values such as:

```text
Brightness
Confidence
FRP
Industry Type
Distance to Industry
Persistence Count
```

The trained Random Forest model then predicts:

```text
Predicted Source Class
Prediction Confidence
```

---

## 📁 Project Structure

```text
SIH26162/
│
├── app/
│   └── app.py
│
├── data/
│   └── classified_fire_data.csv
│
├── models/
│   └── trained_model.pkl
│
└── README.md
```

---

## 🛠️ Technologies Used

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Core programming          |
| Pandas       | Data processing           |
| NumPy        | Numerical operations      |
| GeoPandas    | GIS/geospatial processing |
| Scikit-learn | Machine Learning          |
| Folium       | Interactive GIS maps      |
| Streamlit    | Dashboard                 |
| Matplotlib   | Data visualization        |
| Pickle       | Model storage             |

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Open the project

```bash
cd SIH26162
```

### 3. Create virtual environment

```bash
python -m venv .venv
```

### 4. Activate virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install pandas numpy geopandas folium matplotlib scikit-learn streamlit
```

### 6. Run the dashboard

```bash
streamlit run app/app.py
```

The Streamlit dashboard will open in the browser.

---

## 📂 Dataset

The prototype currently uses a **FIRMS-like/sample thermal anomaly dataset**.

Important fields include:

```text
latitude
longitude
brightness
confidence
acq_date
acq_time
frp
land_cover
vegetation_index
distance_to_industry_km
persistence_count
land_cover_code
source_class
classification_confidence
confidence_percent
risk_score
nearest_industry
industry_type
```

The system can later be connected to real satellite/fire datasets.

---

## 🔮 Future Scope

Future versions can include:

* NASA FIRMS real-time data integration
* OSM-based automatic industrial facility extraction
* Satellite imagery integration
* Improved land-cover classification
* More training data
* Additional thermal-source classes
* Deep Learning based classification
* Real-time fire alerts
* Historical fire analysis
* Risk heatmaps
* Automated reporting
* Cloud deployment

---

## 👨‍💻 Project

**Smart India Hackathon — SIH26162**

### Project Focus

> GIS + Machine Learning based Thermal Anomaly and Industrial Fire Source Classification

---

## 📜 Note

This project is currently a **working prototype** developed for Smart India Hackathon. The machine learning performance depends on the quality, distribution, and representativeness of the training dataset.

