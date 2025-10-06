# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import requests
import json

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(page_title="Energy Data ML Dashboard", layout="wide")

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
    <style>
    /* Sidebar background and style */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        color: white;
        font-size: 20px;
        font-weight: bold;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
        color: white;
        font-size: 18px;
    }
    /* Buttons */
    div.stButton > button:first-child {
        background-color: #3b82f6;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1rem;
    }
    div.stButton > button:hover {
        background-color: #2563eb;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Load dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.xls")
    df["gap"] = df["energy_requirement_mu"] - df["energy_availability_mu"]
    df["deficit_flag"] = (df["energy_deficit"] > 0).astype(int)
    return df

df = load_data()

# ----------------------------
# Load India GeoJSON
# ----------------------------
@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/udit-001/india-maps-data/main/geojson/india.geojson"
    response = requests.get(url)
    india_geo = json.loads(response.text)
    return india_geo

india_geo = load_geojson()

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("⚡ Navigation")
section = st.sidebar.radio(
    "Go to section:",
    ["📘 Dataset Description", "📊 EDA", "🤖 ML Models", "🔮 Prediction"]
)

# ----------------------------
# Section 1: Dataset Description
# ----------------------------
if section == "📘 Dataset Description":
    st.title("📘 Dataset Description")
    st.markdown("""
    This dashboard uses **Power Supply Data of India**.  
    Data source: [India Data Portal](https://indiadataportal.com/p/power/r/mop-power_supply_position-st-mn-aaa)
    """)
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Column Details")
    st.markdown("""
    - **region**: Geographical region of India (North, South, etc.).  
    - **state**: State name in India.  
    - **is_union_territory**: True if UT, else False.  
    - **month**: Month of data reporting.  
    - **quarter**: Quarter of data reporting.  
    - **energy_requirement_mu**: Energy required (Million Units).  
    - **energy_availability_mu**: Energy available (Million Units).  
    - **energy_deficit**: Difference between requirement & availability (Million Units).  
    - **gap**: Calculated as requirement minus availability (Million Units).  
    - **deficit_flag**: 1 if deficit > 0, else 0.
    """)

# ----------------------------
# Section 2: EDA
# ----------------------------
elif section == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        region = st.selectbox("Select Region:", ["All"] + sorted(df["region"].unique()))
    with col2:
        quarter = st.selectbox("Select Quarter:", ["All"] + sorted(df["quarter"].unique()))

    filtered_df = df.copy()
    if region != "All":
        filtered_df = filtered_df[filtered_df["region"] == region]
    if quarter != "All":
        filtered_df = filtered_df[filtered_df["quarter"] == quarter]

    st.subheader("Filtered Dataset Preview")
    st.dataframe(filtered_df.head())

    st.subheader("Key Statistics")
    st.write("Shape:", filtered_df.shape)
    st.dataframe(filtered_df[["energy_requirement_mu", "energy_availability_mu", "energy_deficit"]].describe())

    # Plots
    st.subheader("Monthly Energy Requirement vs Availability")
    fig1 = px.bar(
        filtered_df,
        x="month",
        y=["energy_requirement_mu", "energy_availability_mu"],
        barmode="group",
        labels={"value": "Energy (MU)", "month": "Month"},
        title="Monthly Energy Requirement vs Availability"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Energy Deficit Distribution by Region")
    fig2 = px.box(
        filtered_df,
        x="region",
        y="energy_deficit",
        color="region",
        labels={"energy_deficit": "Deficit (MU)"},
        title="Energy Deficit Across Regions"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("India Map: Energy Deficit")
    state_avg = filtered_df.groupby("state")["energy_deficit"].mean().reset_index()
    fig3 = px.choropleth_mapbox(
        state_avg,
        geojson=india_geo,
        locations="state",
        featureidkey="properties.ST_NM",
        color="energy_deficit",
        color_continuous_scale="Reds",
        mapbox_style="carto-positron",
        zoom=3,
        center={"lat": 22, "lon": 78},
        opacity=0.6,
        labels={"energy_deficit": "Avg Deficit (MU)"},
        title="Average Energy Deficit by State"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# Section 3: ML Models
# ----------------------------
elif section == "🤖 ML Models":
    st.title("🤖 ML Model Evaluation")

    st.markdown("Predict **Energy Deficit (MU)** using different ML models.")

    model_choice = st.multiselect(
        "Select ML Models to Evaluate:",
        ["Linear Regression", "KNN Classifier", "Naive Bayes", "Logistic Regression", "K-Means Clustering"],
        default=["Linear Regression", "KNN Classifier"]
    )

    # Features and target
    X_reg = df[["energy_requirement_mu", "energy_availability_mu"]]
    y_reg = df["energy_deficit"]
    X_clf = X_reg
    y_clf = (y_reg > 0).astype(int)

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)

    for model in model_choice:
        st.subheader(f"📌 {model} Results")

        if model == "Linear Regression":
            lr = LinearRegression()
            lr.fit(X_train_reg, y_train_reg)
            y_pred_lr = lr.predict(X_test_reg)
            st.write("RMSE:", round(mean_squared_error(y_test_reg, y_pred_lr, squared=False), 2))
            st.write("R² Score:", round(r2_score(y_test_reg, y_pred_lr), 2))
            st.line_chart(pd.DataFrame({"Actual": y_test_reg.values, "Predicted": y_pred_lr}))
            st.markdown("**Insight:** Linear Regression predicts the deficit values based on energy inputs.")

        elif model == "KNN Classifier":
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train_clf, y_train_clf)
            y_pred_knn = knn.predict(X_test_clf)
            st.write("Accuracy:", round(accuracy_score(y_test_clf, y_pred_knn), 2))
            st.text(classification_report(y_test_clf, y_pred_knn))
            st.markdown("**Insight:** KNN classifies deficit occurrence (yes/no), not exact MU.")

        elif model == "Naive Bayes":
            nb = GaussianNB()
            nb.fit(X_train_clf, y_train_clf)
            y_pred_nb = nb.predict(X_test_clf)
            st.write("Accuracy:", round(accuracy_score(y_test_clf, y_pred_nb), 2))
            st.text(classification_report(y_test_clf, y_pred_nb))
            st.markdown("**Insight:** Naive Bayes is for classification of deficit presence, not exact values.")

        elif model == "Logistic Regression":
            log_reg = LogisticRegression(max_iter=1000)
            log_reg.fit(X_train_clf, y_train_clf)
            y_pred_log = log_reg.predict(X_test_clf)
            st.write("Accuracy:", round(accuracy_score(y_test_clf, y_pred_log), 2))
            st.text(classification_report(y_test_clf, y_pred_log))
            st.markdown("**Insight:** Logistic Regression predicts deficit yes/no (binary classification).")

        elif model == "K-Means Clustering":
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_reg)
            st.write("Cluster Centers:")
            st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=X_reg.columns))
            cluster_df = X_reg.copy()
            cluster_df["Cluster"] = clusters
            st.write(cluster_df.head())
            st.markdown("**Insight:** K-Means groups similar energy patterns, not for prediction.")

# ----------------------------
# Section 4: Prediction
# ----------------------------
elif section == "🔮 Prediction":
    st.title("🔮 Predict Energy Deficit")
    st.markdown("Choose parameters to predict **Energy Deficit (MU)** for a state and quarter.")

    col1, col2, col3 = st.columns(3)
    with col1:
        pred_region = st.selectbox("Select Region:", sorted(df["region"].unique()))
    with col2:
        pred_state = st.selectbox("Select State:", sorted(df["state"].unique()))
    with col3:
        pred_quarter = st.selectbox("Select Quarter:", sorted(df["quarter"].unique()))

    energy_req = st.number_input("Energy Requirement (MU)", min_value=0.0, step=100.0)
    energy_avail = st.number_input("Energy Availability (MU)", min_value=0.0, step=100.0)

    if st.button("Predict"):
        lr = LinearRegression()
        lr.fit(df[["energy_requirement_mu", "energy_availability_mu"]], df["energy_deficit"])
        pred_value = lr.predict([[energy_req, energy_avail]])[0]
        st.success(f"✅ Predicted Energy Deficit for {pred_state}, {pred_quarter}: **{pred_value:.2f} MU**")
