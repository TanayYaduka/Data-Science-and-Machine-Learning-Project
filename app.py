# ----------------------------
# app.py
# ----------------------------
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
# Page Config
# ----------------------------
st.set_page_config(page_title="Energy Data ML Dashboard", layout="wide")

# ----------------------------
# Sidebar Styling
# ----------------------------
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        color: white;
        font-size: 18px;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
        color: white;
    }
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
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.xls")  # Replace with your cleaned dataset
    df["gap"] = df["energy_requirement_mu"] - df["energy_availability_mu"]
    df["deficit_flag"] = (df["energy_deficit"] > 0).astype(int)
    return df

df = load_data()

# ----------------------------
# Load India GeoJSON for map
# ----------------------------
@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/udit-001/india-maps-data/main/geojson/india.geojson"
    r = requests.get(url)
    india_geo = r.json()
    return india_geo

india_geo = load_geojson()

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("⚡ Navigation")
section = st.sidebar.radio(
    "Go to section:",
    ["📘 Introduction", "📊 Exploratory Data Analysis", "🤖 ML Results", "🔮 Prediction"]
)

# ----------------------------
# Section 1: Introduction
# ----------------------------
if section == "📘 Introduction":
    st.title("📘 Introduction to Energy Dataset Analysis")
    st.markdown("""
    This dashboard provides a comprehensive analysis of energy data across Indian states.  
    The dataset contains energy-related information for different regions, states, months, and quarters in India.
    """)
    
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())
    
    st.subheader("ℹ️ Dataset Overview")
    st.write("Total rows:", df.shape[0])
    st.write("Total columns:", df.shape[1])
    st.write("Regions:", df["region"].nunique())
    st.write("States:", df["state"].nunique())
    st.write("Months covered:", df["month"].nunique())
    st.write("Quarters covered:", df["quarter"].nunique())
    
    st.subheader("📄 Column Details")
    col_info = {
        "region": "Region of India (North, South, East, West, etc.) – useful to analyze regional energy patterns.",
        "state": "State name – to identify energy trends at the state level.",
        "is_union_territory": "Boolean indicating if the state is a Union Territory – may influence energy distribution and policy.",
        "month": "Month of observation – useful for monthly trends and seasonality.",
        "quarter": "Quarter of observation (Q1-Q4) – for aggregated trend analysis.",
        "energy_requirement_mu": "Energy Requirement in Million Units (MU) – main variable representing demand.",
        "energy_availability_mu": "Energy Availability in MU – actual supply available to meet demand.",
        "energy_deficit": "Energy Deficit in MU – difference between requirement and availability; target for analysis."
    }
    for col, desc in col_info.items():
        st.markdown(f"**{col}**: {desc}")

# ----------------------------
# Section 2: EDA
# ----------------------------
elif section == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")

    st.subheader("🔍 Filter Options")
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

    # Step 1: Filtered Dataset
    st.subheader("📋 Filtered Dataset Preview")
    st.dataframe(filtered_df.head())

    # Step 2: Key Statistics
    st.subheader("📊 Key Statistics")
    st.write("Shape:", filtered_df.shape)
    st.write("Numeric Summary:")
    st.dataframe(filtered_df[["energy_requirement_mu","energy_availability_mu","energy_deficit"]].describe())

    # Step 3: Plots
    st.subheader("⚙️ Energy Requirement vs Availability")
    fig1 = px.bar(
        filtered_df,
        x="month",
        y=["energy_requirement_mu","energy_availability_mu"],
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={"value":"Energy (MU)", "month":"Month"},
        title="Monthly Energy Requirement vs Availability (in MU)"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("💡 Energy Deficit Distribution")
    fig2 = px.box(
        filtered_df,
        x="region",
        y="energy_deficit",
        color="region",
        labels={"energy_deficit":"Deficit (MU)", "region":"Region"},
        title="Distribution of Energy Deficit Across Regions"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🗺️ Energy Requirement by State (Map)")
    state_summary = filtered_df.groupby("state").agg({
        "energy_requirement_mu":"sum",
        "energy_availability_mu":"sum",
        "energy_deficit":"sum"
    }).reset_index()

    fig_map = px.choropleth(
        state_summary,
        geojson=india_geo,
        featureidkey="properties.ST_NM",
        locations="state",
        color="energy_deficit",
        color_continuous_scale="Reds",
        labels={"energy_deficit":"Deficit (MU)"},
        title="Energy Deficit Across Indian States"
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig_map, use_container_width=True)

# ----------------------------
# Section 3: ML Results
# ----------------------------
elif section == "🤖 ML Results":
    st.title("🤖 Machine Learning Results and Insights")
    st.markdown("Below are results from different ML models applied to predict energy deficit.")

    features = ["energy_requirement_mu","energy_availability_mu","energy_deficit","gap"]
    X = filtered_df[features]
    y_class = filtered_df["deficit_flag"]
    y_reg = filtered_df["energy_deficit"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.3, random_state=42)

    st.subheader("🔹 K-Nearest Neighbors Classifier")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    st.write("Accuracy:", accuracy_score(y_test, y_pred_knn))
    st.text(classification_report(y_test, y_pred_knn))

    st.subheader("🔹 Naive Bayes Classifier")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    st.write("Accuracy:", accuracy_score(y_test, y_pred_nb))
    st.text(classification_report(y_test, y_pred_nb))

    st.subheader("🔹 Linear Regression")
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X, y_reg, test_size=0.3, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train_lr, y_train_lr)
    y_pred_lr = lr.predict(X_test_lr)
    st.write("RMSE:", mean_squared_error(y_test_lr, y_pred_lr, squared=False))
    st.write("R² Score:", r2_score(y_test_lr, y_pred_lr))
    st.line_chart(pd.DataFrame({"Actual":y_test_lr, "Predicted":y_pred_lr}))

    st.subheader("🔹 Logistic Regression")
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_class)
    y_pred_log = log_reg.predict(X)
    st.write("Accuracy:", accuracy_score(y_class, y_pred_log))
    st.text(classification_report(y_class, y_pred_log))

    st.subheader("🔹 K-Means Clustering")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    cluster_df = filtered_df.copy()
    cluster_df["Cluster"] = clusters
    st.dataframe(cluster_df[["state","energy_deficit","Cluster"]].head())
    st.write("Cluster Centers:")
    st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=features))

# ----------------------------
# Section 4: Prediction
# ----------------------------
elif section == "🔮 Prediction":
    st.title("🔮 Predict Energy Deficit")
    st.markdown("Input Energy Requirement and Availability to predict deficit using Linear Regression.")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            energy_req = st.number_input("Energy Requirement (MU)", min_value=0.0, step=100.0)
        with c2:
            energy_avail = st.number_input("Energy Availability (MU)", min_value=0.0, step=100.0)

        submitted = st.form_submit_button("Predict")
        if submitted:
            input_df = pd.DataFrame([[energy_req, energy_avail]], columns=["energy_requirement_mu","energy_availability_mu"])
            lr_model = LinearRegression()
            lr_model.fit(X_train_lr[["energy_requirement_mu","energy_availability_mu"]], y_train_lr)
            prediction = lr_model.predict(input_df)[0]
            st.success(f"✅ Predicted Energy Deficit: **{prediction:.2f} MU**")
