import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import requests
import json

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="Energy Data ML Dashboard", layout="wide")

# ----------------------------
# Sidebar Styling (with larger font)
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
        font-size: 18px;
    }
    [data-testid="stSidebar"] .stRadio {
        background-color: #334155;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
    }
    div.stButton > button:first-child {
        background-color: #3b82f6;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1rem;
        font-size: 16px;
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
    df = pd.read_csv("cleaned_dataset.xls")  # Replace with your dataset path
    df["gap"] = df["energy_requirement_mu"] - df["energy_availability_mu"]
    df["deficit_flag"] = (df["energy_deficit"] > 0).astype(int)
    return df

df = load_data()

# ----------------------------
# Load Indian GeoJSON
# ----------------------------
@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/udit-001/india-maps-data/main/geojson/india.geojson"
    response = requests.get(url)
    india_geojson = response.json()
    return india_geojson

india_geojson = load_geojson()

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("⚡ Navigation Panel")
section = st.sidebar.radio(
    "Select Section:",
    ["📘 Introduction", "📊 Exploratory Data Analysis", "🤖 ML Results", "🔮 Prediction"]
)

# ----------------------------
# Section 1: Introduction
# ----------------------------
if section == "📘 Introduction":
    st.title("📘 Introduction to Energy Dataset Analysis")
    st.markdown("""
    This dashboard provides a comprehensive analysis of energy data across Indian states.  
    **Dataset includes:**  
    - Region, State, Month, Quarter  
    - Energy Requirement (MU), Energy Availability (MU)  
    - Energy Deficit (MU)  

    **Objectives of this project:**  
    1. Perform **Exploratory Data Analysis (EDA)** to understand trends and distributions.  
    2. Apply **Machine Learning Algorithms** (KNN, Naive Bayes, Linear & Logistic Regression) to analyze patterns.  
    3. Allow **Prediction** of Energy Deficit based on input parameters.  
    4. Visualize energy distribution geographically using **India Map**.
    """)

# ----------------------------
# Section 2: EDA
# ----------------------------
elif section == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")

    st.subheader("🔍 Filter Options")
    col1, col2 = st.columns(2)
    with col1:
        region = st.selectbox("Select Region:", ["All"] + sorted(df["region"].unique().tolist()))
    with col2:
        quarter = st.selectbox("Select Quarter:", ["All"] + sorted(df["quarter"].unique().tolist()))

    filtered_df = df.copy()
    if region != "All":
        filtered_df = filtered_df[filtered_df["region"] == region]
    if quarter != "All":
        filtered_df = filtered_df[filtered_df["quarter"] == quarter]

    # Filtered dataset preview
    st.subheader("📋 Filtered Dataset")
    st.dataframe(filtered_df.head())

    # Key statistics
    st.subheader("📊 Key Statistics")
    st.write("Shape:", filtered_df.shape)
    st.dataframe(filtered_df[["energy_requirement_mu","energy_availability_mu","energy_deficit"]].describe())

    # Plot: Monthly Energy Requirement vs Availability
    st.subheader("⚙️ Energy Requirement vs Availability")
    fig1 = px.bar(
        filtered_df,
        x="month",
        y=["energy_requirement_mu", "energy_availability_mu"],
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={"value":"Energy (MU)", "month":"Month"},
        title="Monthly Energy Requirement vs Availability"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Plot: Energy Deficit Distribution
    st.subheader("💡 Energy Deficit Distribution")
    fig2 = px.box(
        filtered_df,
        x="region",
        y="energy_deficit",
        color="region",
        labels={"energy_deficit":"Deficit (MU)","region":"Region"},
        title="Energy Deficit Across Regions"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Plot: Correlation Heatmap
    st.subheader("📈 Correlation Heatmap")
    corr = filtered_df[["energy_requirement_mu","energy_availability_mu","energy_deficit"]].corr()
    fig3 = px.imshow(corr, text_auto=True, color_continuous_scale="Blues", title="Correlation Heatmap")
    st.plotly_chart(fig3, use_container_width=True)

    # Plot: India Map Choropleth
    st.subheader("🗺️ Energy Deficit Across India")
    map_df = filtered_df.groupby("state").sum().reset_index()
    fig4 = px.choropleth(
        map_df,
        geojson=india_geojson,
        featureidkey="properties.ST_NM",
        locations="state",
        color="energy_deficit",
        color_continuous_scale="Reds",
        labels={"energy_deficit":"Deficit (MU)"},
        title="Total Energy Deficit by State"
    )
    fig4.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig4, use_container_width=True)

# ----------------------------
# Section 3: ML Results
# ----------------------------
elif section == "🤖 ML Results":
    st.title("🤖 Machine Learning Analysis")

    # Common preprocessing
    features = ["energy_requirement_mu","energy_availability_mu","energy_deficit","gap"]
    X = filtered_df[features]
    y_class = filtered_df["deficit_flag"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KNN
    st.subheader("1️⃣ K-Nearest Neighbors (KNN)")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_class, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    st.write("Accuracy:", accuracy_score(y_test, y_pred_knn))
    st.text(classification_report(y_test, y_pred_knn))

    # Naive Bayes
    st.subheader("2️⃣ Naive Bayes")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    st.write("Accuracy:", accuracy_score(y_test, y_pred_nb))
    st.text(classification_report(y_test, y_pred_nb))

    # Linear Regression
    st.subheader("3️⃣ Linear Regression")
    X_lr = filtered_df[["energy_requirement_mu","energy_deficit","gap"]]
    y_lr = filtered_df["energy_availability_mu"]
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.3, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train_lr, y_train_lr)
    y_pred_lr = lr.predict(X_test_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test_lr, y_pred_lr))
    st.write("RMSE:", rmse_lr)
    st.write("R² Score:", r2_score(y_test_lr, y_pred_lr))
    st.line_chart(pd.DataFrame({"Actual": y_test_lr.values, "Predicted": y_pred_lr}))

    # Logistic Regression
    st.subheader("4️⃣ Logistic Regression")
    X_log = filtered_df[["energy_requirement_mu","energy_availability_mu","energy_deficit","gap"]]
    y_log = filtered_df["deficit_flag"]
    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.3, random_state=42)
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_log, y_train_log)
    y_pred_log = log_reg.predict(X_test_log)
    st.write("Accuracy:", accuracy_score(y_test_log, y_pred_log))
    st.text(classification_report(y_test_log, y_pred_log))

# ----------------------------
# Section 4: Prediction
# ----------------------------
elif section == "🔮 Prediction":
    st.title("🔮 Predict Energy Deficit")
    st.markdown("Input Energy Requirement and Availability to predict Energy Deficit using Linear Regression.")

    with st.form("prediction_form"):
        energy_req = st.number_input("Energy Requirement (MU)", min_value=0.0, step=100.0)
        energy_avail = st.number_input("Energy Availability (MU)", min_value=0.0, step=100.0)
        submitted = st.form_submit_button("Predict")

        if submitted:
            gap_input = energy_req - energy_avail
            input_df = pd.DataFrame([[energy_req, energy_avail, 0, gap_input]], columns=["energy_requirement_mu","energy_availability_mu","energy_deficit","gap"])
            pred = lr.predict(input_df)[0]
            st.success(f"✅ Predicted Energy Deficit: **{pred:.2f} MU**")
