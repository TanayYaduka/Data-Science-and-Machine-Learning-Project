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

# ----------------------------
# Page config
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
    font-size:18px;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
    color: white;
}
[data-testid="stSidebar"] .stRadio, [data-testid="stSidebar"] .stButton {
    background-color: #334155;
    border-radius: 8px;
    padding: 10px;
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
# Load data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.xls")
    df["gap"] = df["energy_requirement_mu"] - df["energy_availability_mu"]
    df["deficit_flag"] = (df["energy_deficit"] > 0).astype(int)
    return df

df = load_data()

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("⚡ Navigation Panel")
menu = st.sidebar.radio(
    "Select Section:",
    ["📘 Introduction", "📊 EDA", "🤖 ML Model Results", "🔮 Prediction"]
)

# ----------------------------
# Section 1: Introduction
# ----------------------------
if menu == "📘 Introduction":
    st.title("📘 Introduction to the Energy Dataset")
    st.markdown("""
This dashboard explores India's monthly energy supply dataset.  
The dataset is sourced from **[India Data Portal](https://indiadataportal.com/p/power/r/mop-power_supply_position-st-mn-aaa)**.  

**Columns:**  
- **region:** Region of India  
- **state:** State/UT  
- **is_union_territory:** Boolean flag  
- **month:** Month of observation  
- **quarter:** Financial quarter (Q1-Q4)  
- **energy_requirement_mu:** Energy required (Million Units)  
- **energy_availability_mu:** Energy available (MU)  
- **energy_deficit:** Deficit (MU)  
- **gap:** Requirement - Availability  

**Dataset Preview:**  
""")
    st.dataframe(df.head())

# ----------------------------
# Section 2: EDA
# ----------------------------
elif menu == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        region = st.selectbox("Select Region:", ["All"] + sorted(df["region"].unique()))
    with col2:
        quarter = st.selectbox("Select Quarter:", ["All"] + sorted(df["quarter"].unique()))
    with col3:
        month = st.selectbox("Select Month:", ["All"] + sorted(df["month"].unique()))

    filtered_df = df.copy()
    if region != "All": filtered_df = filtered_df[filtered_df["region"]==region]
    if quarter != "All": filtered_df = filtered_df[filtered_df["quarter"]==quarter]
    if month != "All": filtered_df = filtered_df[filtered_df["month"]==month]

    st.subheader("📋 Filtered Dataset Preview")
    st.dataframe(filtered_df.head())

    st.subheader("📊 Key Statistics")
    st.write("Shape:", filtered_df.shape)
    st.dataframe(filtered_df[["energy_requirement_mu","energy_availability_mu","energy_deficit","gap"]].describe())

    # Plots
    st.subheader("📈 Energy Requirement vs Availability")
    fig1 = px.bar(filtered_df, x="month", y=["energy_requirement_mu","energy_availability_mu"],
                  barmode="group", labels={"value":"Energy (MU)"}, color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("💡 Energy Deficit Distribution by Region")
    fig2 = px.box(filtered_df, x="region", y="energy_deficit", color="region",
                  labels={"energy_deficit":"Deficit (MU)","region":"Region"})
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🗺️ Energy Deficit Map of India")
    geo_url = "https://raw.githubusercontent.com/udit-001/india-maps-data/main/geojson/india.geojson"
    india_geojson = requests.get(geo_url).json()
    state_deficit = filtered_df.groupby("state")["energy_deficit"].sum().reset_index()
    fig_map = px.choropleth_mapbox(state_deficit, geojson=india_geojson, locations="state",
                                   featureidkey="properties.st_nm", color="energy_deficit",
                                   color_continuous_scale="OrRd", mapbox_style="carto-positron",
                                   zoom=3.5, center={"lat":23.5937,"lon":80.9629},
                                   opacity=0.7, labels={"energy_deficit":"Deficit (MU)"})
    st.plotly_chart(fig_map, use_container_width=True)

# ----------------------------
# Section 3: ML Model Results
# ----------------------------
elif menu == "🤖 ML Model Results":
    st.title("🤖 ML Model Results")
    features = ["energy_requirement_mu","energy_availability_mu","energy_deficit","gap"]
    X = df[features]
    y_clf = df["deficit_flag"]
    y_reg = df["energy_availability_mu"]

    # Classification split
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.3, random_state=42)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_clf, y_train_clf)
    y_pred_knn = knn.predict(X_test_clf)
    st.subheader("KNN Classifier")
    st.write("Accuracy:", round(accuracy_score(y_test_clf, y_pred_knn),3))
    st.text(classification_report(y_test_clf, y_pred_knn))

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train_clf, y_train_clf)
    y_pred_nb = nb.predict(X_test_clf)
    st.subheader("Naive Bayes")
    st.write("Accuracy:", round(accuracy_score(y_test_clf, y_pred_nb),3))
    st.text(classification_report(y_test_clf, y_pred_nb))

    # Linear Regression
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.3, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train_reg, y_train_reg)
    y_pred_lr = lr.predict(X_test_reg)
    # Compute RMSE correctly
    rmse_lr = mean_squared_error(y_test_reg.astype(float), y_pred_lr.astype(float), squared=False)
    r2_lr = r2_score(y_test_reg, y_pred_lr)
    st.subheader("Linear Regression")
    st.write("RMSE:", round(rmse_lr, 2))
    st.write("R² Score:", round(r2_lr,3))
    st.line_chart(pd.DataFrame({"Actual":y_test_reg.values,"Predicted":y_pred_lr}))

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_clf, y_train_clf)
    y_pred_log = log_reg.predict(X_test_clf)
    st.subheader("Logistic Regression")
    st.write("Accuracy:", round(accuracy_score(y_test_clf, y_pred_log),3))
    st.text(classification_report(y_test_clf, y_pred_log))

    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    cluster_df = X.copy()
    cluster_df["Cluster"] = clusters
    st.subheader("K-Means Clustering")
    st.dataframe(cluster_df.head())

# ----------------------------
# Section 4: Prediction
# ----------------------------
elif menu == "🔮 Prediction":
    st.title("🔮 Predict Energy Deficit")
    st.markdown("Select **State**, **Quarter**, and enter energy values to predict deficit.")

    pred_state = st.selectbox("Select State:", sorted(df["state"].unique()))
    pred_quarter = st.selectbox("Select Quarter:", sorted(df["quarter"].unique()))
    energy_req = st.number_input("Energy Requirement (MU)", min_value=0.0, step=100.0)
    energy_avail = st.number_input("Energy Availability (MU)", min_value=0.0, step=100.0)

    if st.button("Predict Deficit"):
        input_df = pd.DataFrame([[energy_req, energy_avail, energy_req - energy_avail, energy_req - energy_avail]],
                                columns=["energy_requirement_mu","energy_availability_mu","energy_deficit","gap"])
        lr_pred = LinearRegression()
        lr_pred.fit(X_train_reg, y_train_reg)
        pred_deficit = lr_pred.predict(input_df)[0]
        st.success(f"Predicted Energy Deficit for {pred_state}, {pred_quarter}: **{pred_deficit:.2f} MU**")


