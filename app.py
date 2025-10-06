import streamlit as st
import pandas as pd
import plotly.express as px
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="Energy Data ML Dashboard", layout="wide")

# ----------------------------
# Custom Sidebar Styling
# ----------------------------
st.markdown("""
    <style>
    [data-testid="stSidebar"] {background-color: #1e293b; color: white;}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {color: white;}
    [data-testid="stSidebar"] .stSelectbox, [data-testid="stSidebar"] .stRadio {background-color: #334155; border-radius: 8px; padding: 10px;}
    div.stButton > button:first-child {background-color: #3b82f6; color: white; border-radius: 10px; border: none; padding: 0.6rem 1rem;}
    div.stButton > button:hover {background-color: #2563eb; color: white;}
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Data
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
st.sidebar.title("⚡ Navigation")
menu = st.sidebar.radio(
    "Go to section:",
    ["📊 Exploratory Data Analysis", "🤖 ML Models", "🔮 Prediction"],
    index=0
)

# ----------------------------
# Exploratory Data Analysis (EDA)
# ----------------------------
if menu == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("""
    Explore the dataset interactively. You can filter and visualize trends in 
    **Energy Requirement**, **Availability**, and **Deficit (in MU)**.
    """)

    # Filter options
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

    # Show filtered data
    st.subheader("📋 Filtered Dataset Preview")
    st.dataframe(filtered_df.head())

    # Key statistics
    st.subheader("📊 Key Statistics")
    st.write("Shape:", filtered_df.shape)
    st.dataframe(filtered_df[["energy_requirement_mu", "energy_availability_mu", "energy_deficit"]].describe())

    # ---- Plots ----
    # Bar plot: Energy Requirement vs Availability
    st.subheader("⚙️ Energy Requirement vs Availability")
    fig1 = px.bar(
        filtered_df,
        x="month",
        y=["energy_requirement_mu", "energy_availability_mu"],
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={"value": "Energy (Million Units - MU)", "month": "Month"},
        title="Monthly Energy Requirement vs Availability"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Box plot: Energy Deficit by Region
    st.subheader("💡 Energy Deficit Distribution")
    fig2 = px.box(
        filtered_df,
        x="region",
        y="energy_deficit",
        color="region",
        labels={"energy_deficit": "Deficit (MU)", "region": "Region"},
        title="Distribution of Energy Deficit Across Regions"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Correlation Heatmap
    st.subheader("📈 Correlation Heatmap")
    corr = filtered_df[["energy_requirement_mu", "energy_availability_mu", "energy_deficit"]].corr()
    fig3 = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ---- Indian Map: Energy Deficit ----
    st.subheader("🗺️ State-wise Energy Deficit in India")
    # Load GeoJSON
    with open("india.geojson", "r") as f:
        india_geo = json.load(f)

    state_data = filtered_df.groupby("state")["energy_deficit"].sum().reset_index()
    fig_map = px.choropleth(
        state_data,
        geojson=india_geo,
        featureidkey="properties.ST_NM",
        locations="state",
        color="energy_deficit",
        color_continuous_scale="Reds",
        labels={"energy_deficit": "Deficit (MU)"},
        hover_data=["state", "energy_deficit"],
        title="State-wise Energy Deficit in India"
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig_map, use_container_width=True)

# ----------------------------
# ML Models Section
# ----------------------------
elif menu == "🤖 ML Models":
    st.title("🤖 Machine Learning Models")
    st.markdown("""
    Apply PCA, KNN, Naive Bayes, Linear Regression, Logistic Regression, and K-Means Clustering.
    """)

    features = ["energy_requirement_mu", "energy_availability_mu", "energy_deficit", "gap"]
    X_scaled = StandardScaler().fit_transform(df[features])

    # PCA
    st.subheader("📌 PCA Visualization")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
    st.scatter_chart(pd.DataFrame(X_pca, columns=["PCA1", "PCA2"]))

    # KNN Classifier
    st.subheader("📌 KNN Classifier")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, df["deficit_flag"], test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text(classification_report(y_test, y_pred))

    # Naive Bayes
    st.subheader("📌 Naive Bayes Classifier")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    st.write("Accuracy:", accuracy_score(y_test, y_pred_nb))
    st.text(classification_report(y_test, y_pred_nb))

    # Linear Regression
    st.subheader("📌 Linear Regression")
    X_lr = df[["energy_requirement_mu", "energy_deficit", "gap"]]
    y_lr = df["energy_availability_mu"]
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.3, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train_lr, y_train_lr)
    y_pred_lr = lr.predict(X_test_lr)
    st.write("RMSE:", mean_squared_error(y_test_lr, y_pred_lr, squared=False))
    st.write("R² Score:", r2_score(y_test_lr, y_pred_lr))

    # Logistic Regression
    st.subheader("📌 Logistic Regression")
    X_log = df[features]
    y_log = df["deficit_flag"]
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_log, y_log)
    y_pred_log = log_reg.predict(X_log)
    st.write("Accuracy:", accuracy_score(y_log, y_pred_log))
    st.text(classification_report(y_log, y_pred_log))

    # K-Means
    st.subheader("📌 K-Means Clustering")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    cluster_df = pd.DataFrame(X_scaled, columns=features)
    cluster_df["Cluster"] = clusters
    st.write(cluster_df.head())

# ----------------------------
# Prediction Section
# ----------------------------
elif menu == "🔮 Prediction":
    st.title("🔮 Predict Energy Deficit")
    st.markdown("Enter values to predict Energy Deficit using Linear Regression.")
    with st.form("predict_form"):
        energy_req = st.number_input("Energy Requirement (MU)", min_value=0.0, step=100.0)
        energy_avail = st.number_input("Energy Availability (MU)", min_value=0.0, step=100.0)
        submitted = st.form_submit_button("Predict")
        if submitted:
            gap_val = energy_req - energy_avail
            input_df = pd.DataFrame([[energy_req, energy_avail, gap_val]], columns=["energy_requirement_mu","energy_availability_mu","gap"])
            pred = lr.predict(input_df)[0]
            st.success(f"Predicted Energy Deficit: {pred:.2f} MU")
