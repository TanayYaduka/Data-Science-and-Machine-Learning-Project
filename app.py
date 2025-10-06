import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.csv")
    df["gap"] = df["energy_requirement_mu"] - df["energy_availability_mu"]
    df["deficit_flag"] = (df["energy_deficit"] > 0).astype(int)
    return df

df = load_data()

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="Energy Data ML Dashboard", layout="wide")
st.title("⚡ Energy Data ML Dashboard")

# Sidebar Navigation (Radio instead of dropdown)
page = st.sidebar.radio(
    "🔍 Select Section",
    [
        "Data Overview & EDA",
        "PCA Analysis",
        "KNN Classifier",
        "Naive Bayes",
        "Linear Regression",
        "Logistic Regression",
        "K-Means Clustering",
        "Prediction Tool"
    ]
)

# Preprocessing
features = ["energy_requirement_mu", "energy_availability_mu", "energy_deficit", "gap"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# ============================================================
# 1️⃣ DATA OVERVIEW & INTERACTIVE EDA
# ============================================================
if page == "Data Overview & EDA":
    st.header("📊 Interactive Exploratory Data Analysis")

    col1, col2, col3 = st.columns(3)
    with col1:
        region = st.selectbox("Select Region", ["All"] + sorted(df["region"].unique().tolist()))
    with col2:
        quarter = st.selectbox("Select Quarter", ["All"] + sorted(df["quarter"].unique().tolist()))
    with col3:
        month = st.selectbox("Select Month", ["All"] + sorted(df["month"].unique().tolist()))

    # Apply filters
    df_filtered = df.copy()
    if region != "All":
        df_filtered = df_filtered[df_filtered["region"] == region]
    if quarter != "All":
        df_filtered = df_filtered[df_filtered["quarter"] == quarter]
    if month != "All":
        df_filtered = df_filtered[df_filtered["month"] == month]

    # KPI metrics
    st.markdown("### 📈 Key Statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg. Energy Requirement", round(df_filtered["energy_requirement_mu"].mean(), 2))
    c2.metric("Avg. Availability", round(df_filtered["energy_availability_mu"].mean(), 2))
    c3.metric("Avg. Deficit", round(df_filtered["energy_deficit"].mean(), 2))

    # Choose plot
    st.markdown("### 🧭 Choose Plot Type")
    plot_type = st.radio("Select Visualization:", ["Line Chart", "Bar Chart", "Scatter Plot"], horizontal=True)

    if plot_type == "Line Chart":
        st.line_chart(df_filtered[["energy_requirement_mu", "energy_availability_mu", "energy_deficit"]])
    elif plot_type == "Bar Chart":
        st.bar_chart(df_filtered[["energy_requirement_mu", "energy_availability_mu", "energy_deficit"]])
    elif plot_type == "Scatter Plot":
        st.scatter_chart(df_filtered, x="energy_requirement_mu", y="energy_availability_mu")

    st.write("### 🧾 Filtered Data Preview")
    st.dataframe(df_filtered.head())

# ============================================================
# 2️⃣ PCA ANALYSIS
# ============================================================
elif page == "PCA Analysis":
    st.header("🔍 Principal Component Analysis (PCA)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
    st.scatter_chart(pd.DataFrame(X_pca, columns=["PCA1", "PCA2"]))

# ============================================================
# 3️⃣ KNN CLASSIFIER
# ============================================================
elif page == "KNN Classifier":
    st.header("🤖 K-Nearest Neighbors (KNN) Classifier")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, df["deficit_flag"], test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text(classification_report(y_test, y_pred))

# ============================================================
# 4️⃣ NAIVE BAYES
# ============================================================
elif page == "Naive Bayes":
    st.header("🧠 Naive Bayes Classifier")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, df["deficit_flag"], test_size=0.3, random_state=42)
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text(classification_report(y_test, y_pred))

# ============================================================
# 5️⃣ LINEAR REGRESSION
# ============================================================
elif page == "Linear Regression":
    st.header("📉 Simple Linear Regression")
    X = df[["energy_requirement_mu", "energy_deficit", "gap"]]
    y = df["energy_availability_mu"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    st.write("R² Score:", r2_score(y_test, y_pred))
    st.line_chart(pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}))

# ============================================================
# 6️⃣ LOGISTIC REGRESSION
# ============================================================
elif page == "Logistic Regression":
    st.header("📈 Logistic Regression Classifier")
    X = df[features]
    y = df["deficit_flag"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)

    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text(classification_report(y_test, y_pred))

# ============================================================
# 7️⃣ K-MEANS CLUSTERING
# ============================================================
elif page == "K-Means Clustering":
    st.header("🎯 K-Means Clustering")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    st.write("Cluster Centers:")
    st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=features))

    cluster_df = pd.DataFrame(X_scaled, columns=features)
    cluster_df["Cluster"] = clusters
    st.scatter_chart(cluster_df, x="energy_requirement_mu", y="energy_availability_mu", color="Cluster")

# ============================================================
# 8️⃣ PREDICTION TOOL
# ============================================================
elif page == "Prediction Tool":
    st.header("🔮 Predict Energy Deficit or Availability")

    st.write("Select model and enter input values to get predictions")

    model_choice = st.selectbox("Choose Model", ["Linear Regression", "Logistic Regression"])
    req = st.number_input("Energy Requirement (MU)", min_value=0.0, value=1000.0)
    avail = st.number_input("Energy Availability (MU)", min_value=0.0, value=950.0)
    deficit = req - avail
    gap = deficit

    input_data = np.array([[req, avail, deficit, gap]])

    if st.button("Predict"):
        if model_choice == "Linear Regression":
            X = df[["energy_requirement_mu", "energy_deficit", "gap"]]
            y = df["energy_availability_mu"]
            model = LinearRegression().fit(X, y)
            pred = model.predict(np.array([[req, deficit, gap]]))
            st.success(f"Predicted Energy Availability: {pred[0]:.2f} MU")

        elif model_choice == "Logistic Regression":
            X = df[features]
            y = df["deficit_flag"]
            model = LogisticRegression(max_iter=1000).fit(X, y)
            pred = model.predict(input_data)
            result = "⚠️ Deficit Expected" if pred[0] == 1 else "✅ No Deficit"
            st.success(result)
