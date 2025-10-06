# app.py
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

# -------------------------------------------------------
# 🧩 Load dataset
# -------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.xls")
    df["gap"] = df["energy_requirement_mu"] - df["energy_availability_mu"]
    df["deficit_flag"] = (df["energy_deficit"] > 0).astype(int)
    return df

df = load_data()

st.set_page_config(
    page_title="⚡ Energy Data ML Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Energy Data ML Dashboard")
st.markdown("""
This Streamlit app performs:
- **Exploratory Analysis**
- **Dimensionality Reduction (PCA)**
- **Classification (KNN, Naive Bayes, Logistic Regression)**
- **Regression (Linear)**
- **Clustering (K-Means)**
""")

# Sidebar
choice = st.sidebar.selectbox(
    "Choose Analysis Type:",
    [
        "Data Overview",
        "PCA",
        "KNN Classifier",
        "Naive Bayes",
        "Linear Regression",
        "Logistic Regression",
        "K-Means Clustering"
    ]
)

# Common preprocessing
features = ["energy_requirement_mu", "energy_availability_mu", "energy_deficit", "gap"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# -------------------------------------------------------
# 🧾 1. Data Overview
# -------------------------------------------------------
if choice == "Data Overview":
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())
    
    st.markdown("### Dataset Info")
    st.write("Shape:", df.shape)
    st.write("Number of Missing Values per Column:")
    st.write(df.isnull().sum())

    st.markdown("### Numeric Summary")
    st.dataframe(df.describe())

    st.markdown("### Correlation Heatmap")
    st.bar_chart(df[features].corr()["energy_deficit"])

# -------------------------------------------------------
# 🌀 2. PCA Visualization
# -------------------------------------------------------
elif choice == "PCA":
    st.subheader("🌀 PCA Visualization (2 Components)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
    
    st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
    st.scatter_chart(pca_df)

# -------------------------------------------------------
# 🎯 3. KNN Classifier
# -------------------------------------------------------
elif choice == "KNN Classifier":
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, df["deficit_flag"], test_size=0.3, random_state=42
    )
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    st.subheader("🎯 KNN Classification Results")
    st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    st.text(classification_report(y_test, y_pred))

# -------------------------------------------------------
# 📊 4. Naive Bayes Classifier
# -------------------------------------------------------
elif choice == "Naive Bayes":
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, df["deficit_flag"], test_size=0.3, random_state=42
    )
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    st.subheader("📊 Naive Bayes Classification Results")
    st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    st.text(classification_report(y_test, y_pred))

# -------------------------------------------------------
# 📈 5. Linear Regression
# -------------------------------------------------------
elif choice == "Linear Regression":
    X = df[["energy_requirement_mu", "energy_deficit", "gap"]]
    y = df["energy_availability_mu"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    st.subheader("📈 Linear Regression Results")
    st.write("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 3))
    st.write("R² Score:", round(r2_score(y_test, y_pred), 3))

    comparison_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    st.line_chart(comparison_df)

# -------------------------------------------------------
# 🧠 6. Logistic Regression
# -------------------------------------------------------
elif choice == "Logistic Regression":
    X = df[["energy_requirement_mu", "energy_availability_mu", "energy_deficit", "gap"]]
    y = df["deficit_flag"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)

    st.subheader("🧠 Logistic Regression Results")
    st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    st.text(classification_report(y_test, y_pred))

# -------------------------------------------------------
# 🔹 7. K-Means Clustering
# -------------------------------------------------------
elif choice == "K-Means Clustering":
    st.subheader("🔹 K-Means Clustering Results")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df_clustered = df.copy()
    df_clustered["Cluster"] = clusters
    st.write("Cluster Centers:")
    st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=features))

    st.markdown("### Cluster Distribution")
    st.bar_chart(df_clustered["Cluster"].value_counts())
    st.dataframe(df_clustered.head())

st.markdown("---")
st.markdown("Made with ❤️ using **Streamlit** | Author: *Your Name*")

