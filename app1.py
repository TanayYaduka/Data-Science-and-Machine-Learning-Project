# -------------------------------------------------
# app.py — Energy Forecasting Dashboard
# -------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.set_page_config(page_title="Energy Forecasting Dashboard", layout="wide")

# -------------------------------------------------
# 1️⃣ Upload & Describe Dataset
# -------------------------------------------------
st.title("⚡ Energy Forecasting — Base vs Seasonality Models")

uploaded_file = st.file_uploader("Upload your cleaned dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully!")
    st.write("### Dataset Preview:")
    st.dataframe(df.head())

    st.write("#### Dataset Description:")
    st.write(df.describe())

    # -------------------------------------------------
    # 2️⃣ EDA
    # -------------------------------------------------
    st.header("🔍 Exploratory Data Analysis (EDA)")

    # Distribution plots
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(df["energy_requirement_mu"], kde=True, ax=axs[0])
    sns.histplot(df["energy_availability_mu"], kde=True, ax=axs[1])
    sns.histplot(df["energy_deficit"], kde=True, ax=axs[2])
    axs[0].set_title("Energy Requirement Distribution")
    axs[1].set_title("Energy Availability Distribution")
    axs[2].set_title("Energy Deficit Distribution")
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(6, 4))
    sns.heatmap(df[["energy_requirement_mu", "energy_availability_mu", "energy_deficit"]].corr(),
                annot=True, cmap="YlGnBu", fmt=".2f")
    st.pyplot(plt)

    # -------------------------------------------------
    # 3️⃣ Feature Preparation
    # -------------------------------------------------
    st.header("⚙️ Feature Preparation")

    base_features = ["energy_requirement_mu", "energy_availability_mu", "gap"]
    season_features_numeric = base_features
    season_features_cat = ["month", "quarter"]

    # One-hot encode categorical month/quarter
    df_season = pd.get_dummies(df[season_features_numeric + season_features_cat], drop_first=True)

    # Targets
    X_base = df[base_features].values
    y_class = df["deficit_flag"].values
    y_reg = df["energy_deficit"].values
    X_season = df_season.values

    # Train-test split
    X_train_base_clf, X_test_base_clf, y_train_clf, y_test_clf = train_test_split(
        X_base, y_class, test_size=0.3, random_state=42)
    X_train_base_reg, X_test_base_reg, y_train_reg, y_test_reg = train_test_split(
        X_base, y_reg, test_size=0.3, random_state=42)
    X_train_season_clf, X_test_season_clf, _, _ = train_test_split(
        X_season, y_class, test_size=0.3, random_state=42)
    X_train_season_reg, X_test_season_reg, _, _ = train_test_split(
        X_season, y_reg, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_base_scaled = scaler.fit_transform(X_base)
    X_season_scaled = scaler.fit_transform(X_season)

    # -------------------------------------------------
    # 4️⃣ ML Models
    # -------------------------------------------------
    st.header("🤖 Machine Learning Models")

    # Helper to compute RMSE manually
    def compute_rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    # ---------------- Linear Regression ----------------
    st.subheader("Linear Regression")
    lr_base = LinearRegression().fit(X_train_base_reg, y_train_reg)
    lr_season = LinearRegression().fit(X_train_season_reg, y_train_reg)
    y_pred_lr_base = lr_base.predict(X_test_base_reg)
    y_pred_lr_season = lr_season.predict(X_test_season_reg)
    st.write(f"Base RMSE: {compute_rmse(y_test_reg, y_pred_lr_base):.3f}")
    st.write(f"Seasonality RMSE: {compute_rmse(y_test_reg, y_pred_lr_season):.3f}")

    # ---------------- KNN ----------------
    st.subheader("K-Nearest Neighbors (Classification)")
    knn_base = KNeighborsClassifier(n_neighbors=5).fit(X_train_base_clf, y_train_clf)
    knn_season = KNeighborsClassifier(n_neighbors=5).fit(X_train_season_clf, y_train_clf)
    y_pred_knn_base = knn_base.predict(X_test_base_clf)
    y_pred_knn_season = knn_season.predict(X_test_season_clf)
    st.write(f"KNN Base Accuracy: {accuracy_score(y_test_clf, y_pred_knn_base):.3f}")
    st.write(f"KNN Seasonality Accuracy: {accuracy_score(y_test_clf, y_pred_knn_season):.3f}")

    # ---------------- Naive Bayes ----------------
    st.subheader("Naive Bayes (Classification)")
    nb_base = GaussianNB().fit(X_train_base_clf, y_train_clf)
    nb_season = GaussianNB().fit(X_train_season_clf, y_train_clf)
    y_pred_nb_base = nb_base.predict(X_test_base_clf)
    y_pred_nb_season = nb_season.predict(X_test_season_clf)
    st.write(f"NB Base Accuracy: {accuracy_score(y_test_clf, y_pred_nb_base):.3f}")
    st.write(f"NB Seasonality Accuracy: {accuracy_score(y_test_clf, y_pred_nb_season):.3f}")

    # ---------------- Logistic Regression ----------------
    st.subheader("Logistic Regression")
    log_base = LogisticRegression(max_iter=1000).fit(X_train_base_clf, y_train_clf)
    log_season = LogisticRegression(max_iter=1000).fit(X_train_season_clf, y_train_clf)
    y_pred_log_base = log_base.predict(X_test_base_clf)
    y_pred_log_season = log_season.predict(X_test_season_clf)
    st.write(f"Base Accuracy: {accuracy_score(y_test_clf, y_pred_log_base):.3f}")
    st.write(f"Seasonality Accuracy: {accuracy_score(y_test_clf, y_pred_log_season):.3f}")

    # ---------------- K-Means Clustering ----------------
    st.subheader("K-Means Clustering")
    kmeans_base = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_base_scaled)
    kmeans_season = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_season_scaled)
    df["cluster_base"] = kmeans_base.labels_
    df["cluster_season"] = kmeans_season.labels_

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(x=X_base_scaled[:, 0], y=X_base_scaled[:, 1],
                    hue=kmeans_base.labels_, palette="Set2", ax=ax[0])
    ax[0].set_title("K-Means Base Clustering")
    sns.scatterplot(x=X_season_scaled[:, 0], y=X_season_scaled[:, 1],
                    hue=kmeans_season.labels_, palette="Set1", ax=ax[1])
    ax[1].set_title("K-Means Seasonality Clustering")
    st.pyplot(fig)

    # ---------------- SVM ----------------
    st.subheader("Support Vector Machine (SVM)")
    svm_base = SVC(kernel='rbf').fit(X_train_base_clf, y_train_clf)
    svm_season = SVC(kernel='rbf').fit(X_train_season_clf, y_train_clf)
    st.write(f"SVM Base Accuracy: {accuracy_score(y_test_clf, svm_base.predict(X_test_base_clf)):.3f}")
    st.write(f"SVM Seasonality Accuracy: {accuracy_score(y_test_clf, svm_season.predict(X_test_season_clf)):.3f}")

    # ---------------- Random Forest ----------------
    st.subheader("Random Forest")
    rf_base = RandomForestClassifier(random_state=42).fit(X_train_base_clf, y_train_clf)
    rf_season = RandomForestClassifier(random_state=42).fit(X_train_season_clf, y_train_clf)
    st.write(f"RF Base Accuracy: {accuracy_score(y_test_clf, rf_base.predict(X_test_base_clf)):.3f}")
    st.write(f"RF Seasonality Accuracy: {accuracy_score(y_test_clf, rf_season.predict(X_test_season_clf)):.3f}")

    # ---------------- Decision Tree ----------------
    st.subheader("Decision Tree")
    dt_base = DecisionTreeClassifier(random_state=42).fit(X_train_base_clf, y_train_clf)
    dt_season = DecisionTreeClassifier(random_state=42).fit(X_train_season_clf, y_train_clf)
    st.write(f"Decision Tree Base Accuracy: {accuracy_score(y_test_clf, dt_base.predict(X_test_base_clf)):.3f}")
    st.write(f"Decision Tree Seasonality Accuracy: {accuracy_score(y_test_clf, dt_season.predict(X_test_season_clf)):.3f}")

    # ---------------- XGBoost ----------------
    st.subheader("XGBoost")
    xgb_base = XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X_train_base_clf, y_train_clf)
    xgb_season = XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X_train_season_clf, y_train_clf)
    st.write(f"XGBoost Base Accuracy: {accuracy_score(y_test_clf, xgb_base.predict(X_test_base_clf)):.3f}")
    st.write(f"XGBoost Seasonality Accuracy: {accuracy_score(y_test_clf, xgb_season.predict(X_test_season_clf)):.3f}")

    # ---------------- Neural Network ----------------
    st.subheader("Neural Network (Keras Sequential)")
    nn_base = Sequential([
        Dense(16, activation='relu', input_shape=(X_train_base_clf.shape[1],)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    nn_base.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    nn_base.fit(X_train_base_clf, y_train_clf, epochs=20, batch_size=8, verbose=0)
    loss, acc = nn_base.evaluate(X_test_base_clf, y_test_clf, verbose=0)
    st.write(f"Neural Network Base Accuracy: {acc:.3f}")

    # -------------------------------------------------
    # 5️⃣ User Prediction
    # -------------------------------------------------
    st.header("🔮 User-Based Prediction")

    st.sidebar.title("Enter Energy Inputs")
    req = st.sidebar.number_input("Energy Requirement (mu)", min_value=0.0)
    avail = st.sidebar.number_input("Energy Availability (mu)", min_value=0.0)
    gap = st.sidebar.number_input("Gap", min_value=-100.0, max_value=100.0)

    month = st.sidebar.selectbox("Month", sorted(df["month"].unique()))
    quarter = st.sidebar.selectbox("Quarter", sorted(df["quarter"].unique()))

    user_base = np.array([[req, avail, gap]])
    user_season = pd.DataFrame([[req, avail, gap, month, quarter]],
                               columns=["energy_requirement_mu", "energy_availability_mu", "gap", "month", "quarter"])
    user_season = pd.get_dummies(user_season, drop_first=True).reindex(columns=df_season.columns, fill_value=0)

    if st.sidebar.button("Predict Energy Deficit"):
        pred_base = lr_base.predict(user_base)[0]
        pred_season = lr_season.predict(user_season)[0]
        st.success(f"🔹 Base Model Predicted Deficit: {pred_base:.3f}")
        st.success(f"🔸 Seasonality Model Predicted Deficit: {pred_season:.3f}")
else:
    st.info("👆 Please upload your cleaned dataset to begin.")
