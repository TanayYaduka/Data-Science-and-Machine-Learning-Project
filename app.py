# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import json

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False
    
# ----------------------------
# Page config & simple styling
# ----------------------------
st.set_page_config(page_title="Energy Data ML Dashboard", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #0f172a;
    color: white;
    font-size: 18px;
    padding: 18px;
}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] h1 {
    color: white;
    font-weight: 600;
}
div.stButton > button:first-child {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    padding: 0.6rem 1rem;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Data loader (accept CSV or Excel)
# ----------------------------
@st.cache_data
def load_data(path="cleaned_dataset.csv"):
    # Try CSV then Excel
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_excel(path)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not read {path} as CSV or Excel. Put the cleaned dataset (cleaned_dataset.csv or .xls/.xlsx) next to app.py."
            ) from e

    # Ensure expected columns exist
    expected = {"region", "state", "is_union_territory", "month", "quarter",
                "energy_requirement_mu", "energy_availability_mu", "energy_deficit"}
    missing = expected - set(df.columns)
    if missing:
        raise KeyError(f"Dataset missing required columns: {missing}")

    # Create derived columns
    df["gap"] = df["energy_requirement_mu"] - df["energy_availability_mu"]
    df["deficit_flag"] = (df["energy_deficit"] > 0).astype(int)

    # Ensure numeric types
    for col in ["energy_requirement_mu", "energy_availability_mu", "energy_deficit", "gap"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# load dataset (file name used by you earlier)
try:
    df = load_data("cleaned_dataset.csv")
except Exception:
    # fallback to xls
    df = load_data("cleaned_dataset.xls")

# ----------------------------
# Helper: robust geojson featureidkey picker
# ----------------------------
@st.cache_data
def load_geojson(url):
    raw = requests.get(url, timeout=10)
    geo = raw.json()
    # inspect first feature for property keys
    props = geo.get("features", [{}])[0].get("properties", {})
    # common possible name keys in various India geojson files
    candidates = ["ST_NM", "st_nm", "NAME_1", "state_name", "NAME", "STATE", "st_name"]
    for c in candidates:
        if c in props:
            return geo, f"properties.{c}"
    # fallback: try any property that looks like a name (string)
    for k, v in props.items():
        if isinstance(v, str) and len(v) > 1:
            return geo, f"properties.{k}"
    # final fallback (may fail later)
    return geo, "properties.ST_NM"

geo_url = "https://raw.githubusercontent.com/udit-001/india-maps-data/main/geojson/india.geojson"
try:
    india_geo, featureidkey = load_geojson(geo_url)
except Exception:
    india_geo, featureidkey = None, None

# ----------------------------
# Sidebar navigation
# ----------------------------
st.sidebar.title("⚡ Navigation")
page = st.sidebar.radio("Go to:", ["📘 Dataset Description", "📊 EDA", "🤖 ML Models", "🔮 Prediction"])

# ----------------------------
# Page 1: Dataset Description
# ----------------------------
if page == "📘 Dataset Description":
    st.title("📘 Dataset Description")
    st.markdown(
        "Data source: [India Data Portal](https://indiadataportal.com/p/power/r/mop-power_supply_position-st-mn-aaa)"
    )

    st.subheader("Preview")
    st.dataframe(df.head())

    st.subheader("Dataset info & column descriptions")
    st.write(f"Rows: {df.shape[0]}   |   Columns: {df.shape[1]}")
    st.markdown("""
    **Columns**
    - **region** — geographical region (e.g., North, South). Useful for regional analysis.
    - **state** — state / union territory name. Used in maps and drill-downs.
    - **is_union_territory** — True/False flag.
    - **month** — month name/abbrev (seasonality).
    - **quarter** — financial quarter (Q1..Q4).
    - **energy_requirement_mu** — energy requirement in Million Units (MU) (numeric).
    - **energy_availability_mu** — available energy in MU (numeric).
    - **energy_deficit** — deficit in MU (numeric) — can be target for regression/analysis.
    - **gap** — derived: requirement - availability.
    - **deficit_flag** — derived: (energy_deficit > 0) as 0/1 — for classification tasks.
    """)

    st.subheader("Basic distributions")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Energy Requirement (MU)**")
        st.write(df["energy_requirement_mu"].describe())
    with col2:
        st.markdown("**Energy Availability (MU)**")
        st.write(df["energy_availability_mu"].describe())

# ----------------------------
# Page 2: EDA
# ----------------------------
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    # filters
    col1, col2, col3 = st.columns(3)
    with col1:
        sel_region = st.selectbox("Region", ["All"] + sorted(df["region"].dropna().unique().tolist()))
    with col2:
        sel_quarter = st.selectbox("Quarter", ["All"] + sorted(df["quarter"].dropna().unique().tolist()))
    with col3:
        sel_month = st.selectbox("Month", ["All"] + sorted(df["month"].dropna().unique().tolist()))

    # apply filters
    filtered = df.copy()
    if sel_region != "All":
        filtered = filtered[filtered["region"] == sel_region]
    if sel_quarter != "All":
        filtered = filtered[filtered["quarter"] == sel_quarter]
    if sel_month != "All":
        filtered = filtered[filtered["month"] == sel_month]

    st.subheader("Filtered dataset (top rows)")
    st.dataframe(filtered.head())

    st.subheader("Key numeric statistics")
    st.write("Shape:", filtered.shape)
    st.dataframe(filtered[["energy_requirement_mu", "energy_availability_mu", "energy_deficit", "gap"]].describe())

    # Time-series / month plot - ensure month order if possible
    st.subheader("Energy Requirement vs Availability (by month)")
    # If month values are month names, try to order them
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    if filtered["month"].dtype == object and set(filtered["month"].unique()).issubset(set(month_order)):
        filtered["month"] = pd.Categorical(filtered["month"], categories=month_order, ordered=True)
    fig = px.bar(filtered, x="month", y=["energy_requirement_mu", "energy_availability_mu"],
                 barmode="group", labels={"value":"Energy (MU)", "month":"Month"},
                 title="Monthly Requirement vs Availability (MU)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Energy Deficit distribution by region")
    fig2 = px.box(filtered, x="region", y="energy_deficit", color="region",
                  labels={"energy_deficit":"Deficit (MU)"}, title="Deficit distribution")
    st.plotly_chart(fig2, use_container_width=True)

    # Map
    if india_geo is not None:
        st.subheader("State-wise Energy Deficit (choropleth)")
        state_sum = filtered.groupby("state", dropna=False)["energy_deficit"].sum().reset_index()
        # some state names may not match geojson; user should ensure state names are consistent
        try:
            fig_map = px.choropleth_mapbox(
                state_sum,
                geojson=india_geo,
                locations="state",
                featureidkey=featureidkey,
                color="energy_deficit",
                color_continuous_scale="OrRd",
                mapbox_style="carto-positron",
                zoom=3.5,
                center={"lat": 23.0, "lon": 82.0},
                opacity=0.7,
                labels={"energy_deficit": "Deficit (MU)"},
                title="Total Energy Deficit by State (MU)"
            )
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception as e:
            st.warning("Could not plot map: state names in dataset may not match GeoJSON properties.")
            st.write("GeoJSON property key used:", featureidkey)
            st.write("Example state names in dataset:", state_sum["state"].unique()[:10])
    else:
        st.info("India geojson not available; map disabled.")

# ----------------------------
# Page 3: ML Models (independent of EDA filters)
# ----------------------------
elif page == "🤖 ML Models":
    st.title("🤖 Machine Learning Models and Results")
    st.markdown("Models are trained on the **full cleaned dataset** (no filters). Choose which models to run:")

    model_opts = st.multiselect("Select models:", 
                                ["Linear Regression (Regression)", 
                                 "Decision Tree (Regression)",
                                 "Random Forest (Regression)",
                                 "Gradient Boosting (Regression)",
                                 "XGBoost (Regression)",
                                 "KNN (Classification deficit_flag)",
                                 "Naive Bayes (Classification deficit_flag)",
                                 "Logistic Regression (Classification deficit_flag)",
                                 "K-Means (Clustering)"],
                                default=["Linear Regression (Regression)", "Random Forest (Regression)"])

    # --- Prepare the dataset ---
    df_ml = df.dropna(subset=["energy_requirement_mu","energy_availability_mu","energy_deficit","gap"])
    df_ml["quarter"] = df_ml["quarter"].astype("category")

    # --- One-hot encode quarter for seasonal learning ---
    df_ml = pd.get_dummies(df_ml, columns=["quarter"], drop_first=True)

    # --- Regression features/target ---
    X_reg = df_ml[["energy_requirement_mu","energy_availability_mu","gap"] + 
                  [col for col in df_ml.columns if col.startswith("quarter_")]]
    y_reg = df_ml["energy_deficit"].astype(float)

    # --- Classification features/target ---
    X_clf = X_reg.copy()
    y_clf = df_ml["deficit_flag"].astype(int)

    # --- Train-test splits ---
    Xtr_reg, Xte_reg, ytr_reg, yte_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
    Xtr_clf, Xte_clf, ytr_clf, yte_clf = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)

    # --- Helper RMSE function ---
    def safe_rmse(y_true, y_pred):
        y_true_arr = np.array(y_true, dtype=float).ravel()
        y_pred_arr = np.array(y_pred, dtype=float).ravel()
        mse = np.mean((y_true_arr - y_pred_arr) ** 2)
        return np.sqrt(mse)

    # --- Train and Evaluate Regression Models ---
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    import matplotlib.pyplot as plt

    regression_models = {
        "Linear Regression (Regression)": LinearRegression(),
        "Decision Tree (Regression)": DecisionTreeRegressor(random_state=42),
        "Random Forest (Regression)": RandomForestRegressor(random_state=42),
        "Gradient Boosting (Regression)": GradientBoostingRegressor(random_state=42),
        "XGBoost (Regression)": XGBRegressor(random_state=42, n_estimators=200, learning_rate=0.1)
    }

    model_results = []

    for name, model in regression_models.items():
        if name in model_opts:
            st.subheader(name)
            model.fit(Xtr_reg, ytr_reg)
            preds = model.predict(Xte_reg)
            rmse = safe_rmse(yte_reg, preds)
            r2 = r2_score(yte_reg, preds)
            model_results.append([name, round(rmse, 3), round(r2, 3)])
            st.write(f"RMSE: {rmse:.3f} | R²: {r2:.3f}")

            # Predicted vs Actual Plot
            fig, ax = plt.subplots()
            ax.scatter(yte_reg, preds, alpha=0.7, color='teal')
            ax.plot([yte_reg.min(), yte_reg.max()], [yte_reg.min(), yte_reg.max()], 'r--')
            ax.set_xlabel("Actual Energy Deficit")
            ax.set_ylabel("Predicted Energy Deficit")
            ax.set_title(f"Predicted vs Actual: {name}")
            st.pyplot(fig)

    # Display results table
    if model_results:
        st.write("### 📊 Regression Model Summary")
        st.dataframe(pd.DataFrame(model_results, columns=["Model", "RMSE", "R²"]))

    # --- Classification Models ---
    if "KNN (Classification deficit_flag)" in model_opts:
        st.subheader("KNN Classifier (deficit_flag)")
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(Xtr_clf, ytr_clf)
        pred_knn = knn.predict(Xte_clf)
        st.write("Accuracy:", round(accuracy_score(yte_clf, pred_knn), 3))
        st.text(classification_report(yte_clf, pred_knn))

    if "Naive Bayes (Classification deficit_flag)" in model_opts:
        st.subheader("Gaussian Naive Bayes (deficit_flag)")
        nb = GaussianNB()
        nb.fit(Xtr_clf, ytr_clf)
        pred_nb = nb.predict(Xte_clf)
        st.write("Accuracy:", round(accuracy_score(yte_clf, pred_nb), 3))
        st.text(classification_report(yte_clf, pred_nb))

    if "Logistic Regression (Classification deficit_flag)" in model_opts:
        st.subheader("Logistic Regression (deficit_flag)")
        logr = LogisticRegression(max_iter=1000)
        logr.fit(Xtr_clf, ytr_clf)
        pred_logr = logr.predict(Xte_clf)
        st.write("Accuracy:", round(accuracy_score(yte_clf, pred_logr), 3))
        st.text(classification_report(yte_clf, pred_logr))

    # --- Clustering ---
    if "K-Means (Clustering)" in model_opts:
        st.subheader("K-Means Clustering")
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_reg)
        df_cluster = X_reg.copy()
        df_cluster["Cluster"] = clusters
        st.dataframe(df_cluster.head())

# ----------------------------
# Page 4: Prediction (user inputs)
# ----------------------------
elif page == "🔮 Prediction":
    st.title("🔮 Predict Energy Deficit / Surplus")
    st.markdown("""
    This prediction uses models trained on the **full dataset** (no EDA filters).  
    Choose the model, provide inputs, and see if the state is likely to have **deficit** or **surplus**.
    """)

    # User selects state, quarter, and model
    col1, col2, col3 = st.columns(3)
    with col1:
        state_in = st.selectbox("State", sorted(df["state"].dropna().unique()))
    with col2:
        quarter_in = st.selectbox("Quarter", sorted(df["quarter"].dropna().unique()))
    with col3:
        model_choice = st.selectbox("Model for prediction", [
            "Linear Regression", 
            "Random Forest Regressor", 
            "Decision Tree Regressor", 
            "Gradient Boosting Regressor"
        ])

    # Numeric inputs
    nr1, nr2 = st.columns(2)
    with nr1:
        req_in = st.number_input(
            "Energy Requirement (MU)", 
            min_value=0.0, 
            value=float(df["energy_requirement_mu"].median()), 
            step=100.0
        )
    with nr2:
        avail_in = st.number_input(
            "Energy Availability (MU)", 
            min_value=0.0, 
            value=float(df["energy_availability_mu"].median()), 
            step=100.0
        )

    if st.button("Predict"):
        # Prepare dataset for training
        df_train = df.dropna(subset=["energy_requirement_mu","energy_availability_mu","energy_deficit","gap"])
        X_full = df_train[["energy_requirement_mu","energy_availability_mu","gap"]]
        y_full = df_train["energy_deficit"].astype(float)

        # Input gap
        gap_in = req_in - avail_in
        x_input = np.array([[req_in, avail_in, gap_in]])

        # Train selected model
        if model_choice == "Linear Regression":
            model = LinearRegression().fit(X_full, y_full)
        elif model_choice == "Random Forest Regressor":
            model = RandomForestRegressor(random_state=42).fit(X_full, y_full)
        elif model_choice == "Decision Tree Regressor":
            model = DecisionTreeRegressor(random_state=42).fit(X_full, y_full)
        elif model_choice == "Gradient Boosting Regressor":
            model = GradientBoostingRegressor(random_state=42).fit(X_full, y_full)

        # Predict
        pred_val = model.predict(x_input)[0]

        # Display prediction with interpretation
        if pred_val > 0:
            st.error(f"⚠️ Predicted Energy Deficit for {state_in} ({quarter_in}): {pred_val:.2f} MU")
        else:
            st.success(f"✅ Predicted Energy Surplus for {state_in} ({quarter_in}): {abs(pred_val):.2f} MU")

        st.caption("Model trained on full dataset. Negative value indicates surplus energy available.")

