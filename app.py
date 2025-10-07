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
    st.title("🤖 Machine Learning Models and Results (Seasonality Aware)")
    st.markdown("""
    Models are trained on the **full cleaned dataset** (no filters).  
    Regression models are tested **with and without quarter features** to see the impact of seasonality.
    """)

    model_opts = st.multiselect("Select models:", 
                                ["Linear Regression (Regression)", 
                                 "Decision Tree (Regression)",
                                 "Random Forest (Regression)",
                                 "Gradient Boosting (Regression)",
                                 "XGBoost (Regression)"],
                                default=["Linear Regression (Regression)", "Random Forest (Regression)"])

    # --- Prepare dataset ---
    df_ml = df.dropna(subset=["energy_requirement_mu","energy_availability_mu","energy_deficit","gap"])
    df_ml["quarter"] = df_ml["quarter"].astype("category")

    # Base features
    base_features = ["energy_requirement_mu","energy_availability_mu","gap"]
    X_base = df_ml[base_features]
    y_reg = df_ml["energy_deficit"].astype(float)

    # Seasonality-aware features (one-hot encode quarters)
    df_season = pd.get_dummies(df_ml, columns=["quarter"], drop_first=True)
    seasonal_features = base_features + [col for col in df_season.columns if col.startswith("quarter_")]
    X_season = df_season[seasonal_features]

    # Train-test split
    Xtr_base, Xte_base, ytr_base, yte_base = train_test_split(X_base, y_reg, test_size=0.3, random_state=42)
    Xtr_season, Xte_season, ytr_season, yte_season = train_test_split(X_season, y_reg, test_size=0.3, random_state=42)

    # Helper RMSE function
    def safe_rmse(y_true, y_pred):
        y_true_arr = np.array(y_true, dtype=float).ravel()
        y_pred_arr = np.array(y_pred, dtype=float).ravel()
        mse = np.mean((y_true_arr - y_pred_arr) ** 2)
        return np.sqrt(mse)

    # Import regression models
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    import matplotlib.pyplot as plt

    regression_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42, n_estimators=200, learning_rate=0.1)
    }

    results_list = []

    for name, model in regression_models.items():
        if name in model_opts:
            st.subheader(f"{name} (Base Features)")
            model.fit(Xtr_base, ytr_base)
            preds_base = model.predict(Xte_base)
            rmse_base = safe_rmse(yte_base, preds_base)
            r2_base = r2_score(yte_base, preds_base)
            st.write(f"RMSE: {rmse_base:.3f} | R²: {r2_base:.3f}")

            # Plot predicted vs actual
            fig, ax = plt.subplots()
            ax.scatter(yte_base, preds_base, alpha=0.7, color='teal')
            ax.plot([yte_base.min(), yte_base.max()], [yte_base.min(), yte_base.max()], 'r--')
            ax.set_xlabel("Actual Energy Deficit")
            ax.set_ylabel("Predicted Energy Deficit")
            ax.set_title(f"Predicted vs Actual ({name} - Base)")
            st.pyplot(fig)

            # Seasonality-aware
            st.subheader(f"{name} (Seasonality-Aware)")
            model.fit(Xtr_season, ytr_season)
            preds_season = model.predict(Xte_season)
            rmse_season = safe_rmse(yte_season, preds_season)
            r2_season = r2_score(yte_season, preds_season)
            st.write(f"RMSE: {rmse_season:.3f} | R²: {r2_season:.3f}")

            # Plot predicted vs actual (seasonality-aware)
            fig2, ax2 = plt.subplots()
            ax2.scatter(yte_season, preds_season, alpha=0.7, color='orange')
            ax2.plot([yte_season.min(), yte_season.max()], [yte_season.min(), yte_season.max()], 'r--')
            ax2.set_xlabel("Actual Energy Deficit")
            ax2.set_ylabel("Predicted Energy Deficit")
            ax2.set_title(f"Predicted vs Actual ({name} - Seasonality)")
            st.pyplot(fig2)

            # Store results
            results_list.append([name, round(rmse_base,3), round(r2_base,3), round(rmse_season,3), round(r2_season,3)])

    # Show summary table
    if results_list:
        st.write("### 📊 Regression Model Comparison (Base vs Seasonality-Aware)")
        st.dataframe(pd.DataFrame(results_list, columns=["Model","RMSE_Base","R2_Base","RMSE_Season","R2_Season"]))


# ----------------------------
# Page 4: Prediction (user inputs with seasonal check)
# ----------------------------
elif page == "🔮 Prediction":
    st.title("🔮 Predict Energy Deficit / Surplus (Seasonality Aware)")
    st.markdown("""
    This prediction uses models trained on the **full dataset**.  
    You can provide inputs, select a model, and check **seasonal energy patterns**.
    """)

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
            step=50.0
        )
    with nr2:
        avail_in = st.number_input(
            "Energy Availability (MU)", 
            min_value=0.0, 
            value=float(df["energy_availability_mu"].median()), 
            step=50.0
        )

    if st.button("Predict"):
        # Train model on full dataset
        df_train = df.dropna(subset=["energy_requirement_mu","energy_availability_mu","energy_deficit","gap"])
        X_full = df_train[["energy_requirement_mu","energy_availability_mu","gap"]]
        y_full = df_train["energy_deficit"].astype(float)
        gap_in = req_in - avail_in
        x_input = np.array([[req_in, avail_in, gap_in]])

        # Train chosen model
        if model_choice == "Linear Regression":
            model = LinearRegression().fit(X_full, y_full)
        elif model_choice == "Random Forest Regressor":
            model = RandomForestRegressor(random_state=42).fit(X_full, y_full)
        elif model_choice == "Decision Tree Regressor":
            model = DecisionTreeRegressor(random_state=42).fit(X_full, y_full)
        elif model_choice == "Gradient Boosting Regressor":
            model = GradientBoostingRegressor(random_state=42).fit(X_full, y_full)

        # Predict deficit/surplus
        pred_val = model.predict(x_input)[0]

        # ---- Seasonal check ----
        seasonal_median = df[(df["state"]==state_in) & (df["quarter"]==quarter_in)]["energy_requirement_mu"].median()
        if req_in > seasonal_median * 1.2:  # 20% above typical
            st.warning(f"⚠️ Energy requirement is unusually high for {state_in} in {quarter_in} (Expected ~{seasonal_median:.1f} MU).")
        elif req_in < seasonal_median * 0.8:  # 20% below typical
            st.info(f"ℹ️ Energy requirement is unusually low for {state_in} in {quarter_in} (Expected ~{seasonal_median:.1f} MU).")

        # ---- Display predicted deficit/surplus ----
        if pred_val > 0:
            st.error(f"Predicted Energy Deficit for {state_in} ({quarter_in}): {pred_val:.2f} MU")
        else:
            st.success(f"Predicted Energy Surplus for {state_in} ({quarter_in}): {abs(pred_val):.2f} MU")

        



