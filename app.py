import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="Energy Data ML Dashboard", layout="wide")

# ----------------------------
# Custom Sidebar Styling
# ----------------------------
st.markdown("""
    <style>
    /* Sidebar background and style */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        color: white;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
        color: white;
    }
    [data-testid="stSidebar"] .stSelectbox, [data-testid="stSidebar"] .stRadio {
        background-color: #334155;
        border-radius: 8px;
        padding: 10px;
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
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_energy_data.csv")  # replace with your file
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
    ["📊 Exploratory Data Analysis", "🤖 Model Comparison", "🔮 Prediction"],
    index=0
)

# ----------------------------
# Exploratory Data Analysis (EDA)
# ----------------------------
if menu == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("""
    Explore the dataset interactively.  
    You can view trends in **Energy Requirement**, **Availability**, and **Deficit (in Million Units)** 
    based on **Region**, **State**, **Month**, or **Quarter**.
    """)

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

    # ---- Step 1: Show Filtered Dataset ----
    st.subheader("📋 Filtered Dataset Preview")
    st.dataframe(filtered_df.head())

    # ---- Step 2: Show Key Statistics ----
    st.subheader("📊 Key Statistics")
    st.write("Shape:", filtered_df.shape)
    st.write("Numeric Summary:")
    st.dataframe(filtered_df[["energy_requirement_mu", "energy_availability_mu", "energy_deficit"]].describe())

    # ---- Step 3: Plots ----
    # Energy Requirement vs Availability
    st.subheader("⚙️ Energy Requirement vs Availability")
    fig1 = px.bar(
        filtered_df,
        x="month",
        y=["energy_requirement_mu", "energy_availability_mu"],
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={"value": "Energy (Million Units - MU)", "month": "Month"},
        title="Monthly Energy Requirement vs Availability (in MU)"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Energy Deficit by Region
    st.subheader("💡 Energy Deficit Distribution")
    fig2 = px.box(
        filtered_df,
        x="region",
        y="energy_deficit",
        color="region",
        labels={"energy_deficit": "Deficit (MU)", "region": "Region"},
        title="Distribution of Energy Deficit Across Regions (in MU)"
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

# ----------------------------
# Model Comparison Section
# ----------------------------
elif menu == "🤖 Model Comparison":
    st.title("🤖 Model Comparison and Performance")
    st.markdown("""
    Below is a comparison of the performance of multiple Machine Learning models 
    used to predict **Energy Deficit (MU)**.
    """)

    model_results = pd.read_csv("model_comparison_results.csv")  # Replace with your file
    st.dataframe(model_results)

    fig = px.bar(
        model_results,
        x="Model",
        y="R2_Score",
        color="Model",
        title="Model Performance Comparison (R² Score)",
        labels={"R2_Score": "R² Score"}
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Prediction Section
# ----------------------------
elif menu == "🔮 Prediction":
    st.title("🔮 Predict Energy Deficit")

    st.markdown("""
    Choose a model and input parameters below to predict **Energy Deficit (MU)**.
    """)

    model_choice = st.selectbox(
        "Choose Model:",
        ["Linear Regression", "Random Forest", "XGBoost"]
    )

    # Load models
    models = {
        "Linear Regression": "linear_model.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost_model.pkl"
    }

    model = joblib.load(models[model_choice])

    # User input form
    with st.form("prediction_form"):
        st.subheader("Enter Input Values")
        c1, c2 = st.columns(2)
        with c1:
            energy_req = st.number_input("Energy Requirement (MU)", min_value=0.0, step=100.0)
        with c2:
            energy_avail = st.number_input("Energy Availability (MU)", min_value=0.0, step=100.0)
        submitted = st.form_submit_button("Predict")

        if submitted:
            input_df = pd.DataFrame([[energy_req, energy_avail]], columns=["energy_requirement_mu", "energy_availability_mu"])
            prediction = model.predict(input_df)[0]
            st.success(f"✅ Predicted Energy Deficit: **{prediction:.2f} MU**")
