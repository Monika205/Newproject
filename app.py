

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "sales_forecasting_bundle.joblib"
)


# =========================================================
# 2. LOAD MODEL BUNDLE
# =========================================================
@st.cache_resource
def load_bundle(path):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Keep "
        "'sales_forecasting_bundle.joblib' in the same folder as app.py."
    )
    st.stop()

try:
    bundle = load_bundle(str(MODEL_PATH))
    model = bundle["model"]
    monthly_sales = bundle["monthly_sales"]
    feature_columns = list(bundle["feature_columns"])

except Exception as error:
    st.error(f"Unable to load model bundle: {error}")
    st.stop()


# =========================================================
# 3. PREPARE MONTHLY SALES
# =========================================================
if isinstance(monthly_sales, pd.DataFrame):
    if monthly_sales.shape[1] != 1:
        st.error("monthly_sales must contain one sales column.")
        st.stop()

    monthly_sales = monthly_sales.iloc[:, 0]

monthly_sales = pd.Series(monthly_sales).copy()
monthly_sales.index = pd.to_datetime(monthly_sales.index)
monthly_sales = monthly_sales.astype(float).dropna()

# Normalize each date to the first day of its month
