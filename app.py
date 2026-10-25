
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
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

FORECAST_HORIZON = 3

# ==================================================
# 2. CUSTOM STYLING
# ==================================================
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.18);
    }

    .info-box {
        padding: 14px;
        border-radius: 10px;
        background-color: rgba(100, 149, 237, 0.10);
        border-left: 4px solid #6495ED;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. LOAD TRAINED MODEL BUNDLE
# ==================================================
@st.cache_resource
def load_bundle(path):
    return joblib.load(path)


st.title("📈 Sales Forecasting Dashboard")

st.caption(
    "Monthly sales analysis and future sales forecasting "
    "using a trained Ridge Regression model."
)

st.markdown("""
<div class="info-box">
<b>Forecasting period:</b> The dashboard uses historical monthly
sales from the dataset and automatically forecasts the next
three months. No manual year or date selection is required.
</div>
""", unsafe_allow_html=True)


if not MODEL_PATH.exists():
    st.error(
        "Model bundle not found. Please keep "
        "'sales_forecasting_bundle.joblib' in the same folder as app.py."
    )
    st.stop()

try:
    bundle = load_bundle(str(MODEL_PATH))

    model = bundle["model"]
    monthly_sales = bundle["monthly_sales"]
    feature_columns = list(bundle["feature_columns"])

except Exception as error:
    st.error(f"Unable to load the trained model bundle: {error}")
    st.stop()


# ==================================================
# 4. PREPARE HISTORICAL MONTHLY SALES
# ==================================================
if isinstance(monthly_sales, pd.DataFrame):
    if monthly_sales.shape[1] != 1:
        st.error(
            "The monthly_sales data must contain only one sales column."
        )
        st.stop()

    monthly_sales = monthly_sales.iloc[:, 0]

monthly_sales = pd.Series(monthly_sales).copy()

monthly_sales.index = pd.to_datetime(
    monthly_sales.index,
    errors="coerce"
)

monthly_sales = monthly_sales[
    ~monthly_sales.index.isna()
]

monthly_sales = monthly_sales.astype(float).dropna()

# Normalize dates to the first day of each month.
monthly_sales.index = (
    monthly_sales.index.to_period("M").to_timestamp()
)

monthly_sales = (
    monthly_sales.groupby(level=0)
    .sum()
    .sort_index()
)

if len(monthly_sales) < 12:
    st.error(
        "At least 12 months of historical sales are required "
        "for the lag and rolling features."
    )
    st.stop()

first_historical_month = monthly_sales.index.min()
last_historical_month = monthly_sales.index.max()

forecast_start = (
    last_historical_month + pd.offsets.MonthBegin(1)
)

forecast_end = (
    forecast_start + pd.DateOffset(months=FORECAST_HORIZON - 1)
)

# Expected dataset: January 2011 to September 2014.
# Therefore, the forecast should cover October to December 2014.
if (
    first_historical_month.year != 2011
    or last_historical_month != pd.Timestamp("2014-09-01")
):
    st.warning(
        "The loaded dataset does not match the expected period "
        "January 2011 to September 2014. Forecasts will still start "
        "after the last available historical month."
    )


# ==================================================
# 5. FORECAST FUNCTION
# ==================================================
def generate_forecast(
    model,
    historical_sales,
    feature_columns,
    horizon
):
    """
    Generate recursive monthly forecasts using the saved model.

    Feature formulas must match those used during model training.
    """
    history = (
        historical_sales.copy()
        .astype(float)
        .sort_index()
    )

    predictions = []

    expected_features = [
        "month",
        "year_number",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_12",
        "rolling_mean_3",
        "rolling_mean_12"
    ]

    missing_features = [
        col for col in expected_features
        if col not in feature_columns
    ]

    unexpected_features = [
        col for col in feature_columns
        if col not in expected_features
    ]

    if missing_features or unexpected_features:
        raise ValueError(
            f"Feature mismatch. Missing: {missing_features}; "
            f"unexpected: {unexpected_features}"
        )

    for _ in range(horizon):

        next_month = (
            history.index.max() + pd.offsets.MonthBegin(1)
        )

        features = {
            "month": next_month.month,

            "year_number": (
                next_month.year - historical_sales.index.min().year + 1
            ),

            "lag_1": history.iloc[-1],
            "lag_2": history.iloc[-2],
            "lag_3": history.iloc[-3],
            "lag_12": history.iloc[-12],

            "rolling_mean_3": history.iloc[-3:].mean(),
            "rolling_mean_12": history.iloc[-12:].mean()
        }

        input_data = pd.DataFrame(
            [[features[col] for col in feature_columns]],
            columns=feature_columns
        )

        predicted_sales = float(
            model.predict(input_data)[0]
        )

        # Sales cannot be negative.
        predicted_sales = max(0.0, predicted_sales)

        predictions.append({
            "Month": next_month,
            "Predicted Sales": predicted_sales
        })

        # Use the prediction to calculate the following month's features.
        history.loc[next_month] = predicted_sales

    return pd.DataFrame(predictions)


# ==================================================
# 6. DATASET INFORMATION
# ==================================================
st.sidebar.header("Dataset Information")

st.sidebar.write(
    f"**First recorded month:** "
    f"{first_historical_month.strftime('%B %Y')}"
)

st.sidebar.write(
    f"**Last recorded month:** "
    f"{last_historical_month.strftime('%B %Y')}"
)

st.sidebar.write(
    f"**Historical months:** {len(monthly_sales)}"
)

st.sidebar.write(
    f"**Forecast period:** "
    f"{forecast_start.strftime('%B %Y')} – "
    f"{forecast_end.strftime('%B %Y')}"
)

st.sidebar.info(
    "The forecast automatically begins after the last "
    "month available in the dataset."
)


# ==================================================
# 7. OVERVIEW METRICS
# ==================================================
total_historical_sales = monthly_sales.sum()
average_monthly_sales = monthly_sales.mean()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Historical Sales",
    f"{total_historical_sales:,.2f}"
)

col2.metric(
    "Average Monthly Sales",
    f"{average_monthly_sales:,.2f}"
)

col3.metric(
    "Historical Months",
    f"{len(monthly_sales)}"
)


# ==================================================
# 8. HISTORICAL SALES CHART
# ==================================================
st.divider()
st.subheader("Historical Monthly Sales")

historical_chart = monthly_sales.rename("Sales").reset_index()
historical_chart.columns = ["Month", "Sales"]

st.line_chart(
    historical_chart,
    x="Month",
    y="Sales",
    use_container_width=True
)

with st.expander("View Historical Sales Data"):
    historical_display = historical_chart.copy()

    historical_display["Month"] = (
        historical_display["Month"].dt.strftime("%B %Y")
    )

    st.dataframe(
        historical_display.sort_values(
            "Month",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# 9. AUTOMATIC THREE-MONTH FORECAST
# ==================================================
st.divider()
st.subheader("Future Sales Forecast")

st.write(
    f"**Forecast period:** "
    f"{forecast_start.strftime('%B %Y')} to "
    f"{forecast_end.strftime('%B %Y')}"
)

st.caption(
    "The model predicts monthly sales for the next three months. "
    "It does not generate daily sales predictions."
)

if st.button(
    "Generate Sales Forecast",
    type="primary",
    use_container_width=True
):
    try:
        forecast_df = generate_forecast(
            model=model,
            historical_sales=monthly_sales,
            feature_columns=feature_columns,
            horizon=FORECAST_HORIZON
        )

        st.success("Three-month sales forecast generated successfully.")

        # ------------------------------------------
        # FORECAST METRICS
        # ------------------------------------------
        total_forecast = forecast_df["Predicted Sales"].sum()
        average_forecast = forecast_df["Predicted Sales"].mean()

        kpi1, kpi2, kpi3 = st.columns(3)

        kpi1.metric(
            "Total Predicted Sales",
            f"{total_forecast:,.2f}"
        )

        kpi2.metric(
            "Average Monthly Forecast",
            f"{average_forecast:,.2f}"
        )

        kpi3.metric(
            "Forecast Horizon",
            "3 Months"
        )

        # ------------------------------------------
        # FORECAST TABLE
        # ------------------------------------------
        st.subheader("Monthly Forecast Results")

        display_forecast = forecast_df.copy()

        display_forecast["Month"] = (
            display_forecast["Month"].dt.strftime("%B %Y")
        )

        display_forecast["Predicted Sales"] = (
            display_forecast["Predicted Sales"].round(2)
        )

        st.dataframe(
            display_forecast,
            use_container_width=True,
            hide_index=True
        )

        # ------------------------------------------
        # FORECAST CHART
        # ------------------------------------------
        st.subheader("Predicted Monthly Sales")

        st.line_chart(
            forecast_df,
            x="Month",
            y="Predicted Sales",
            use_container_width=True
        )

        # ------------------------------------------
        # HISTORICAL VS FORECAST CHART
        # ------------------------------------------
        st.subheader("Historical vs Forecast Sales")

        historical_part = pd.DataFrame({
            "Month": monthly_sales.index,
            "Historical Sales": monthly_sales.values
        })

        forecast_part = pd.DataFrame({
            "Month": forecast_df["Month"],
            "Forecast Sales": forecast_df["Predicted Sales"]
        })

        comparison_chart = pd.merge(
            historical_part,
            forecast_part,
            on="Month",
            how="outer"
        ).sort_values("Month")

        st.line_chart(
            comparison_chart,
            x="Month",
            y=["Historical Sales", "Forecast Sales"],
            use_container_width=True
        )

        # ------------------------------------------
        # CSV DOWNLOAD
        # ------------------------------------------
        csv_data = forecast_df.copy()

        csv_data["Month"] = (
            csv_data["Month"].dt.strftime("%Y-%m-%d")
        )

        st.download_button(
            label="Download Forecast as CSV",
            data=csv_data.to_csv(index=False).encode("utf-8"),
            file_name="sales_forecast_oct_dec_2014.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.caption(
            "Forecasts are estimates generated by the trained model. "
            "Actual sales may differ from predicted sales."
        )

    except Exception as error:
        st.error(f"Forecast generation failed: {error}")

        st.info(
            "Check that the feature engineering and year_number "
            "calculation match those used during model training."
        )


# ==================================================
# 10. FOOTER
# ==================================================
st.divider()

st.caption(
    "Sales Forecasting Project | Ridge Regression | "
    "Monthly Time-Series Forecasting"
)
