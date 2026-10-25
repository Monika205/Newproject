
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------
st.markdown("""
<style>
    .block-container {
        padding-top: 1.8rem;
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

# --------------------------------------------------
# LOAD TRAINED MODEL BUNDLE
# --------------------------------------------------
MODEL_PATH = Path("sales_forecasting_bundle.joblib")


@st.cache_resource
def load_bundle(path):
    return joblib.load(path)


# --------------------------------------------------
# FORECAST FUNCTION
# --------------------------------------------------
def generate_forecast(model, historical_sales, feature_columns, horizon):
    history = historical_sales.copy().astype(float).sort_index()
    predictions = []

    for _ in range(horizon):
        next_month = history.index.max() + pd.offsets.MonthBegin(1)

        # Features must match the model's training features.
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
            [[features[column] for column in feature_columns]],
            columns=feature_columns
        )

        predicted_sales = float(model.predict(input_data)[0])

        predictions.append({
            "Month": next_month,
            "Predicted Sales": predicted_sales
        })

        # Recursive forecasting: each prediction is used for
        # the following month's lag and rolling features.
        history.loc[next_month] = predicted_sales

    return pd.DataFrame(predictions)


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("📈 Sales Forecasting Dashboard")
st.caption(
    "Monthly sales analysis and future sales forecasting "
    "using a trained Ridge Regression model."
)

st.markdown("""
<div class="info-box">
<b>Forecasting approach:</b> Predictions start from the month
after the final month in the dataset. The app does not allow
manual selection of an unrelated current or future date.
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# CHECK MODEL FILE
# --------------------------------------------------
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
    feature_columns = bundle["feature_columns"]

except Exception as error:
    st.error(f"Could not load the trained model bundle: {error}")
    st.stop()

# --------------------------------------------------
# PREPARE HISTORICAL SALES
# --------------------------------------------------
if isinstance(monthly_sales, pd.DataFrame):
    monthly_sales = monthly_sales.squeeze()

monthly_sales = pd.Series(monthly_sales).copy()
monthly_sales.index = pd.to_datetime(monthly_sales.index)
monthly_sales = monthly_sales.sort_index()
monthly_sales = monthly_sales[~monthly_sales.index.duplicated(keep="last")]
monthly_sales = monthly_sales.astype(float).dropna()

if len(monthly_sales) < 12:
    st.error("At least 12 months of historical sales are required.")
    st.stop()

last_month = monthly_sales.index.max()
forecast_start = last_month + pd.offsets.MonthBegin(1)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("Forecast Settings")

horizon = st.sidebar.selectbox(
    "Select forecast horizon",
    options=[3, 6, 12],
    format_func=lambda value: f"{value} months"
)

st.sidebar.caption(
    "Forecasting begins automatically after the dataset's last month."
)

st.sidebar.divider()
st.sidebar.subheader("Dataset Information")
st.sidebar.write(f"Historical months: **{len(monthly_sales)}**")
st.sidebar.write(
    f"Last recorded month: **{last_month.strftime('%B %Y')}**"
)
st.sidebar.write(
    f"Forecast starts: **{forecast_start.strftime('%B %Y')}**"
)

# --------------------------------------------------
# OVERVIEW METRICS
# --------------------------------------------------
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

# --------------------------------------------------
# HISTORICAL SALES CHART
# --------------------------------------------------
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

with st.expander("View historical sales data"):
    st.dataframe(
        historical_chart.sort_values("Month", ascending=False),
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# FORECAST SECTION
# --------------------------------------------------
st.divider()
st.subheader("Future Sales Forecast")

st.write(
    f"Selected horizon: **{horizon} months** | "
    f"Forecast starts: **{forecast_start.strftime('%d %b %Y')}**"
)

if st.button("Generate Sales Forecast", type="primary", use_container_width=True):
    try:
        forecast_df = generate_forecast(
            model=model,
            historical_sales=monthly_sales,
            feature_columns=feature_columns,
            horizon=horizon
        )

        first_forecast_month = forecast_df["Month"].min()
        last_forecast_month = forecast_df["Month"].max()

        # Inclusive calendar range
        range_start = first_forecast_month
        range_end = last_forecast_month + pd.offsets.MonthEnd(0)
        calendar_days = (range_end - range_start).days + 1

        total_forecast = forecast_df["Predicted Sales"].sum()
        average_forecast = forecast_df["Predicted Sales"].mean()

        st.success("Forecast generated successfully.")

        # Forecast range information
        date_col1, date_col2, date_col3 = st.columns(3)

        date_col1.metric(
            "Forecast Start Date",
            range_start.strftime("%d %b %Y")
        )

        date_col2.metric(
            "Forecast End Date",
            range_end.strftime("%d %b %Y")
        )

        date_col3.metric(
            "Calendar Days Covered",
            f"{calendar_days} days"
        )

        # Forecast KPIs
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
            f"{horizon} months"
        )

        st.caption(
            "Calendar days represent the inclusive date range covered "
            "by the monthly forecasts. They are not daily predictions."
        )

        # Forecast table
        st.subheader("Monthly Forecast Results")

        display_forecast = forecast_df.copy()
        display_forecast["Month"] = display_forecast["Month"].dt.strftime(
            "%B %Y"
        )
        display_forecast["Predicted Sales"] = (
            display_forecast["Predicted Sales"].round(2)
        )

        st.dataframe(
            display_forecast,
            use_container_width=True,
            hide_index=True
        )

        # Forecast chart
        st.subheader("Predicted Monthly Sales")

        st.line_chart(
            forecast_df,
            x="Month",
            y="Predicted Sales",
            use_container_width=True
        )

        # Historical + forecast comparison chart
        st.subheader("Historical vs Forecast Sales")

        historical_part = pd.DataFrame({
            "Month": monthly_sales.index,
            "Sales": monthly_sales.values,
            "Type": "Historical"
        })

        forecast_part = pd.DataFrame({
            "Month": forecast_df["Month"],
            "Sales": forecast_df["Predicted Sales"],
            "Type": "Forecast"
        })

        combined_chart = pd.concat(
            [historical_part, forecast_part],
            ignore_index=True
        )

        combined_chart["Historical Sales"] = combined_chart["Sales"].where(
            combined_chart["Type"] == "Historical"
        )
        combined_chart["Forecast Sales"] = combined_chart["Sales"].where(
            combined_chart["Type"] == "Forecast"
        )

        st.line_chart(
            combined_chart,
            x="Month",
            y=["Historical Sales", "Forecast Sales"],
            use_container_width=True
        )

        # CSV download
        csv_data = forecast_df.copy()
        csv_data["Month"] = csv_data["Month"].dt.strftime("%Y-%m-%d")

        st.download_button(
            label="Download Forecast as CSV",
            data=csv_data.to_csv(index=False).encode("utf-8"),
            file_name=f"sales_forecast_{horizon}_months.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.caption(
            "These estimates are model-based forecasts, not guaranteed "
            "actual sales. Forecast quality depends on the historical "
            "data and the patterns learned during training."
        )

    except Exception as error:
        st.error(f"Forecast generation failed: {error}")
        st.info(
            "Please verify that the feature engineering in this app "
            "matches the feature engineering used when training the model."
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "Sales Forecasting Project | Ridge Regression | "
    "Monthly time-series forecasting"
)
