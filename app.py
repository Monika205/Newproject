
import streamlit as st
import pandas as pd
import numpy as np
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

# ==================================================
# 2. CUSTOM CSS
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
MODEL_PATH = Path("sales_forecasting_bundle.joblib")


@st.cache_resource
def load_bundle(model_path):
    return joblib.load(model_path)


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
    st.error(f"Unable to load the model bundle: {error}")
    st.stop()

# ==================================================
# 4. PREPARE HISTORICAL MONTHLY SALES
# ==================================================
if isinstance(monthly_sales, pd.DataFrame):
    monthly_sales = monthly_sales.squeeze()

monthly_sales = pd.Series(monthly_sales).copy()
monthly_sales.index = pd.to_datetime(monthly_sales.index)
monthly_sales = monthly_sales.sort_index()
monthly_sales = monthly_sales[
    ~monthly_sales.index.duplicated(keep="last")
]
monthly_sales = monthly_sales.astype(float).dropna()

if len(monthly_sales) < 12:
    st.error(
        "At least 12 months of historical sales are required "
        "for the current forecasting features."
    )
    st.stop()

last_month = monthly_sales.index.max()
forecast_start = last_month + pd.offsets.MonthBegin(1)

# ==================================================
# 5. FORECAST FUNCTION
# ==================================================
def generate_forecast(model, historical_sales, feature_columns, horizon):

    history = historical_sales.copy().astype(float).sort_index()
    predictions = []

    for _ in range(horizon):

        # Forecast starts from the month after the latest data
        next_month = history.index.max() + pd.offsets.MonthBegin(1)

        # Create the same features used during model training.
        # IMPORTANT: These definitions must match your notebook.
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

        predicted_sales = float(model.predict(input_data)[0])

        predictions.append({
            "Month": next_month,
            "Predicted Sales": predicted_sales
        })

        # Use the prediction for the following month's features
        history.loc[next_month] = predicted_sales

    return pd.DataFrame(predictions)


# ==================================================
# 6. DASHBOARD HEADER
# ==================================================
st.title("📈 Sales Forecasting Dashboard")

st.write(
    "Analyze historical sales and generate future monthly sales "
    "forecasts using the trained Ridge Regression model."
)

st.markdown("""
<div class="info-box">
<b>Forecasting information:</b> The forecast automatically begins
after the last month available in the dataset. You cannot select
an unrelated current or future date.
</div>
""", unsafe_allow_html=True)

# ==================================================
# 7. SIDEBAR SETTINGS
# ==================================================
st.sidebar.title("⚙️ Forecast Settings")

horizon = st.sidebar.selectbox(
    "Select forecast duration",
    options=list(range(1, 7)),
    format_func=lambda value: (
        f"{value} month" if value == 1 else f"{value} months"
    )
)

st.sidebar.divider()
st.sidebar.subheader("Dataset Information")

st.sidebar.write(f"Historical months: **{len(monthly_sales)}**")
st.sidebar.write(
    f"Last available month: **{last_month.strftime('%B %Y')}**"
)
st.sidebar.write(
    f"Forecast starts: **{forecast_start.strftime('%B %Y')}**"
)

st.sidebar.caption(
    "Choose any forecast duration from 1 to 6 months."
)

# ==================================================
# 8. HISTORICAL SALES SUMMARY
# ==================================================
total_historical_sales = monthly_sales.sum()
average_monthly_sales = monthly_sales.mean()

st.subheader("Historical Sales Overview")

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
# 9. HISTORICAL SALES CHART
# ==================================================
st.divider()
st.subheader("📊 Historical Monthly Sales")

historical_chart = pd.DataFrame({
    "Month": monthly_sales.index,
    "Sales": monthly_sales.values
})

st.line_chart(
    historical_chart,
    x="Month",
    y="Sales",
    use_container_width=True
)

with st.expander("View historical sales records"):
    historical_table = historical_chart.copy()
    historical_table["Month"] = historical_table["Month"].dt.strftime(
        "%B %Y"
    )

    st.dataframe(
        historical_table.sort_values("Month", ascending=False),
        use_container_width=True,
        hide_index=True
    )

# ==================================================
# 10. GENERATE FORECAST
# ==================================================
st.divider()
st.subheader("🔮 Sales Forecast")

st.write(
    f"Selected duration: **{horizon} month(s)**"
)

st.write(
    f"Forecast period begins: **{forecast_start.strftime('%d %B %Y')}**"
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
            horizon=horizon
        )

        # --------------------------------------------------
        # 11. FORECAST DATE RANGE
        # --------------------------------------------------
        range_start = forecast_df["Month"].min()

        range_end = (
            forecast_df["Month"].max()
            + pd.offsets.MonthEnd(0)
        )

        # Count calendar days inclusively
        calendar_days = (range_end - range_start).days + 1

        # --------------------------------------------------
        # 12. FORECAST SUMMARY
        # --------------------------------------------------
        total_forecast = forecast_df["Predicted Sales"].sum()
        average_forecast = forecast_df["Predicted Sales"].mean()

        st.success("Sales forecast generated successfully!")

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

        st.divider()

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
            "Forecast Duration",
            f"{horizon} month(s)"
        )

        st.caption(
            "Calendar days show the complete date range covered "
            "by the monthly forecasts. The model predicts monthly "
            "sales, not individual daily sales."
        )

        # --------------------------------------------------
        # 13. FORECAST RESULTS TABLE
        # --------------------------------------------------
        st.divider()
        st.subheader("📋 Monthly Forecast Results")

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

        # --------------------------------------------------
        # 14. FORECAST CHART
        # --------------------------------------------------
        st.subheader("📈 Predicted Monthly Sales")

        st.line_chart(
            forecast_df,
            x="Month",
            y="Predicted Sales",
            use_container_width=True
        )

        # --------------------------------------------------
        # 15. HISTORICAL VS FORECAST CHART
        # --------------------------------------------------
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

        # --------------------------------------------------
        # 16. DOWNLOAD FORECAST AS CSV
        # --------------------------------------------------
        st.divider()
        st.subheader("Download Forecast")

        download_df = forecast_df.copy()
        download_df["Month"] = (
            download_df["Month"].dt.strftime("%Y-%m-%d")
        )
        download_df["Predicted Sales"] = (
            download_df["Predicted Sales"].round(2)
        )

        csv_data = download_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Forecast as CSV",
            data=csv_data,
            file_name=f"sales_forecast_{horizon}_months.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.caption(
            "Forecasts are estimates, not guaranteed actual sales. "
            "Uncertainty generally increases with longer forecast horizons."
        )

    except Exception as error:
        st.error(f"Forecast generation failed: {error}")

        st.info(
            "Check that the feature names and calculations in app.py "
            "exactly match those used when training the saved model."
        )

# ==================================================
# 17. FOOTER
# ==================================================
st.divider()

st.caption(
    "Sales Forecasting Project | Ridge Regression | "
    "Monthly Time-Series Forecasting"
)
