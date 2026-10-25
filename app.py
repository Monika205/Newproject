
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

# ==================================================
# 2. CUSTOM STYLING
# ==================================================
st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)

# ==================================================
# 3. LOAD TRAINED MODEL BUNDLE
# ==================================================
MODEL_PATH = Path("sales_forecasting_bundle.joblib")


@st.cache_resource
def load_bundle(path):
    return joblib.load(path)


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
            "The saved monthly_sales object has multiple columns. "
            "Expected one monthly sales series."
        )
        st.stop()

    monthly_sales = monthly_sales.iloc[:, 0]

monthly_sales = pd.Series(monthly_sales).copy()

monthly_sales.index = pd.to_datetime(monthly_sales.index)
monthly_sales = monthly_sales.sort_index()
monthly_sales = monthly_sales.astype(float).dropna()

# Standardize the index to the first day of each month
monthly_sales.index = (
    monthly_sales.index.to_period("M").to_timestamp()
)

# Combine values if multiple records refer to the same month
monthly_sales = (
    monthly_sales.groupby(level=0).sum().sort_index()
)

if len(monthly_sales) < 12:
    st.error(
        "At least 12 months of historical sales are required "
        "for the lag_12 and rolling_mean_12 features."
    )
    st.stop()

last_historical_month = monthly_sales.index.max()

forecast_start = (
    last_historical_month + pd.offsets.MonthBegin(1)
)

first_historical_year = monthly_sales.index.min().year

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
    Generate recursive monthly sales forecasts.

    Feature calculations must match the original training notebook.
    """

    history = historical_sales.copy().astype(float).sort_index()
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
        column for column in expected_features
        if column not in feature_columns
    ]

    extra_features = [
        column for column in feature_columns
        if column not in expected_features
    ]

    if missing_features or extra_features:
        raise ValueError(
            "The model's saved feature_columns do not match "
            "the features implemented in this app. "
            f"Missing: {missing_features}; "
            f"unexpected: {extra_features}."
        )

    for _ in range(horizon):

        # Start from the month after the latest available month
        next_month = (
            history.index.max() + pd.offsets.MonthBegin(1)
        )

        # Feature engineering
        # year_number = year - first historical year
        features = {
            "month": next_month.month,

            "year_number": (
                next_month.year
                - historical_sales.index.min().year
            ),

            "lag_1": history.iloc[-1],
            "lag_2": history.iloc[-2],
            "lag_3": history.iloc[-3],
            "lag_12": history.iloc[-12],

            "rolling_mean_3": history.iloc[-3:].mean(),
            "rolling_mean_12": history.iloc[-12:].mean()
        }

        # Use the exact feature order saved with the model
        input_data = pd.DataFrame(
            [[features[column] for column in feature_columns]],
            columns=feature_columns
        )

        predicted_sales = float(
            model.predict(input_data)[0]
        )

        predictions.append({
            "Month": next_month,
            "Predicted Sales": predicted_sales
        })

        # Use each prediction when calculating future lag features
        history.loc[next_month] = predicted_sales

    return pd.DataFrame(predictions)


# ==================================================
# 6. DASHBOARD HEADER
# ==================================================
st.title("📈 Sales Forecasting Dashboard")

st.write(
    "Analyze historical sales and forecast future monthly sales "
    "using a trained Ridge Regression model."
)

st.markdown(
    """
    <div class="info-box">
        <b>Forecasting approach:</b> Forecasts begin after the last
        month available in the historical dataset. Since the dataset
        is not up to date, this application does not claim to predict
        current 2026 sales.
    </div>
    """,
    unsafe_allow_html=True
)

# ==================================================
# 7. SIDEBAR SETTINGS
# ==================================================
st.sidebar.header("⚙️ Forecast Settings")

st.sidebar.subheader("Forecast Start Period")

st.sidebar.text_input(
    "Start month and year",
    value=forecast_start.strftime("%B %Y"),
    disabled=True,
    help="Automatically calculated from the last historical month."
)

st.sidebar.caption(
    f"Last available month: "
    f"{last_historical_month.strftime('%B %Y')}"
)

horizon = st.sidebar.selectbox(
    "Select forecast duration",
    options=[3, 6, 12],
    index=0,
    format_func=lambda value: f"{value} months"
)

st.sidebar.divider()
st.sidebar.subheader("Dataset Information")

st.sidebar.write(
    f"Historical months: **{len(monthly_sales)}**"
)

st.sidebar.write(
    f"Forecast begins: **{forecast_start.strftime('%B %Y')}**"
)

# ==================================================
# 8. HISTORICAL SALES OVERVIEW
# ==================================================
st.divider()
st.subheader("Historical Sales Overview")

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
# 9. HISTORICAL SALES CHART
# ==================================================
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

    historical_table["Month"] = (
        historical_table["Month"].dt.strftime("%B %Y")
    )

    historical_table["Sales"] = (
        historical_table["Sales"].round(2)
    )

    st.dataframe(
        historical_table.sort_values(
            "Month",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

# ==================================================
# 10. FORECAST SECTION
# ==================================================
st.divider()
st.subheader("🔮 Future Sales Forecast")

st.write(
    f"Selected duration: **{horizon} months**"
)

st.write(
    f"Forecast starts from: "
    f"**{forecast_start.strftime('%d %B %Y')}**"
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

        # ==========================================
        # 11. FORECAST DATE RANGE
        # ==========================================
        range_start = forecast_df["Month"].min()

        range_end = (
            forecast_df["Month"].max()
            + pd.offsets.MonthEnd(0)
        )

        # Count both the first and last calendar days
        calendar_days = (
            range_end - range_start
        ).days + 1

        # ==========================================
        # 12. FORECAST METRICS
        # ==========================================
        total_forecast = (
            forecast_df["Predicted Sales"].sum()
        )

        average_forecast = (
            forecast_df["Predicted Sales"].mean()
        )

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

        st.caption(
            "Calendar days show the complete inclusive date range "
            "covered by the monthly forecasts. These are not daily "
            "sales predictions."
        )

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
            f"{horizon} months"
        )

        # ==========================================
        # 13. FORECAST TABLE
        # ==========================================
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

        # ==========================================
        # 14. FORECAST CHART
        # ==========================================
        st.subheader("📈 Predicted Monthly Sales")

        st.line_chart(
            forecast_df,
            x="Month",
            y="Predicted Sales",
            use_container_width=True
        )

        # ==========================================
        # 15. HISTORICAL VS FORECAST CHART
        # ==========================================
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

        # ==========================================
        # 16. CSV DOWNLOAD
        # ==========================================
        st.divider()
        st.subheader("Download Forecast")

        download_df = forecast_df.copy()

        download_df["Month"] = (
            download_df["Month"].dt.strftime("%Y-%m-%d")
        )

        download_df["Predicted Sales"] = (
            download_df["Predicted Sales"].round(2)
        )

        csv_data = (
            download_df.to_csv(index=False).encode("utf-8")
        )

        st.download_button(
            label="⬇️ Download Forecast as CSV",
            data=csv_data,
            file_name=f"sales_forecast_{horizon}_months.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.caption(
            "Forecast values are estimates, not guaranteed actual "
            "sales. Forecast uncertainty can increase with longer "
            "forecast horizons."
        )

    except Exception as error:

        st.error(
            f"Forecast generation failed: {error}"
        )

        st.info(
            "Check whether the model's saved feature names and "
            "feature calculations match the original training notebook."
        )

# ==================================================
# 17. FOOTER
# ==================================================
st.divider()

st.caption(
    "Sales Forecasting Project | Ridge Regression | "
    "Monthly Time-Series Forecasting"
)
