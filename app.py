
import streamlit as st
import pandas as pd
import joblib
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

# =========================================================
# 2. LOAD TRAINED MODEL BUNDLE
# =========================================================
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "sales_forecasting_bundle.joblib"
)


@st.cache_resource
def load_bundle(model_path):
    return joblib.load(model_path)


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please keep "
        "sales_forecasting_bundle.joblib in the same folder as app.py."
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
# 3. PREPARE HISTORICAL MONTHLY SALES
# =========================================================
if isinstance(monthly_sales, pd.DataFrame):
    if monthly_sales.shape[1] != 1:
        st.error(
            "The monthly_sales object must contain one sales column."
        )
        st.stop()

    monthly_sales = monthly_sales.iloc[:, 0]

monthly_sales = pd.Series(monthly_sales).copy()
monthly_sales.index = pd.to_datetime(monthly_sales.index)
monthly_sales = monthly_sales.sort_index()
monthly_sales = monthly_sales.astype(float).dropna()

# Normalize every date to the first day of its month
monthly_sales.index = (
    monthly_sales.index.to_period("M").to_timestamp()
)

# Combine duplicate monthly entries if any
monthly_sales = (
    monthly_sales.groupby(level=0).sum().sort_index()
)

if len(monthly_sales) < 12:
    st.error(
        "At least 12 historical months are required "
        "for this forecasting model."
    )
    st.stop()

first_historical_month = monthly_sales.index.min()
last_historical_month = monthly_sales.index.max()

first_forecast_month = (
    last_historical_month + pd.offsets.MonthBegin(1)
)


# =========================================================
# 4. FORECAST FUNCTION
# =========================================================
def generate_forecast(
    model,
    historical_sales,
    feature_columns,
    horizon
):
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
        col for col in expected_features
        if col not in feature_columns
    ]

    unexpected_features = [
        col for col in feature_columns
        if col not in expected_features
    ]

    if missing_features or unexpected_features:
        raise ValueError(
            "Model feature columns do not match the app. "
            f"Missing: {missing_features}; "
            f"unexpected: {unexpected_features}."
        )

    for _ in range(horizon):

        next_month = (
            history.index.max() + pd.offsets.MonthBegin(1)
        )

        # These calculations must match model training.
        features = {
            "month": next_month.month,

            "year_number": (
                next_month.year - first_historical_month.year
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

        predictions.append({
            "Month": next_month,
            "Predicted Sales": predicted_sales
        })

        # Add this prediction to history for the next month
        history.loc[next_month] = predicted_sales

    return pd.DataFrame(predictions)


# =========================================================
# 5. DASHBOARD HEADER
# =========================================================
st.title("📈 Sales Forecasting Dashboard")

st.write(
    "Analyze historical monthly sales and generate future "
    "sales forecasts using the trained Ridge Regression model."
)

st.info(
    "The dataset ends in September 2014. Forecasts for later "
    "years are model-generated estimates, not actual sales data."
)


# =========================================================
# 6. SIDEBAR: YEAR, MONTH AND DURATION
# =========================================================
st.sidebar.header("⚙️ Forecast Settings")
st.sidebar.subheader("Select Forecast Start Date")

month_names = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

# Change 2035 if you need a longer selection range
available_years = list(
    range(first_forecast_month.year, 2036)
)

selected_year = st.sidebar.selectbox(
    "Select Year",
    options=available_years,
    index=0,
    key="forecast_year"
)

if selected_year == first_forecast_month.year:
    available_months = list(
        range(first_forecast_month.month, 13)
    )
else:
    available_months = list(range(1, 13))

# Reset month safely when the selected year changes
if (
    "forecast_month" not in st.session_state
    or st.session_state["forecast_month"] not in available_months
):
    st.session_state["forecast_month"] = available_months[0]

selected_month = st.sidebar.selectbox(
    "Select Month",
    options=available_months,
    format_func=lambda month: month_names[month],
    key="forecast_month"
)

selected_forecast_start = pd.Timestamp(
    year=selected_year,
    month=selected_month,
    day=1
)

horizon = st.sidebar.selectbox(
    "Forecast Duration",
    options=[3, 6, 12],
    index=0,
    format_func=lambda value: f"{value} months"
)

st.sidebar.divider()
st.sidebar.subheader("Dataset Information")

st.sidebar.write(
    f"First historical month: "
    f"**{first_historical_month.strftime('%B %Y')}**"
)

st.sidebar.write(
    f"Last historical month: "
    f"**{last_historical_month.strftime('%B %Y')}**"
)

st.sidebar.write(
    f"Historical months available: **{len(monthly_sales)}**"
)

st.sidebar.write(
    f"Selected forecast start: "
    f"**{selected_forecast_start.strftime('%B %Y')}**"
)


# =========================================================
# 7. HISTORICAL SALES OVERVIEW
# =========================================================
st.divider()
st.subheader("📊 Historical Sales Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Historical Sales",
    f"{monthly_sales.sum():,.2f}"
)

col2.metric(
    "Average Monthly Sales",
    f"{monthly_sales.mean():,.2f}"
)

col3.metric(
    "Historical Months",
    f"{len(monthly_sales)}"
)


# =========================================================
# 8. HISTORICAL SALES CHART
# =========================================================
st.subheader("Historical Monthly Sales")

historical_chart = pd.DataFrame({
    "Month": monthly_sales.index,
    "Historical Sales": monthly_sales.values
})

st.line_chart(
    historical_chart,
    x="Month",
    y="Historical Sales",
    use_container_width=True
)

with st.expander("View Historical Sales Data"):

    historical_table = historical_chart.copy()

    historical_table["Month"] = (
        historical_table["Month"].dt.strftime("%B %Y")
    )

    historical_table["Historical Sales"] = (
        historical_table["Historical Sales"].round(2)
    )

    st.dataframe(
        historical_table,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 9. FORECAST SECTION
# =========================================================
st.divider()
st.subheader("🔮 Sales Forecast")

st.write(
    f"Selected start month: "
    f"**{selected_forecast_start.strftime('%B %Y')}**"
)

st.write(f"Forecast duration: **{horizon} months**")

if st.button(
    "Generate Sales Forecast",
    type="primary",
    use_container_width=True
):

    try:
        # Calculate the distance from the final historical month
        gap_months = (
            (selected_forecast_start.year - last_historical_month.year)
            * 12
            + selected_forecast_start.month
            - last_historical_month.month
        )

        if gap_months < 1:
            st.error(
                "Please select a month after the last historical month."
            )
            st.stop()

        # Generate through the selected start and requested horizon
        all_forecasts = generate_forecast(
            model=model,
            historical_sales=monthly_sales,
            feature_columns=feature_columns,
            horizon=gap_months + horizon
        )

        # Keep only the requested forecast period
        forecast_df = all_forecasts[
            all_forecasts["Month"] >= selected_forecast_start
        ].head(horizon).copy()

        forecast_df = forecast_df.reset_index(drop=True)

        if len(forecast_df) != horizon:
            st.error(
                "The complete forecast period could not be generated."
            )
            st.stop()

        # =============================================
        # 10. FORECAST DATES
        # =============================================
        range_start = forecast_df["Month"].min()

        range_end = (
            forecast_df["Month"].max()
            + pd.offsets.MonthEnd(0)
        )

        calendar_days = (range_end - range_start).days + 1

        # =============================================
        # 11. FORECAST KPIs
        # =============================================
        st.success("Forecast generated successfully!")

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
            "Calendar days represent the inclusive date range "
            "covered by the monthly forecast. They are not daily "
            "sales predictions."
        )

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Total Predicted Sales",
            f"{forecast_df['Predicted Sales'].sum():,.2f}"
        )

        metric2.metric(
            "Average Monthly Forecast",
            f"{forecast_df['Predicted Sales'].mean():,.2f}"
        )

        metric3.metric(
            "Forecast Duration",
            f"{horizon} months"
        )

        # =============================================
        # 12. FORECAST TABLE
        # =============================================
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

        # =============================================
        # 13. FORECAST CHART
        # =============================================
        st.subheader("📈 Predicted Monthly Sales")

        forecast_chart = forecast_df.copy()

        forecast_chart["Month"] = (
            forecast_chart["Month"].dt.strftime("%b %Y")
        )

        st.line_chart(
            forecast_chart,
            x="Month",
            y="Predicted Sales",
            use_container_width=True
        )

        # =============================================
        # 14. HISTORICAL VS FORECAST CHART
        # =============================================
        st.subheader("Historical vs Forecast Sales")

        # Show only the latest 12 historical months
        recent_history = monthly_sales.tail(12)

        historical_part = pd.DataFrame({
            "Month": recent_history.index,
            "Historical Sales": recent_history.values
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
        ).sort_values("Month").reset_index(drop=True)

        # Format labels only after sorting real dates
        comparison_chart["Month"] = (
            comparison_chart["Month"].dt.strftime("%b %Y")
        )

        st.caption(
            "Showing the last 12 historical months and the "
            "selected forecast period."
        )

        st.line_chart(
            comparison_chart,
            x="Month",
            y=["Historical Sales", "Forecast Sales"],
            use_container_width=True
        )

        # =============================================
        # 15. CSV DOWNLOAD
        # =============================================
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
            "Forecast values are estimates and are not guaranteed "
            "to match actual future sales."
        )

    except Exception as error:
        st.error(f"Forecast generation failed: {error}")

        st.info(
            "Please verify that the saved model's feature calculations "
            "match those used during model training."
        )


# =========================================================
# 16. FOOTER
# =========================================================
st.divider()

st.caption(
    "Sales Forecasting Dashboard | Ridge Regression | "
    "Monthly Time-Series Forecasting"
)
