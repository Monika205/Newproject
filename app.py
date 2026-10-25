
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# FILE PATH
# =====================================================
MODEL_PATH = Path(__file__).parent / "sales_forecasting_bundle.joblib"

# =====================================================
# LOAD TRAINED MODEL BUNDLE
# =====================================================
@st.cache_resource
def load_model_bundle(path):
    return joblib.load(path)


# =====================================================
# FORECASTING FUNCTION
# =====================================================
def predict_next_month(history, forecast_date, model, feature_columns, first_year):
    """
    Predict one month using the trained model and previously
    observed/predicted monthly sales.
    """

    history = history.sort_index()

    features = {
        "month": forecast_date.month,
        "year_number": forecast_date.year - first_year,
        "lag_1": history.iloc[-1],
        "lag_2": history.iloc[-2] if len(history) >= 2 else np.nan,
        "lag_3": history.iloc[-3] if len(history) >= 3 else np.nan,
        "lag_12": history.iloc[-12] if len(history) >= 12 else np.nan,
        "rolling_mean_3": history.tail(3).mean(),
        "rolling_mean_12": history.tail(12).mean()
    }

    # Use the same feature order as during model training
    input_df = pd.DataFrame([features])
    input_df = input_df[feature_columns]

    prediction = float(model.predict(input_df)[0])

    # Sales cannot be negative
    return max(0.0, prediction)


# =====================================================
# GENERATE FORECAST FOR A SELECTED START MONTH
# =====================================================
def generate_forecast(
    monthly_sales,
    model,
    feature_columns,
    selected_start,
    horizon,
    first_year
):
    """
    Forecast sequentially from the last actual month.
    If the selected start month is later, intermediate
    predictions are generated first so lag features remain valid.
    """

    last_actual_month = monthly_sales.index.max().to_period("M")
    selected_start_period = selected_start.to_period("M")

    history = monthly_sales.copy().sort_index()
    forecast_records = []

    next_period = last_actual_month + 1
    final_period = selected_start_period + horizon - 1

    while next_period <= final_period:
        forecast_date = next_period.to_timestamp()

        prediction = predict_next_month(
            history=history,
            forecast_date=forecast_date,
            model=model,
            feature_columns=feature_columns,
            first_year=first_year
        )

        history.loc[forecast_date] = prediction

        # Display only the requested forecast period
        if next_period >= selected_start_period:
            forecast_records.append({
                "Forecast Month": forecast_date,
                "Predicted Sales": round(prediction, 2)
            })

        next_period += 1

    forecast_df = pd.DataFrame(forecast_records)
    return forecast_df


# =====================================================
# MAIN APPLICATION
# =====================================================
st.title("📊 Sales Forecasting Dashboard")

st.write(
    "Forecast future monthly sales using the trained "
    "Ridge Regression model."
)

st.divider()

# =====================================================
# CHECK AND LOAD MODEL
# =====================================================
if not MODEL_PATH.exists():
    st.error(
        "Model file not found: sales_forecasting_bundle.joblib. "
        "Place the model file in the same folder as app.py."
    )
    st.stop()

try:
    bundle = load_model_bundle(str(MODEL_PATH))

    model = bundle["model"]
    monthly_sales = bundle["monthly_sales"].copy()
    feature_columns = bundle["feature_columns"]

except Exception as error:
    st.error(f"Could not load the model bundle: {error}")
    st.stop()


# =====================================================
# PREPARE HISTORICAL DATA
# =====================================================
monthly_sales.index = pd.to_datetime(monthly_sales.index)
monthly_sales = monthly_sales.sort_index()
monthly_sales = monthly_sales[~monthly_sales.index.duplicated(keep="last")]
monthly_sales = pd.to_numeric(monthly_sales, errors="coerce").dropna()

if len(monthly_sales) < 12:
    st.error(
        "At least 12 months of monthly sales history are required "
        "for this forecasting setup."
    )
    st.stop()

last_actual_month = monthly_sales.index.max().to_period("M")
first_year = monthly_sales.index.min().year

# Allow selecting one of the next six forecast start months.
# This avoids choosing an arbitrary current-year date far beyond
# the available historical data.
available_start_dates = pd.date_range(
    start=(last_actual_month + 1).to_timestamp(),
    periods=6,
    freq="MS"
)

available_years = sorted(
    {date.year for date in available_start_dates}
)

# =====================================================
# HISTORICAL SUMMARY
# =====================================================
st.subheader("Historical Sales Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "First Historical Month",
        monthly_sales.index.min().strftime("%B %Y")
    )

with col2:
    st.metric(
        "Last Historical Month",
        last_actual_month.to_timestamp().strftime("%B %Y")
    )

with col3:
    st.metric(
        "Historical Months",
        len(monthly_sales)
    )

# =====================================================
# HISTORICAL SALES CHART
# =====================================================
st.subheader("Historical Monthly Sales")

historical_chart = monthly_sales.rename("Actual Sales").to_frame()
st.line_chart(historical_chart, y="Actual Sales", use_container_width=True)

st.divider()

# =====================================================
# FORECAST CONTROLS
# =====================================================
st.subheader("Configure Your Forecast")

col1, col2, col3 = st.columns(3)

with col1:
    selected_year = st.selectbox(
        "Forecast Start Year",
        options=available_years,
        index=0
    )

months_for_selected_year = [
    date for date in available_start_dates
    if date.year == selected_year
]

with col2:
    selected_month_label = st.selectbox(
        "Forecast Start Month",
        options=[date.strftime("%B") for date in months_for_selected_year]
    )

selected_start = next(
    date for date in months_for_selected_year
    if date.strftime("%B") == selected_month_label
)

with col3:
    horizon = st.selectbox(
        "Forecast Horizon",
        options=[1, 2, 3, 4, 5, 6],
        index=2,
        format_func=lambda x: f"{x} Month{'s' if x != 1 else ''}"
    )

forecast_end = (
    selected_start.to_period("M") + horizon - 1
).to_timestamp()

st.caption(
    f"Selected forecast period: "
    f"{selected_start.strftime('%B %Y')} to "
    f"{forecast_end.strftime('%B %Y')}."
)

st.info(
    "The dataset ends in September 2014. Forecasts start after the "
    "last historical month. If you choose a later start month, the app "
    "generates intermediate monthly predictions before showing your "
    "selected period. Predictions farther from the historical data "
    "are increasingly uncertain."
)

# =====================================================
# GENERATE FORECAST
# =====================================================
if st.button("Generate Sales Forecast", type="primary", use_container_width=True):

    try:
        with st.spinner("Generating monthly sales forecast..."):
            forecast_df = generate_forecast(
                monthly_sales=monthly_sales,
                model=model,
                feature_columns=feature_columns,
                selected_start=selected_start,
                horizon=horizon,
                first_year=first_year
            )

        if forecast_df.empty:
            st.warning("No forecast results were generated.")
            st.stop()

        # =============================================
        # FORECAST SUMMARY
        # =============================================
        st.success("Forecast generated successfully!")

        total_forecast = forecast_df["Predicted Sales"].sum()
        average_forecast = forecast_df["Predicted Sales"].mean()

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric(
                "Total Forecast Sales",
                f"{total_forecast:,.2f}"
            )

        with metric2:
            st.metric(
                "Average Monthly Forecast",
                f"{average_forecast:,.2f}"
            )

        with metric3:
            st.metric(
                "Forecast Months",
                len(forecast_df)
            )

        # =============================================
        # FORECAST TABLE
        # =============================================
        st.subheader("Forecast Results")

        display_df = forecast_df.copy()
        display_df["Forecast Month"] = (
            display_df["Forecast Month"].dt.strftime("%B %Y")
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # =============================================
        # FORECAST CHART
        # =============================================
        st.subheader("Predicted Monthly Sales")

        forecast_chart = forecast_df.set_index("Forecast Month")[
            ["Predicted Sales"]
        ]

        st.line_chart(
            forecast_chart,
            y="Predicted Sales",
            use_container_width=True
        )

        # =============================================
        # HISTORICAL VS FORECAST CHART
        # =============================================
        st.subheader("Historical Sales vs Forecast")

        historical_part = monthly_sales.rename("Actual Sales").to_frame()

        forecast_part = forecast_df.set_index("Forecast Month")[
            ["Predicted Sales"]
        ]

        comparison_df = historical_part.join(
            forecast_part,
            how="outer"
        )

        st.line_chart(
            comparison_df,
            use_container_width=True
        )

        # =============================================
        # DOWNLOAD FORECAST CSV
        # =============================================
        csv_data = forecast_df.copy()
        csv_data["Forecast Month"] = (
            csv_data["Forecast Month"].dt.strftime("%Y-%m-%d")
        )

        csv_bytes = csv_data.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Forecast as CSV",
            data=csv_bytes,
            file_name="sales_forecast.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.caption(
            "Note: Forecast values are model estimates, not actual sales. "
            "This app produces monthly predictions, not daily predictions."
        )

    except Exception as error:
        st.error(f"An error occurred while generating the forecast: {error}")

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption("Sales Forecasting Project | Ridge Regression")
