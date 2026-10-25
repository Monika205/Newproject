
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

monthly_sales.index = (
    monthly_sales.index.to_period("M").to_timestamp()
)

monthly_sales = (
    monthly_sales.groupby(level=0).sum().sort_index()
)

if len(monthly_sales) < 12:
    st.error("At least 12 historical months are required.")
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

    missing = [
        column for column in expected_features
        if column not in feature_columns
    ]

    unexpected = [
        column for column in feature_columns
        if column not in expected_features
    ]

    if missing or unexpected:
        raise ValueError(
            f"Model feature mismatch. Missing: {missing}; "
            f"unexpected: {unexpected}"
        )

    for _ in range(horizon):
        next_month = (
            history.index.max() + pd.offsets.MonthBegin(1)
        )

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
            [[features[column] for column in feature_columns]],
            columns=feature_columns
        )

        predicted_sales = float(model.predict(input_data)[0])

        predictions.append({
            "Month": next_month,
            "Predicted Sales": predicted_sales
        })

        # Use predicted sales for the next month's lag features.
        history.loc[next_month] = predicted_sales

    return pd.DataFrame(predictions)


# =========================================================
# 5. PLOTLY CHART HELPER
# =========================================================
def create_sales_chart(
    historical_data,
    forecast_data=None,
    title="Monthly Sales",
    height=450
):
    fig = go.Figure()

    if historical_data is not None and not historical_data.empty:
        hist = historical_data.sort_index()

        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist.values,
                mode="lines+markers",
                name="Historical Sales",
                line=dict(color="#1687E8", width=2.5),
                marker=dict(size=5),
                hovertemplate=(
                    "Month: %{x|%b %Y}<br>"
                    "Historical Sales: %{y:,.2f}"
                    "<extra></extra>"
                )
            )
        )

    if forecast_data is not None and not forecast_data.empty:
        forecast = forecast_data.sort_values("Month")

        fig.add_trace(
            go.Scatter(
                x=forecast["Month"],
                y=forecast["Predicted Sales"],
                mode="lines+markers",
                name="Forecast Sales",
                line=dict(color="#FF9F43", width=3, dash="dash"),
                marker=dict(size=9, symbol="diamond"),
                hovertemplate=(
                    "Month: %{x|%b %Y}<br>"
                    "Forecast Sales: %{y:,.2f}"
                    "<extra></extra>"
                )
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=65, b=30),
        xaxis=dict(
            title="Month",
            type="date",
            tickformat="%b %Y",
            dtick="M1",
            tickangle=-45,
            showgrid=False,
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            title="Sales",
            tickformat=",.0f",
            rangemode="tozero",
            gridcolor="rgba(150,150,150,0.2)"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    return fig


# =========================================================
# 6. DASHBOARD HEADER
# =========================================================
st.title("📈 Sales Forecasting Dashboard")

st.write(
    "Analyze historical monthly sales and generate future "
    "sales forecasts using Ridge Regression."
)

st.info(
    f"Historical data available from "
    f"{first_historical_month.strftime('%B %Y')} to "
    f"{last_historical_month.strftime('%B %Y')}. "
    "Forecasts are estimates, not actual future sales."
)


# =========================================================
# 7. SIDEBAR SETTINGS
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

# Year selector: 2014, 2015 and 2016
MIN_FORECAST_YEAR = first_forecast_month.year
MAX_FORECAST_YEAR = 2017

if MIN_FORECAST_YEAR > MAX_FORECAST_YEAR:
    st.error(
        "The historical data ends after the configured forecast "
        "year range. Please update MAX_FORECAST_YEAR."
    )
    st.stop()

available_years = list(
    range(MIN_FORECAST_YEAR, MAX_FORECAST_YEAR + 1)
)

selected_year = st.sidebar.selectbox(
    "Select Year",
    options=available_years,
    index=0,
    key="forecast_year"
)

# For the first forecast year, only months after the last
# historical month are available.
if selected_year == first_forecast_month.year:
    available_months = list(
        range(first_forecast_month.month, 13)
    )
else:
    available_months = list(range(1, 13))

# Keep the selected month valid when the year changes.
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

st.sidebar.caption(
    "Forecasts are generated from the last historical month. "
    "Predictions for distant start dates are less reliable."
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
# 8. HISTORICAL SALES OVERVIEW
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
# 9. HISTORICAL SALES CHART
# =========================================================
st.subheader("Historical Monthly Sales")

historical_fig = create_sales_chart(
    historical_data=monthly_sales,
    title="Historical Monthly Sales",
    height=450
)

st.plotly_chart(
    historical_fig,
    use_container_width=True,
    config={"displaylogo": False}
)

with st.expander("View Historical Sales Data"):
    historical_table = pd.DataFrame({
        "Month": monthly_sales.index.strftime("%B %Y"),
        "Historical Sales": monthly_sales.values.round(2)
    })

    st.dataframe(
        historical_table,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 10. FORECAST SECTION
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

        # Generate predictions from the end of historical data
        # through the end of the selected forecast period.
        forecast_steps = gap_months + horizon - 1

        all_forecasts = generate_forecast(
            model=model,
            historical_sales=monthly_sales,
            feature_columns=feature_columns,
            horizon=forecast_steps
        )

        # Display only the selected start month and duration.
        forecast_df = all_forecasts[
            all_forecasts["Month"] >= selected_forecast_start
        ].head(horizon).copy().reset_index(drop=True)

        if len(forecast_df) != horizon:
            st.error(
                "The complete forecast period could not be generated."
            )
            st.stop()

        # ---------------------------------------------
        # 11. FORECAST DATE RANGE
        # ---------------------------------------------
        range_start = forecast_df["Month"].min()

        range_end = (
            forecast_df["Month"].max()
            + pd.offsets.MonthEnd(0)
        )

        calendar_days = (range_end - range_start).days + 1

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
            "Calendar days represent the inclusive period covered "
            "by monthly forecasts. They are not daily predictions."
        )

        # ---------------------------------------------
        # 12. FORECAST KPIs
        # ---------------------------------------------
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

        # ---------------------------------------------
        # 13. FORECAST TABLE
        # ---------------------------------------------
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

        # ---------------------------------------------
        # 14. FORECAST CHART
        # ---------------------------------------------
        st.subheader("📈 Predicted Monthly Sales")

        forecast_fig = create_sales_chart(
            historical_data=None,
            forecast_data=forecast_df,
            title="Predicted Monthly Sales",
            height=400
        )

        st.plotly_chart(
            forecast_fig,
            use_container_width=True,
            config={"displaylogo": False}
        )

        # ---------------------------------------------
        # 15. HISTORICAL VS FORECAST
        # ---------------------------------------------
        st.subheader("Historical vs Forecast Sales")

        # Last 12 actual historical months before the
        # selected forecast start date.
        recent_history = monthly_sales.tail(12).copy()

        comparison_history = recent_history[
            recent_history.index < selected_forecast_start
        ]

        comparison_fig = create_sales_chart(
            historical_data=comparison_history,
            forecast_data=forecast_df,
            title="Historical vs Forecast Sales",
            height=500
        )

        chart_start = comparison_history.index.min()

        if pd.isna(chart_start):
            chart_start = selected_forecast_start

        chart_end = (
            forecast_df["Month"].max()
            + pd.offsets.MonthEnd(0)
        )

        comparison_fig.update_xaxes(
            range=[chart_start, chart_end]
        )

        st.caption(
            "Historical sales and forecast sales are shown separately "
            "in chronological order. If the selected forecast starts "
            "far in the future, a time gap will appear between them."
        )

        st.plotly_chart(
            comparison_fig,
            use_container_width=True,
            config={"displaylogo": False}
        )

        # ---------------------------------------------
        # 16. DOWNLOAD FORECAST CSV
        # ---------------------------------------------
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
            "Forecast values are estimates and may differ from actual sales."
        )

    except Exception as error:
        st.error(f"Forecast generation failed: {error}")
        st.info(
            "Check that the model feature calculations match "
            "those used during training."
        )


# =========================================================
# 17. FOOTER
# =========================================================
st.divider()

st.caption(
    "Sales Forecasting Dashboard | Ridge Regression | "
    "Monthly Time-Series Forecasting"
)
