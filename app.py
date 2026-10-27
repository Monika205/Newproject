
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "https://newproject-gvg1.onrender.com"
HISTORY_API_URL = f"{API_BASE_URL}/history"
DAILY_FORECAST_API_URL = f"{API_BASE_URL}/daily-forecast"
DAILY_CSV_PATH = "daily_historical_sales.csv"


# =========================================================
# 2. LOAD MONTHLY HISTORICAL SALES
# =========================================================
@st.cache_data(ttl=3600)
def load_historical_sales():
    response = requests.get(HISTORY_API_URL, timeout=90)
    response.raise_for_status()

    data = response.json()

    if "historical_sales" not in data:
        raise ValueError("historical_sales is missing from the API.")

    df = pd.DataFrame(data["historical_sales"])

    if not {"Month", "Sales"}.issubset(df.columns):
        raise ValueError("Monthly history needs Month and Sales columns.")

    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Month", "Sales"])

    df["Month"] = df["Month"].dt.to_period("M").dt.to_timestamp()

    monthly = df.groupby("Month")["Sales"].sum().sort_index()

    if monthly.empty:
        raise ValueError("No monthly historical data was returned.")

    return monthly


# =========================================================
# 3. LOAD CONTINUOUS DAILY HISTORICAL SALES
# =========================================================
@st.cache_data(ttl=3600)
def load_daily_historical_sales():
    df = pd.read_csv(DAILY_CSV_PATH)

    required = {"Date", "Historical Sales"}

    if not required.issubset(df.columns):
        raise ValueError(
            "daily_historical_sales.csv must contain Date "
            "and Historical Sales columns."
        )

    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce"
    ).dt.normalize()

    df["Historical Sales"] = pd.to_numeric(
        df["Historical Sales"], errors="coerce"
    )

    df = df.dropna(subset=["Date", "Historical Sales"])

    # Combine multiple records on the same date.
    daily = (
        df.groupby("Date")["Historical Sales"]
        .sum()
        .sort_index()
    )

    if daily.empty:
        raise ValueError("No daily historical sales records found.")

    # IMPORTANT: Create every calendar date.
    # Dates with no recorded sales receive a value of zero.
    all_dates = pd.date_range(
        start=daily.index.min(),
        end=daily.index.max(),
        freq="D"
    )

    daily = daily.reindex(all_dates, fill_value=0)
    daily.index.name = "Date"
    daily.name = "Historical Sales"

    return daily


# =========================================================
# 4. CHART HELPER
# =========================================================
def create_sales_chart(
    history=None,
    forecast=None,
    title="Sales Trend",
    date_format="%b %Y"
):
    fig = go.Figure()

    if history is not None and not history.empty:
        fig.add_trace(
            go.Scatter(
                x=history.index,
                y=history.values,
                mode="lines+markers",
                name="Historical Sales",
                line=dict(width=2.5),
                marker=dict(size=5),
                hovertemplate=(
                    "Date: %{x|%d %b %Y}<br>"
                    "Sales: %{y:,.2f}<extra></extra>"
                )
            )
        )

    if forecast is not None and not forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast["Date"],
                y=forecast["Predicted Sales"],
                mode="lines+markers",
                name="Predicted Sales",
                line=dict(width=2.5, dash="dash"),
                marker=dict(size=4),
                hovertemplate=(
                    "Date: %{x|%d %b %Y}<br>"
                    "Predicted Sales: %{y:,.2f}<extra></extra>"
                )
            )
        )

    fig.update_layout(
        title=title,
        height=430,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=35),
        xaxis=dict(
            title="Date",
            type="date",
            tickformat=date_format,
            showgrid=False
        ),
        yaxis=dict(
            title="Sales",
            tickformat=",.2f",
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
# 5. DASHBOARD HEADER AND DATA
# =========================================================
st.title("📈 Sales Forecasting Dashboard")

st.write(
    "Explore historical sales and generate daily sales predictions "
    "using the Ridge Regression model through a FastAPI backend."
)

try:
    with st.spinner("Loading historical sales data..."):
        monthly_sales = load_historical_sales()
        daily_sales = load_daily_historical_sales()

except Exception as error:
    st.error(f"Unable to load historical data: {error}")
    st.info(
        "Check the Render backend and confirm that "
        "daily_historical_sales.csv exists beside app.py in GitHub."
    )
    st.stop()

first_historical_date = daily_sales.index.min()
last_historical_date = daily_sales.index.max()

st.info(
    f"Daily historical data: {first_historical_date:%d %B %Y} "
    f"to {last_historical_date:%d %B %Y}. "
    "Predictions are estimates, not actual future sales."
)


# =========================================================
# 6. SIDEBAR FORECAST SETTINGS
# =========================================================
st.sidebar.header("⚙️ Forecast Settings")

st.sidebar.subheader("Select Forecast Start")

selected_year = st.sidebar.selectbox(
    "Select Year",
    options=[2014, 2015, 2016, 2017],
    index=0,
    key="forecast_year"
)

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

selected_month = st.sidebar.selectbox(
    "Select Month",
    options=list(range(1, 13)),
    format_func=lambda month: month_names[month],
    index=9 if selected_year == 2014 else 0,
    key="forecast_month"
)

duration_months = st.sidebar.selectbox(
    "Forecast Duration",
    options=[1, 2, 3, 4, 5, 6],
    index=2,
    format_func=lambda months: (
        f"{months} month" if months == 1
        else f"{months} months"
    ),
    key="forecast_duration"
)

# One selected month equals 30 forecast days.
forecast_days = duration_months * 30

selected_month_start = pd.Timestamp(
    year=selected_year,
    month=selected_month,
    day=1
)

# If the selected month is historical, start after the last
# historical date. Otherwise, use the selected month's first day.
forecast_start = max(
    selected_month_start,
    last_historical_date + pd.Timedelta(days=1)
)

forecast_end = forecast_start + pd.Timedelta(
    days=forecast_days - 1
)

st.sidebar.caption(f"Forecast days: {forecast_days}")
st.sidebar.caption(f"Start date: {forecast_start:%d %b %Y}")
st.sidebar.caption(f"End date: {forecast_end:%d %b %Y}")

if selected_month_start <= last_historical_date:
    st.sidebar.warning(
        "The selected month overlaps historical data. "
        "Forecasting will start after the last historical date."
    )

st.sidebar.divider()
st.sidebar.subheader("Dataset Information")
st.sidebar.write(
    f"First date: **{first_historical_date:%d %b %Y}**"
)
st.sidebar.write(
    f"Last date: **{last_historical_date:%d %b %Y}**"
)
st.sidebar.write(
    f"Continuous historical days: **{len(daily_sales)}**"
)


# =========================================================
# 7. HISTORICAL SALES OVERVIEW
# =========================================================
st.divider()
st.subheader("📊 Historical Sales Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Historical Sales",
    f"{daily_sales.sum():,.2f}"
)
col2.metric(
    "Average Daily Sales",
    f"{daily_sales.mean():,.2f}"
)
col3.metric(
    "Historical Days",
    f"{len(daily_sales):,}"
)

st.subheader("Historical Monthly Sales")

st.plotly_chart(
    create_sales_chart(
        history=monthly_sales,
        title="Historical Monthly Sales",
        date_format="%b %Y"
    ),
    use_container_width=True,
    config={"displaylogo": False}
)

with st.expander("View Historical Monthly Sales"):
    monthly_table = monthly_sales.rename_axis("Month").reset_index()
    monthly_table["Month"] = monthly_table["Month"].dt.strftime("%B %Y")
    monthly_table["Sales"] = monthly_table["Sales"].round(2)

    st.dataframe(
        monthly_table,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 8. CONTINUOUS DAILY HISTORICAL TABLE
# =========================================================
st.divider()
st.subheader("📅 Date-wise Historical Sales")

st.caption(
    "Every calendar date is included. Dates without recorded sales "
    "are displayed with 0.00 sales."
)

with st.expander("View Historical Daily Sales", expanded=True):
    daily_table = daily_sales.rename_axis("Date").reset_index()

    daily_table["Date"] = daily_table["Date"].dt.strftime("%d %B %Y")
    daily_table["Historical Sales"] = (
        daily_table["Historical Sales"].round(2)
    )

    st.dataframe(
        daily_table[["Date", "Historical Sales"]],
        use_container_width=True,
        hide_index=True,
        height=450
    )

    historical_csv = daily_sales.rename_axis("Date").reset_index()
    historical_csv["Historical Sales"] = (
        historical_csv["Historical Sales"].round(2)
    )

    st.download_button(
        "⬇️ Download Daily Historical Sales",
        data=historical_csv.to_csv(index=False).encode("utf-8"),
        file_name="continuous_daily_historical_sales.csv",
        mime="text/csv"
    )


# =========================================================
# 9. DAILY FORECAST API
# =========================================================
st.divider()
st.subheader("🔮 Daily Sales Forecast")

st.write(
    f"Forecast period: **{forecast_start:%d %B %Y}** to "
    f"**{forecast_end:%d %B %Y}**"
)

st.write(f"Forecast duration: **{forecast_days} days**")

if st.button(
    "Generate Daily Forecast",
    type="primary",
    use_container_width=True
):
    try:
        payload = {
            "start_date": forecast_start.strftime("%Y-%m-%d"),
            "days": int(forecast_days)
        }

        with st.spinner("Generating daily predictions from Render..."):
            response = requests.post(
                DAILY_FORECAST_API_URL,
                json=payload,
                timeout=180
            )

        if response.status_code >= 400:
            try:
                error_detail = response.json().get(
                    "detail", response.text
                )
            except ValueError:
                error_detail = response.text

            st.error(
                f"Daily forecast API error "
                f"({response.status_code}): {error_detail}"
            )
            st.stop()

        result = response.json()

        records = (
            result.get("forecasts")
            or result.get("daily_forecasts")
            or result.get("predictions")
        )

        if not records:
            st.error(
                "No predictions were returned. Check the "
                "/daily-forecast API response in Render."
            )
            st.code(str(result))
            st.stop()

        forecast_df = pd.DataFrame(records)

        date_column = next(
            (
                column for column in ["date", "Date", "day"]
                if column in forecast_df.columns
            ),
            None
        )

        sales_column = next(
            (
                column for column in [
                    "predicted_sales",
                    "Predicted Sales",
                    "prediction",
                    "predicted_value"
                ]
                if column in forecast_df.columns
            ),
            None
        )

        if date_column is None or sales_column is None:
            st.error(
                "Unexpected API response. The response must contain "
                "a date field and a predicted sales field."
            )
            st.code(str(result))
            st.stop()

        forecast_df = forecast_df.rename(
            columns={
                date_column: "Date",
                sales_column: "Predicted Sales"
            }
        )

        forecast_df["Date"] = pd.to_datetime(
            forecast_df["Date"], errors="coerce"
        ).dt.normalize()

        forecast_df["Predicted Sales"] = pd.to_numeric(
            forecast_df["Predicted Sales"], errors="coerce"
        ).clip(lower=0)

        forecast_df = forecast_df.dropna(
            subset=["Date", "Predicted Sales"]
        )

        forecast_df = (
            forecast_df.groupby("Date", as_index=False)[
                "Predicted Sales"
            ]
            .last()
            .sort_values("Date")
        )

        # Create the exact expected daily date range.
        expected_dates = pd.date_range(
            start=forecast_start,
            periods=forecast_days,
            freq="D"
        )

        forecast_df = (
            forecast_df.set_index("Date")
            .reindex(expected_dates)
            .rename_axis("Date")
            .reset_index()
        )

        missing_mask = forecast_df["Predicted Sales"].isna()

        if missing_mask.any():
            missing_dates = forecast_df.loc[
                missing_mask, "Date"
            ].dt.strftime("%Y-%m-%d").tolist()

            st.error(
                f"The API is missing predictions for "
                f"{len(missing_dates)} date(s): "
                + ", ".join(missing_dates[:20])
            )

            st.warning(
                "The dates are now continuous in the table, but "
                "the missing prediction values must be fixed in "
                "the backend. They have not been fabricated."
            )

        valid_forecast = forecast_df.dropna(
            subset=["Predicted Sales"]
        ).copy()

        if valid_forecast.empty:
            st.error("No valid daily predictions are available.")
            st.stop()

        st.success(
            f"Received {len(valid_forecast)} valid daily predictions."
        )

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Requested Forecast Days",
            forecast_days
        )
        metric2.metric(
            "Total Predicted Sales",
            f"{valid_forecast['Predicted Sales'].sum():,.2f}"
        )
        metric3.metric(
            "Average Daily Forecast",
            f"{valid_forecast['Predicted Sales'].mean():,.2f}"
        )

        # DAILY FORECAST TABLE
        st.subheader("📋 Date-wise Predicted Sales")

        display_forecast = forecast_df.copy()
        display_forecast["Date"] = (
            display_forecast["Date"].dt.strftime("%Y-%m-%d")
        )
        display_forecast["Predicted Sales"] = (
            display_forecast["Predicted Sales"].round(2)
        )

        st.dataframe(
            display_forecast[["Date", "Predicted Sales"]],
            use_container_width=True,
            hide_index=True,
            height=450
        )

        # DAILY FORECAST CHART
        st.subheader("📈 Daily Predicted Sales")

        st.plotly_chart(
            create_sales_chart(
                forecast=valid_forecast,
                title="Daily Sales Forecast",
                date_format="%d %b %Y"
            ),
            use_container_width=True,
            config={"displaylogo": False}
        )

        # DOWNLOAD DAILY FORECAST
        download_df = valid_forecast.copy()
        download_df["Date"] = (
            download_df["Date"].dt.strftime("%Y-%m-%d")
        )
        download_df["Predicted Sales"] = (
            download_df["Predicted Sales"].round(2)
        )

        st.download_button(
            "⬇️ Download Daily Forecast CSV",
            data=download_df[
                ["Date", "Predicted Sales"]
            ].to_csv(index=False).encode("utf-8"),
            file_name=f"daily_sales_forecast_{forecast_days}_days.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.caption(
            "Forecasts are estimates and may differ from actual sales."
        )

    except requests.exceptions.RequestException as error:
        st.error(f"Could not connect to Render: {error}")

    except Exception as error:
        st.error(f"Forecast generation failed: {error}")


# =========================================================
# 10. FOOTER
# =========================================================
st.divider()

st.caption(
    "Sales Forecasting Dashboard | Daily Ridge Regression | "
    "FastAPI Backend"
)
