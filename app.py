
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("Sales Forecasting Dashboard")
st.write("Forecast future monthly sales using Ridge Regression.")

@st.cache_resource
def load_bundle():
    return joblib.load("sales_forecasting_bundle.joblib")

bundle = load_bundle()
model = bundle["model"]
monthly_sales = bundle["monthly_sales"]
feature_columns = bundle["feature_columns"]

def forecast_sales(model, history, dates):
    history = history.copy().astype(float)
    results = []
    first_year = monthly_sales.index.min().year

    for date in dates:
        features = {
            "month": date.month,
            "year_number": date.year - first_year,
            "lag_1": history.iloc[-1],
            "lag_2": history.iloc[-2],
            "lag_3": history.iloc[-3],
            "lag_12": history.iloc[-12],
            "rolling_mean_3": history.iloc[-3:].mean(),
            "rolling_mean_12": history.iloc[-12:].mean()
        }

        X = pd.DataFrame([features], columns=feature_columns)
        prediction = max(0.0, float(model.predict(X)[0]))
        results.append(prediction)
        history.loc[date] = prediction

    return results

st.sidebar.header("Forecast Settings")
horizon = st.sidebar.selectbox(
    "Forecast period (months)",
    [3, 6, 12],
    index=2
)

st.subheader("Historical Monthly Sales")
st.line_chart(monthly_sales)

if st.button("Generate Sales Forecast"):
    future_months = pd.date_range(
        start=monthly_sales.index.max() + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS"
    )

    predictions = forecast_sales(model, monthly_sales, future_months)

    forecast_df = pd.DataFrame({
        "Forecast Month": future_months.strftime("%B %Y"),
        "Predicted Sales": np.round(predictions, 2)
    })

    st.session_state["forecast_df"] = forecast_df

if "forecast_df" in st.session_state:
    result = st.session_state["forecast_df"]

    col1, col2 = st.columns(2)
    col1.metric("Total Forecast Sales",
                f'{result["Predicted Sales"].sum():,.2f}')
    col2.metric("Average Monthly Sales",
                f'{result["Predicted Sales"].mean():,.2f}')

    st.dataframe(result, use_container_width=True, hide_index=True)
    st.line_chart(result.set_index("Forecast Month")["Predicted Sales"])

    st.download_button(
        "Download Forecast CSV",
        result.to_csv(index=False).encode("utf-8"),
        file_name="future_monthly_sales_forecast.csv",
        mime="text/csv"
    )
