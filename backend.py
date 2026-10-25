
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import pandas as pd
import joblib

app = FastAPI(title="Sales Forecasting API")

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "sales_forecasting_bundle.joblib"
)

try:
    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    monthly_sales = bundle["monthly_sales"]
    feature_columns = list(bundle["feature_columns"])

    if isinstance(monthly_sales, pd.DataFrame):
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
        raise ValueError("At least 12 historical months are required.")

except Exception as error:
    raise RuntimeError(f"Could not load model bundle: {error}")


first_historical_month = monthly_sales.index.min()


class ForecastRequest(BaseModel):
    year: int = Field(ge=2011, le=2014)
    month: int = Field(ge=1, le=12)
    horizon: int = Field(default=3, ge=1, le=3)


def generate_forecast(start_date, horizon):
    # Use data strictly before the selected forecast month.
    history = monthly_sales[
        monthly_sales.index < start_date
    ].copy()

    if len(history) < 12:
        raise ValueError(
            "Select a later month. At least 12 months of "
            "historical data before the forecast month are required."
        )

    predictions = []

    for _ in range(horizon):
        next_month = history.index.max() + pd.offsets.MonthBegin(1)

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
            "rolling_mean_12": history.iloc[-12:].mean(),
        }

        input_data = pd.DataFrame(
            [[features[column] for column in feature_columns]],
            columns=feature_columns,
        )

        prediction = float(model.predict(input_data)[0])

        # Sales quantity/amount predictions should not be negative.
        prediction = max(0.0, prediction)

        history.loc[next_month] = prediction

        predictions.append({
            "month": next_month.strftime("%Y-%m-%d"),
            "predicted_sales": prediction,
        })

    return predictions


@app.get("/")
def home():
    return {"message": "Sales Forecasting API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/forecast")
def forecast(request: ForecastRequest):
    try:
        start_date = pd.Timestamp(
            year=request.year,
            month=request.month,
            day=1,
        )

        result = generate_forecast(
            start_date=start_date,
            horizon=request.horizon,
        )

        return {
            "forecast_start": result[0]["month"],
            "forecast_end": result[-1]["month"],
            "horizon": request.horizon,
            "forecasts": result,
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast generation failed: {error}",
        )
