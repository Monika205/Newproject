@app.get("/history")
def history():
    return {
        "historical_sales": [
            {
                "Month": date.strftime("%Y-%m-%d"),
                "Sales": float(sales)
            }
            for date, sales in monthly_sales.items()
        ]
    }
