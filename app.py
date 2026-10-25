        # ==================================================
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

        # ==================================================
        # 15. CSV DOWNLOAD
        # ==================================================
        st.divider()
        st.subheader("Download Forecast")

        download_df = forecast_df.copy()
        download_df["Month"] = download_df["Month"].dt.strftime("%Y-%m-%d")
        download_df["Predicted Sales"] = (
            download_df["Predicted Sales"].round(2)
        )

        st.download_button(
            label="⬇️ Download Forecast as CSV",
            data=download_df.to_csv(index=False).encode("utf-8"),
            file_name=f"sales_forecast_{selected_year}_{month_names[selected_month]}_{horizon}_months.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.caption(
            "Forecasts are estimates, not guaranteed actual sales. "
            "Longer recursive forecasts can be less reliable."
        )

    except Exception as error:
        st.error(f"Forecast generation failed: {error}")
        st.info(
            "Please verify that the saved model's feature names and "
            "feature calculations match the original training notebook."
        )

# ==================================================
# 16. FOOTER
# ==================================================
st.divider()
st.caption(
    "Sales Forecasting Project | Ridge Regression | "
    "Monthly Time-Series Forecasting"
)
