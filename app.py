                    # 12. FORECAST CHART
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
                    # 13. HISTORICAL VS FORECAST
                    # ---------------------------------------------
                    st.subheader("Historical vs Forecast Sales")

                    recent_history = monthly_sales[
                        monthly_sales.index < selected_forecast_start
                    ].tail(12)

                    comparison_fig = create_sales_chart(
                        historical_data=recent_history,
                        forecast_data=forecast_df,
                        title="Historical vs Forecast Sales",
                        height=500
                    )

                    chart_start = (
                        recent_history.index.min()
                        if not recent_history.empty
                        else forecast_df["Month"].min()
                    )

                    chart_end = (
                        forecast_df["Month"].max()
                        + pd.offsets.MonthEnd(0)
                    )

                    comparison_fig.update_xaxes(
                        range=[chart_start, chart_end]
                    )

                    st.plotly_chart(
                        comparison_fig,
                        use_container_width=True,
                        config={"displaylogo": False}
                    )

                    # ---------------------------------------------
                    # 14. DOWNLOAD FORECAST CSV
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

                    st.download_button(
                        label="⬇️ Download Forecast as CSV",
                        data=download_df.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name=(
                            f"sales_forecast_{duration_days}_days.csv"
                        ),
                        mime="text/csv",
                        use_container_width=True
                    )

                    st.caption(
                        "Forecast values are estimates and may differ "
                        "from actual sales."
                    )

    except requests.exceptions.RequestException as error:
        st.error(
            f"Could not connect to the forecasting backend: {error}"
        )

    except Exception as error:
        st.error(f"Forecast generation failed: {error}")


# =========================================================
# 15. FOOTER
# =========================================================
st.divider()

st.caption(
    "Sales Forecasting Dashboard | Ridge Regression | "
    "Monthly Time-Series Forecasting"
)
