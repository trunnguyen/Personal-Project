import streamlit as st
import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.db_manager import JobDB
from src.analytics.forecaster import JobForecaster
base_data_path = os.path.join(PROJECT_ROOT, 'data')
db_path= os.path.join(base_data_path,"jobs.db")
csv_path = os.path.join(base_data_path,"jobs.csv")

db=JobDB(db_path,csv_path)

st.set_page_config(page_title="AI Engineer Job Tracker", layout="wide")
st.title("AI Engineer Job Board")

st.subheader("Market Trend Forecasting")
with st.spinner("Analyzing historical job ingestion patterns..."):
    forecaster = JobForecaster(db_path)
    # Forecast the next 14 days of job postings
    forecast_df = forecaster.forecast_market_trend(periods_to_predict=14)

if forecast_df is not None and not forecast_df.empty:
    # Format dates nicely for displaying on charts
    forecast_df['ds'] = forecast_df['ds'].dt.date

    # Create two columns to isolate the metrics from the visual chart
    metric_col, chart_col = st.columns([1, 3])

    with metric_col:
        # Calculate expected job postings in the near future
        avg_predicted = forecast_df['yhat'].mean()
        st.metric(
            label="Avg. Predicted Daily Postings",
            value=f"{avg_predicted:.1f} jobs/day",
            help="Based on Facebook Prophet time series projection of your local database rows."
        )
        st.caption("This prediction tracks historical scraping ingestion volume to look for seasonal drops or surges.")

    with chart_col:
        # Prepare data for plotting by setting the date as index
        chart_data = forecast_df.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']]
        # Rename columns for a professional dashboard layout
        chart_data.columns = ['Predicted Postings', 'Lower Bound Margin', 'Upper Bound Margin']

        # Plot using Streamlit's high-performance native line chart
        st.line_chart(chart_data, height=220)
else:
    st.info(
        "Collect at least 5 distinct days of historical job scraping entries to activate automated trend forecasting updates.")

st.divider()

jobs = db.get_all_jobs()
st.write(f"Tracking {len(jobs)} unique jobs found in Ho Chi Minh City")

for job in jobs:

    job_id=job["id"]
    title=job["title"]
    company=job["company"]
    location=job["location"]
    time_posted=job["time_posted"]
    url=job["job_url"]
    ai_score=job["ai_score"]
    is_applied=job["is_applied"]


    with st.container():
        st.markdown(
            f"""
            <div style="
                border:1px solid #ddd;
                border-radius:10px;
                padding:15px;
                margin-bottom:15px;
                background-color:#f9f9f9;
                box-shadow:2px 2px 2px rgba(0,0,0,0.1);
                display:flex;
                justify-content:space-between;
                align-items:center;">
                <div>
                <h3 style="margin:0; color:#555">{title}</h3>
                <p style="margin:5px 0;color:#555"><b>Company: {company}</b> |  Location:{location}</p>
                <p style= "margin:2px 0;font-size:0.9em; color:#777;">Posted: {time_posted}</p>
                </span>
                <br><br>
                <a href="{url}" target="_blank" style="text-decoration:none; color:#1f77b4;font-weight:bold">View Job Post</a>
            """,
            unsafe_allow_html=True
        )
        col1, col2 = st.columns([1,5])
        with col1:
            if is_applied == 0:
                if st.button("Interested Jobs",key=f"btn_int_{job_id}"):
                    db.update_score_and_interest(job_id,  is_applied = 1)
                    st.rerun()
            else:
                if st.button("Not Interested",key=f"btn_job_{job_id}"):
                    db.update_score_and_interest(job_id, is_applied = 0)
                    st.rerun()
        st.divider()