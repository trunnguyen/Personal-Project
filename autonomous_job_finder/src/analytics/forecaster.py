import pandas as pd
import sqlite3
from prophet import Prophet
from pathlib import Path
import sys

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.logger import logger

class JobForecaster:
    def __init__(self,db_path):
        self.db_path = db_path

    def forecast_market_trend(self, periods_to_predict=14):
        try:
            with sqlite3.connect(self.db_path) as conn:
                df=pd.read_sql_query(
                    "SELECT date(date_bound) as ds, COUNT(id) as y FROM jobs GROUP BY ds",
                    conn
                )

            if len(df) < 5:
                logger.warning("Not enough data to forecast")
                return None

            model = Prophet(yearly_seasonality=False, daily_seasonality=False)
            model.fit(df)

            future = model.make_future_dataframe(periods=periods_to_predict)
            forecast = model.predict(future)

            logger.info("Market trend forecast completed")
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods_to_predict)
        except Exception as e:
            logger.error(f"Forecasting encountered an error: {e}")
            return None