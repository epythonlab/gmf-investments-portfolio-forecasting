import unittest
import pandas as pd
import numpy as np
import logging
from scripts.future_forecasting import ModelForecaster  # Update this with the actual import path


class TestModelForecaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up logger
        cls.logger = logging.getLogger("ModelForecasterTest")
        logging.basicConfig(level=logging.INFO)

        # Create mock historical data
        dates = pd.date_range(start="2023-01-01", periods=100, freq='D')
        prices = np.linspace(100, 200, 100) + np.random.normal(scale=5, size=100)
        cls.historical_data = pd.DataFrame({"Close": prices}, index=dates)

        # Create mock forecast data
        forecast_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=30, freq='D')
        forecast_prices = np.linspace(200, 220, 30) + np.random.normal(scale=5, size=30)
        forecast_conf_lower = forecast_prices - 10
        forecast_conf_upper = forecast_prices + 10
        cls.forecast_data = pd.DataFrame({
            "forecast": forecast_prices,
            "conf_lower": forecast_conf_lower,
            "conf_upper": forecast_conf_upper
        }, index=forecast_dates)


        # Save mock data to CSV files
        cls.historical_csv = "mock_historical.csv"
        cls.forecast_csv = "mock_forecast.csv"
        cls.historical_data.to_csv(cls.historical_csv)
        cls.forecast_data.to_csv(cls.forecast_csv)

    def setUp(self):
        # Instantiate the ModelForecaster
        self.forecaster = ModelForecaster(
            historical_csv=self.historical_csv,
            forecast_csv=self.forecast_csv,
            logger=self.logger
        )

    def test_load_data(self):
        # Test if data is loaded correctly
        self.assertIsNotNone(self.forecaster.historical_data)
        self.assertIsNotNone(self.forecaster.forecast_data)
        self.assertIn("forecast", self.forecaster.forecast_data.columns)

    def test_plot_forecast(self):
        # Test if forecast plotting works
        try:
            self.forecaster.plot_forecast()
            self.logger.info("Forecast plot generated successfully.")
        except Exception as e:
            self.fail(f"Plotting forecast failed with error: {e}")

    def test_analyze_forecast(self):
        # Test the analysis of forecast
        try:
            analysis_results = self.forecaster.analyze_forecast(threshold=5)
            self.assertIsInstance(analysis_results, pd.DataFrame)
            self.assertIn("Trend", analysis_results.columns)
            self.logger.info("Forecast analysis completed successfully.")
        except Exception as e:
            self.fail(f"Forecast analysis failed with error: {e}")

    def test_volatility_risk_analysis(self):
        # Test the private method _volatility_risk_analysis
        forecast = self.forecaster.forecast_data['forecast'].values
        result = self.forecaster._volatility_risk_analysis(forecast, threshold=5)
        self.assertIsInstance(result, str)
        self.logger.info("Volatility risk analysis returned a valid result.")

    def test_market_opportunities_risks(self):
        # Test the private method _market_opportunities_risks
        result = self.forecaster._market_opportunities_risks(trend="upward", volatility_level="Low")
        self.assertIsInstance(result, str)
        self.assertIn("Opportunity", result)
        self.logger.info("Market opportunities/risks analysis returned a valid result.")

    @classmethod
    def tearDownClass(cls):
        # Clean up mock CSV files
        import os
        if os.path.exists(cls.historical_csv):
            os.remove(cls.historical_csv)
        if os.path.exists(cls.forecast_csv):
            os.remove(cls.forecast_csv)


if __name__ == "__main__":
    unittest.main()
