import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
import matplotlib.pyplot as plt
from scripts.seasonal_analysis import SeasonalAnalysis

class TestSeasonalAnalysis(unittest.TestCase):
    
    def setUp(self):
        """Set up a logger mock, sample data, and an instance of SeasonalAnalysis."""
        # Set up a logger mock to capture log outputs
        self.logger = MagicMock()
        
        # Initialize SeasonalAnalysis with the mock logger
        self.analysis = SeasonalAnalysis(logger=self.logger)
        
        # Generate synthetic time series data with exactly 504 observations
        date_range = pd.date_range(start="2020-01-01", periods=504, freq="D")
        
        # Define a linear trend and a seasonal pattern (sine wave for simplicity)
        trend_component = np.linspace(0, 50, 504)
        seasonal_component = 10 * np.sin(2 * np.pi * date_range.dayofyear / 365.25)
        
        # Adding noise to the data
        noise = np.random.normal(0, 1, 504)
        
        # Creating the synthetic data series with trend, seasonality, and noise
        self.sample_data = pd.DataFrame({
            "Date": date_range,
            "Close": 100 + seasonal_component + trend_component + noise
        }).set_index("Date")



    def test_adf_test(self):
        """Test ADF test for p-value output."""
        p_value = self.analysis.adf_test(self.sample_data["Close"])
        self.assertIsInstance(p_value, float, "ADF test did not return a float p-value.")
        
    def test_difference_series(self):
        """Test that differencing reduces trend and leaves residuals."""
        differenced_series = self.analysis.difference_series(self.sample_data["Close"])
        
        # Check that differencing reduces trend by looking at mean difference
        self.assertGreater(self.sample_data["Close"].std(), differenced_series.std(), 
                           "Differencing did not reduce the standard deviation as expected.")
        
        # Check that differencing doesn't result in NaNs beyond the initial lag
        self.assertFalse(differenced_series.isnull().any(), "Differencing should not produce NaN values beyond the initial lag.")

    def test_decompose_series(self):
        """Test time series decomposition into trend, seasonal, and residuals."""
        decomposition = self.analysis.decompose_series(self.sample_data["Close"])
        
        # If decomposition was possible, check for trend, seasonal, and residual components
        if decomposition:
            self.assertIsNotNone(decomposition.trend, "Decomposition did not produce a trend component.")
            self.assertIsNotNone(decomposition.seasonal, "Decomposition did not produce a seasonal component.")
            self.assertIsNotNone(decomposition.resid, "Decomposition did not produce a residual component.")
        else:
            # Verify that an error was logged if decomposition couldn't proceed
            self.logger.error.assert_called_with(
                "Series length insufficient for decomposition; requires at least 504 observations."
            )


    def test_analyze_trends_and_seasonality(self):
        """Test the overall analysis of trends and seasonality for error-free execution."""
        data_dict = {"TEST": self.sample_data}
        
        # Call the method
        self.analysis.analyze_trends_and_seasonality(data_dict, threshold=0.05)
        
        # Check if the error was called due to insufficient length after differencing
        if len(self.sample_data) - 1 < 504:  # Length after one differencing operation
            self.logger.error.assert_called_with(
                "Series length insufficient for decomposition; requires at least 504 observations."
            )
        else:
            self.logger.error.assert_not_called()



if __name__ == "__main__":
    unittest.main()
