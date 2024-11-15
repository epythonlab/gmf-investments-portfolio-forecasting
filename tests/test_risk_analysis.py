import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from scripts.risk_analysis import RiskAnalysis  # assuming RiskAnalysis is in a file named risk_analysis.py

class TestRiskAnalysis(unittest.TestCase):

    def setUp(self):
        """Set up test environment before each test."""
        # Mock logger to capture log messages
        self.logger = MagicMock()

        # Sample data for testing
        self.sample_data = pd.DataFrame({
            "Date": pd.date_range(start="2020-01-01", periods=5, freq="D"),
            "Close": [100, 102, 101, 103, 102]
        })
        self.sample_data.set_index("Date", inplace=True)
        
        # Dictionary of sample data
        self.data_dict = {"TEST": self.sample_data}

        # Initialize RiskAnalysis with sample data and mock logger
        self.risk_analysis = RiskAnalysis(self.data_dict, risk_free_rate=0.01, logger=self.logger)

    def test_calculate_daily_return(self):
        """Test daily returns calculation."""
        daily_returns = self.risk_analysis.calculate_daily_return(self.sample_data)
        
        # Expected returns calculated manually or based on known values
        expected_returns = pd.Series([0.02, -0.009804, 0.019802, -0.009709], index=self.sample_data.index[1:])
        expected_returns.name = 'Close'  # Set the name to match the daily_returns name
        
        # Use rtol and atol for relative and absolute tolerance comparison
        pd.testing.assert_series_equal(
            daily_returns, 
            expected_returns, 
            rtol=1e-4,  # relative tolerance
            atol=1e-5   # absolute tolerance
        )
    def test_calculate_VaR(self):
        """Test Value at Risk (VaR) calculation at 95% confidence level."""
        daily_returns = self.risk_analysis.calculate_daily_return(self.sample_data)
        VaR = self.risk_analysis.calculate_VaR(daily_returns, confidence_level=0.95)
        
        expected_VaR = np.percentile(daily_returns, 5)  # equivalent to 1 - 0.95 confidence
        self.assertAlmostEqual(VaR, expected_VaR, places=4)

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe Ratio calculation."""
        daily_returns = self.risk_analysis.calculate_daily_return(self.sample_data)
        sharpe_ratio = self.risk_analysis.calculate_sharpe_ratio(daily_returns)
        
        # Expected Sharpe Ratio: calculated manually for the test data
        excess_returns = daily_returns - (0.01 / 252)
        expected_sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        
        self.assertAlmostEqual(sharpe_ratio, expected_sharpe_ratio, places=4)

    def test_empty_data(self):
        """Test that empty DataFrame logs an error and handles it gracefully."""
        self.risk_analysis.data_dict["EMPTY"] = pd.DataFrame()  # Add an empty DataFrame
        self.risk_analysis.analyze_risk_and_return()
        
        # Verify that logger.error was called due to empty DataFrame
        self.logger.error.assert_called_with("DataFrame for EMPTY is empty.")

    def test_analyze_risk_and_return(self):
        """Test analyze_risk_and_return function for valid calculations and error handling."""
        results = self.risk_analysis.analyze_risk_and_return(confidence_level=0.95)
        
        # Check if the results contain the expected metrics for the sample data
        self.assertIn("VaR", results.loc["TEST"])
        self.assertIn("Sharpe Ratio", results.loc["TEST"])

        # Check if VaR and Sharpe Ratio have valid values (not NaN)
        self.assertFalse(pd.isna(results.loc["TEST"]["VaR"]))
        self.assertFalse(pd.isna(results.loc["TEST"]["Sharpe Ratio"]))

    def test_missing_close_column(self):
        """Test that a missing 'Close' column logs an error."""
        # Remove the 'Close' column
        missing_close_data = pd.DataFrame({
            "Date": pd.date_range(start="2020-01-01", periods=5, freq="D"),
            "Open": [100, 101, 102, 103, 104]
        }).set_index("Date")
        
        self.risk_analysis.data_dict["MISSING_CLOSE"] = missing_close_data
        daily_returns = self.risk_analysis.calculate_daily_return(missing_close_data)
        
        # Verify that logger.error was called due to missing 'Close' column
        self.logger.error.assert_called_with("The 'Close' column is missing in the data.")
        
        # Check that the function returned an empty Series
        self.assertTrue(daily_returns.empty)

if __name__ == "__main__":
    unittest.main()
