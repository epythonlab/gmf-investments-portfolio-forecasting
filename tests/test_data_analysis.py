import unittest
import logging
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from scripts.data_analysis import DataAnalysis  # Replace with your actual module path

class TestDataAnalysis(unittest.TestCase):
    
    def setUp(self):
        # Set up a logger to capture log messages
        self.logger = logging.getLogger("DataAnalysisTestLogger")
        self.logger.setLevel(logging.ERROR)
        
        # Initialize DataAnalysis with a mock logger
        self.analysis = DataAnalysis(logger=self.logger)
        
        # Generate sample data
        self.sample_data = {
            'TSLA': pd.DataFrame({
                'Date': pd.date_range(start="2023-01-01", periods=100, freq='D'),
                'Close': np.random.normal(150, 5, 100).cumsum()
            }).set_index('Date')
        }
        
        # Empty data for error handling tests
        self.empty_data = {'TSLA': pd.DataFrame()}
        
        # Mock the logger to catch error messages
        self.analysis._log_error = MagicMock()
        
        
    def test_plot_percentage_change_with_data(self):
        # Test with valid data
        self.analysis.plot_percentage_change(self.sample_data)
        # If no exceptions are raised, the test passes
        self.assertTrue(True)
    
    def test_plot_percentage_change_with_empty_data(self):
        # Run method with empty data
        self.analysis.plot_percentage_change(self.empty_data)
        
        # Check if the correct error message was logged
        self.analysis._log_error.assert_called_with("DataFrame for TSLA is empty.")
    
    def test_analyze_price_trend_with_data(self):
        # Test with valid data and a 20-day window
        self.analysis.analyze_price_trend(self.sample_data, window_size=20)
        self.assertTrue(True)
    
    def test_analyze_price_trend_with_empty_data(self):
       
        # Run method with empty data
        self.analysis.analyze_price_trend(self.empty_data)
        
        # Check if the correct error message was logged
        self.analysis._log_error.assert_called_with("DataFrame for TSLA is empty.")
    
    def test_plot_unusual_daily_return_with_data(self):
        # Test with valid data and a 2.5 threshold
        self.analysis.plot_unusual_daily_return(self.sample_data, threshold=2.5)
        self.assertTrue(True)
    
    def test_plot_unusual_daily_return_with_empty_data(self):
             
        # Run method with empty data
        self.analysis.plot_unusual_daily_return(self.empty_data)
        
        # Check if the correct error message was logged
        self.analysis._log_error.assert_called_with("DataFrame for TSLA is empty.")

if __name__ == "__main__":
    unittest.main()
