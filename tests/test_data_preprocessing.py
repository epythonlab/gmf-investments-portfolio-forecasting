# tests/test_data_preprocessing.py

import unittest
import pandas as pd
from unittest.mock import MagicMock
from scripts.data_preprocessing import DataPreprocessor
import logging

class TestDataPreprocessor(unittest.TestCase):

    def setUp(self):
        """
        Setup test environment and mock dependencies before each test.
        """
        self.data_dir = "./test_data"
        # Set up a logger to capture log messages
        self.logger = logging.getLogger("DataAnalysisTestLogger")
        self.logger.setLevel(logging.ERROR)
        
        # Initialize DataPreprocessor with a mock logger
        self.preprocessor = DataPreprocessor(data_dir=self.data_dir, logger=self.logger)

        # Create a sample data frame for testing
        self.sample_data = pd.DataFrame({
            "Date": pd.date_range(start="2020-01-01", periods=5, freq="D"),
            "Open": [100, 101, 102, 103, 104],
            "High": [110, 111, 112, 113, 114],
            "Low": [90, 91, 92, 93, 94],
            "Close": [105, 106, 107, 108, 109],
            "Volume": [1000, 1100, 1200, 1300, 1400]
        })
        self.sample_data.set_index('Date', inplace=True)
        
        # Mock the logger to catch error messages
        self.analysis._log_error = MagicMock()
        
    def test_get_data(self):
        """
        Test the get_data function for fetching data from YFinance.
        """
        symbols = ['TSLA', 'BND', 'SPY']
        start_date = "2020-01-01"
        end_date = "2020-12-31"
        
        # Simulate data fetching
        data_paths = self.preprocessor.get_data(start_date, end_date, symbols)

        # Verify paths are returned
        self.assertTrue('TSLA' in data_paths)
        self.assertTrue('BND' in data_paths)
        self.assertTrue('SPY' in data_paths)

    def test_load_data(self):
        """
        Test the load_data function for loading data from CSV.
        """
        # Mock the CSV reading function
        symbol = 'TSLA'
        self.preprocessor.load_data(symbol)  # Make sure no exception is raised
        self.logger.error.assert_not_called()  # Ensure no error was logged

    def test_inspect_data(self):
        """
        Test the inspect_data function for checking data types, missing values, and duplicates.
        """
        inspection_results = self.preprocessor.inspect_data(self.sample_data)

        # Check if the result contains correct keys
        self.assertIn("data_types", inspection_results)
        self.assertIn("missing_values", inspection_results)
        self.assertIn("duplicate_rows", inspection_results)

    
    def test_log_error_handling(self):
        """
        Test error handling in case of missing data files.
        """
        self.logger.reset_mock()

        # Simulate missing file
        with self.assertRaises(FileNotFoundError):
            self.preprocessor.load_data('NON_EXISTENT_SYMBOL')

        # Ensure error message was logged
        self.logger.error.assert_called()

if __name__ == "__main__":
    unittest.main()
