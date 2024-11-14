import unittest
import pandas as pd
from unittest.mock import MagicMock, patch
from scripts.data_preprocessing import DataPreprocessor
import logging
import os

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
        self.preprocessor._log_error = MagicMock()

    @patch("scripts.data_preprocessing.pn.data.get")
    def test_get_data(self, mock_get):
        """
        Test the get_data function for fetching data from YFinance.
        """
        # Set up mock data return
        mock_get.return_value = self.sample_data
        
        symbols = ['TSLA', 'BND', 'SPY']
        start_date = "2020-01-01"
        end_date = "2020-12-31"
        
        # Simulate data fetching
        data_paths = self.preprocessor.get_data(start_date, end_date, symbols)

        # Verify that paths are created for each symbol
        for symbol in symbols:
            self.assertIn(symbol, data_paths)
            self.assertTrue(os.path.exists(data_paths[symbol]))

    def test_load_data(self):
        """
        Test the load_data function for loading data from CSV.
        """
        # Save sample data as CSV for testing load_data
        test_symbol = "TEST"
        file_path = os.path.join(self.data_dir, f"{test_symbol}.csv")
        self.sample_data.to_csv(file_path)

        # Load data and verify
        loaded_data = self.preprocessor.load_data(test_symbol)
        pd.testing.assert_frame_equal(loaded_data, self.sample_data)

    def test_inspect_data(self):
        """
        Test the inspect_data function for checking data types, missing values, and duplicates.
        """
        # Insert a missing value and duplicate row for testing
        sample_data_with_issues = self.sample_data.copy()
        sample_data_with_issues.iloc[2, sample_data_with_issues.columns.get_loc("Open")] = None
        sample_data_with_issues = pd.concat([sample_data_with_issues, sample_data_with_issues.iloc[[1]]])

        inspection_results = self.preprocessor.inspect_data(sample_data_with_issues)

        # Check for missing values, duplicate rows, and data types
        self.assertIn("data_types", inspection_results)
        self.assertEqual(inspection_results["missing_values"]["Open"], 1)
        self.assertEqual(inspection_results["duplicate_rows"], 1)


    def test_log_error_handling(self):
        """
        Test error handling in case of missing data files.
        """
        # Simulate missing file
        with self.assertRaises(FileNotFoundError):
            self.preprocessor.load_data('NON_EXISTENT_SYMBOL')

        # Ensure the error was logged
        self.preprocessor._log_error.assert_called_once()
        self.preprocessor._log_error.assert_called_with(
            "Data file for symbol 'NON_EXISTENT_SYMBOL' not found. Run `get_data()` first."
        )
    
    def test_analyze_data(self):
        """
        Test the analyze_data function for calculating mean, median, standard deviation,
        and checking for missing values.
        """
        # Add a missing value for testing
        sample_data_with_missing = self.sample_data.copy()
        sample_data_with_missing.iloc[0, sample_data_with_missing.columns.get_loc("Open")] = None

        # Call the analyze_data method
        analysis_results = self.preprocessor.analyze_data(sample_data_with_missing)

        # Verify that the results contain the expected keys
        self.assertIn("mean", analysis_results)
        self.assertIn("median", analysis_results)
        self.assertIn("std_dev", analysis_results)
        self.assertIn("missing_values", analysis_results)

        # Check values for mean, median, and std_dev for known data
        expected_mean = sample_data_with_missing.mean()
        expected_median = sample_data_with_missing.median()
        expected_std_dev = sample_data_with_missing.std()
        expected_missing_values = sample_data_with_missing.isnull().sum()

        # Use assert_series_equal for numerical comparisons with a relative tolerance for floating-point differences
        pd.testing.assert_series_equal(analysis_results["mean"], expected_mean, rtol=1e-5)
        pd.testing.assert_series_equal(analysis_results["median"], expected_median, rtol=1e-5)
        pd.testing.assert_series_equal(analysis_results["std_dev"], expected_std_dev, rtol=1e-5)
        pd.testing.assert_series_equal(analysis_results["missing_values"], expected_missing_values)


    def tearDown(self):
        """
        Clean up test data directory after each test.
        """
        for file in os.listdir(self.data_dir):
            os.remove(os.path.join(self.data_dir, file))
        os.rmdir(self.data_dir)

if __name__ == "__main__":
    unittest.main()
