import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from io import StringIO

# Assuming your DataPreprocessor class is in a module named `data_preprocessor`
from scripts.data_preprocessing import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):

    @patch("data_preprocessor.pn.data.get")
    def test_get_data(self, mock_get_data):
        """
        Test the get_data method for successful data fetching and saving to CSV.
        """
        # Create mock data to return
        mock_data = pd.DataFrame({
            "Date": pd.date_range(start="2021-01-01", periods=5),
            "Open": [100, 101, 102, 103, 104],
            "Close": [105, 106, 107, 108, 109],
        })
        mock_get_data.return_value = mock_data

        # Initialize DataPreprocessor with a mock logger
        dp = DataPreprocessor(logger=MagicMock())

        # Run the method
        result = dp.get_data("2021-01-01", "2021-01-05", ["TSLA"])

        # Check the result
        self.assertIn("TSLA", result)
        self.assertTrue(result["TSLA"].endswith("TSLA.csv"))
        self.assertTrue(mock_get_data.called)

    @patch("pandas.read_csv")
    def test_load_data(self, mock_read_csv):
        """
        Test the load_data method.
        """
        # Mock data returned by pd.read_csv
        mock_data = pd.DataFrame({
            "Date": pd.date_range(start="2021-01-01", periods=5),
            "Open": [100, 101, 102, 103, 104],
            "Close": [105, 106, 107, 108, 109],
        })
        mock_read_csv.return_value = mock_data

        dp = DataPreprocessor(logger=MagicMock())

        # Test load data
        data = dp.load_data("TSLA")
        self.assertEqual(data.shape, (5, 5))  # Ensure we got the correct number of rows and columns

    def test_detect_outliers_iqr(self):
        """
        Test outlier detection using IQR method.
        """
        # Prepare test data
        data = pd.DataFrame({
            "Date": pd.date_range(start="2021-01-01", periods=5),
            "Open": [100, 200, 300, 1000, 500],
            "Close": [105, 205, 305, 1005, 505],
        })
        data.set_index("Date", inplace=True)

        dp = DataPreprocessor(logger=MagicMock())
        outliers = dp.detect_outliers(data, method="iqr")

        # Check if the outliers were detected correctly
        self.assertTrue(outliers["Open"].iloc[3])  # The 1000 value should be detected as an outlier
        self.assertTrue(outliers["Close"].iloc[3])

    def test_handle_outliers(self):
        """
        Test handling outliers by replacing with NaN and interpolating.
        """
        # Prepare test data with outliers
        data = pd.DataFrame({
            "Date": pd.date_range(start="2021-01-01", periods=5),
            "Open": [100, 200, 300, 1000, 500],
            "Close": [105, 205, 305, 1005, 505],
        })
        data.set_index("Date", inplace=True)

        # Prepare outliers
        outliers = pd.DataFrame({
            "Open": [False, False, False, True, False],
            "Close": [False, False, False, True, False],
        })
        outliers.set_index("Date", inplace=True)

        dp = DataPreprocessor(logger=MagicMock())

        # Handle outliers
        cleaned_data = dp.handle_outliers({"TSLA": data}, {"TSLA": outliers})

        # Check if outliers were handled by replacing with NaN
        self.assertTrue(np.isnan(cleaned_data["TSLA"]["Open"].iloc[3]))  # The outlier row should have NaN
        self.assertTrue(np.isnan(cleaned_data["TSLA"]["Close"].iloc[3]))

    def test_normalize_data(self):
        """
        Test normalization of stock data.
        """
        # Prepare test data
        data = pd.DataFrame({
            "Date": pd.date_range(start="2021-01-01", periods=5),
            "Open": [100, 200, 300, 400, 500],
            "Close": [105, 205, 305, 405, 505],
        })
        data.set_index("Date", inplace=True)

        dp = DataPreprocessor(logger=MagicMock())
        normalized_data = dp.normalize_data(data)

        # Check if the data is normalized
        self.assertTrue(np.allclose(normalized_data["Open"].mean(), 0, atol=0.1))
        self.assertTrue(np.allclose(normalized_data["Close"].mean(), 0, atol=0.1))

    @patch("matplotlib.pyplot.show")
    def test_plot_outliers(self, mock_show):
        """
        Test the plotting of outliers.
        """
        # Prepare test data
        data = pd.DataFrame({
            "Date": pd.date_range(start="2021-01-01", periods=5),
            "Open": [100, 200, 300, 1000, 500],
            "Close": [105, 205, 305, 1005, 505],
        })
        data.set_index("Date", inplace=True)

        dp = DataPreprocessor(logger=MagicMock())
        outliers = dp.detect_outliers(data, method="iqr")

        # Test plotting outliers
        dp.plot_outliers(data, outliers, "TSLA")
        mock_show.assert_called_once()  # Ensure plt.show() was called


if __name__ == "__main__":
    unittest.main()
