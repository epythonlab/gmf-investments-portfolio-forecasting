import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from scripts.model_training import ModelTrainer  # Replace with the correct path to the class

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        """Set up the test data and necessary objects."""
        # Sample DataFrame for testing
        data = {
            'Date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
            'Close': np.random.randn(100) * 10 + 100  # Random data for 'Close' column
        }
        self.df = pd.DataFrame(data)
        self.df.set_index('Date', inplace=True)
        self.logger = MagicMock()
        self.trainer = ModelTrainer(self.df, logger=self.logger)
        self.trainer.prepare_data(train_size=0.8)  # Ensure data is prepared before training

    def test_prepare_data(self):
        """Test data preparation and scaling."""
        self.trainer.prepare_data(train_size=0.8)

        # Test if the train and test datasets are split correctly
        self.assertEqual(len(self.trainer.train), 80)
        self.assertEqual(len(self.trainer.test), 20)

        # Test if the data is scaled (scaled data should be between 0 and 1)
        scaled_values = self.trainer.train['Close']
        self.assertTrue(np.all(scaled_values >= 0) and np.all(scaled_values <= 1))

    def test_train_arima(self):
        """Test ARIMA model training."""
        # Ensure the ARIMA model is trained
        self.trainer.train_arima()
        self.assertIn('ARIMA', self.trainer.model)
        self.logger.info.assert_called_with('Training ARIMA model')  # Check logging

    def test_train_sarima(self):
        """Test SARIMA model training."""
        self.trainer.train_sarima(seasonal_period=5)
        self.assertIn('SARIMA', self.trainer.model)
        self.logger.info.assert_called_with('Training SARIMA model')  # Check logging

    def test_train_lstm(self):
        """Test LSTM model training."""
        self.trainer.train_lstm(seq_length=60, epochs=1, batch_size=32)
        self.assertIn('LSTM', self.trainer.model)
        self.logger.info.assert_called_with('Training LSTM model')  # Check logging

    def test_make_prediction(self):
        """Test making predictions."""
        # Train models first
        self.trainer.train_arima()
        self.trainer.train_sarima(seasonal_period=5)
        self.trainer.train_lstm(seq_length=60, epochs=1, batch_size=32)

        # Make predictions
        self.trainer.make_prediction()

        # Check that predictions exist for each model
        self.assertIn('ARIMA', self.trainer.prediction)
        self.assertIn('SARIMA', self.trainer.prediction)
        self.assertIn('LSTM', self.trainer.prediction)

        # Ensure predictions are non-empty
        self.assertGreater(len(self.trainer.prediction['ARIMA']), 0)
        self.assertGreater(len(self.trainer.prediction['SARIMA']), 0)
        self.assertGreater(len(self.trainer.prediction['LSTM']), 0)

    def test_evaluate_model(self):
        """Test the evaluation of models."""
        # Train models
        self.trainer.train_arima()
        self.trainer.train_sarima(seasonal_period=5)
        self.trainer.train_lstm(seq_length=60, epochs=1, batch_size=32)

        # Generate predictions
        self.trainer.make_prediction()

        # Evaluate models
        self.trainer.evaluate_model()

        # Check that the metric for each model exists
        self.assertIn('ARIMA', self.trainer.metric)
        self.assertIn('SARIMA', self.trainer.metric)
        self.assertIn('LSTM', self.trainer.metric)

        # Ensure MAE exists in metrics
        self.assertIn('MAE', self.trainer.metric['ARIMA'])
        self.assertIn('MAE', self.trainer.metric['SARIMA'])
        self.assertIn('MAE', self.trainer.metric['LSTM'])

    def test_forecast(self):
        """Test the forecasting functionality."""
        # Train models
        self.trainer.train_arima()
        self.trainer.train_sarima(seasonal_period=5)
        self.trainer.train_lstm(seq_length=60, epochs=1, batch_size=32)

        # Forecast for 6 months (e.g., 126 days)
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            self.trainer.forecast(months=6, output_file='forecast_results.csv')
            mock_to_csv.assert_called_once_with('forecast_results.csv')

    def test_forecast_invalid_months(self):
        """Test invalid months argument in forecasting."""
        with self.assertRaises(ValueError):
            self.trainer.forecast(months=7)  # Invalid, should raise an error

if __name__ == '__main__':
    unittest.main()
