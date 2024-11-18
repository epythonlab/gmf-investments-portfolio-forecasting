import unittest
import pandas as pd
import numpy as np
import logging
from scripts.model_training import ModelTrainer  # Replace with your actual filename

class TestModelTrainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a logger
        cls.logger = logging.getLogger("ModelTrainerTest")
        logging.basicConfig(level=logging.INFO)
        
        # Create a sample DataFrame with time-series data
        dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
        cls.sample_data = pd.DataFrame({
            "Close": np.sin(np.linspace(0, 20, 200)) + np.random.normal(0, 0.1, 200)
        }, index=dates)

    def setUp(self):
        # Initialize the ModelTrainer instance
        self.model_trainer = ModelTrainer(data=self.sample_data, logger=self.logger)

    def test_prepare_data(self):
        self.model_trainer.prepare_data(train_size=0.8)
        self.assertIsNotNone(self.model_trainer.train)
        self.assertIsNotNone(self.model_trainer.test)
        self.assertAlmostEqual(len(self.model_trainer.train) / len(self.sample_data), 0.8, delta=0.01)
        self.assertAlmostEqual(len(self.model_trainer.test) / len(self.sample_data), 0.2, delta=0.01)

    def test_train_arima(self):
        self.model_trainer.prepare_data()
        self.model_trainer.train_arima()
        self.assertIn('ARIMA', self.model_trainer.model)

    def test_train_sarima(self):
        self.model_trainer.prepare_data()
        self.model_trainer.train_sarima(seasonal_period=7)
        self.assertIn('SARIMA', self.model_trainer.model)

    def test_train_lstm(self):
        self.model_trainer.prepare_data()
        self.model_trainer.train_lstm(seq_length=30, epochs=1, batch_size=8)  # Reduced epochs for testing
        self.assertIn('LSTM', self.model_trainer.model)

    def test_make_prediction(self):
        self.model_trainer.prepare_data()
        self.model_trainer.train_arima()
        self.model_trainer.make_prediction()
        self.assertIn('ARIMA', self.model_trainer.prediction)
        self.assertEqual(len(self.model_trainer.prediction['ARIMA']), len(self.model_trainer.test))

    def test_evaluate_model(self):
        self.model_trainer.prepare_data()
        self.model_trainer.train_arima()
        self.model_trainer.make_prediction()
        self.model_trainer.evaluate_model()
        self.assertIn('ARIMA', self.model_trainer.metric)
        self.assertGreater(self.model_trainer.metric['ARIMA']['MAE'], 0)

    def test_plot_result(self):
        self.model_trainer.prepare_data()
        self.model_trainer.train_arima()
        self.model_trainer.make_prediction()
        try:
            self.model_trainer.plot_result()  # Check if it runs without error
        except Exception as e:
            self.fail(f"plot_result raised an exception {e}")

    def test_forecast(self):
        self.model_trainer.prepare_data()
        self.model_trainer.train_arima()
        self.model_trainer.make_prediction()
        self.model_trainer.evaluate_model()  # Ensure metrics are computed
        self.model_trainer.forecast(months=6, output_file="test_forecast.csv")
        # Check if the forecast file is created
        forecast_df = pd.read_csv("test_forecast.csv")
        self.assertGreater(len(forecast_df), 0)


if __name__ == "__main__":
    unittest.main()
