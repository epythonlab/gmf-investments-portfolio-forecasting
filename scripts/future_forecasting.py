import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import logging

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelForecaster:
    def __init__(self, model_paths, data, logger=None, column='Close'):
        """
        Initialize the model forecaster.

        Parameters:
        model_paths (dict): Paths to the saved models.
        data (pd.DataFrame): Historical data for generating forecasts.
        logger (logging.Logger): Logger instance for logging information.
        column (str): The column to forecast.
        """
        self.data = data
        self.column = column
        self.logger = logger or logging.getLogger()
        self.scaler = MinMaxScaler()
        self.models = self._load_models(model_paths)
        self.predictions = {}
        self.conf_intervals = {}

    def _load_models(self, model_paths):
        """
        Load saved models from the provided paths.

        Parameters:
        model_paths (dict): Dictionary with model names as keys and file paths as values.

        Returns:
        dict: Dictionary of loaded models.
        """
        models = {}
        try:
            for model_name, path in model_paths.items():
                if model_name in ['ARIMA', 'SARIMA']:
                    models[model_name] = joblib.load(path)
                    self.logger.info(f"{model_name} model loaded successfully from {path}.")
                elif model_name == 'LSTM':
                    models[model_name] = load_model(path)
                    self.logger.info(f"LSTM model loaded successfully from {path}.")
            return models
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise ValueError("Model loading failed")

    def forecast_arima_sarima(self, model, steps=180):
        """
        Generate forecast and confidence intervals for ARIMA/SARIMA.

        Parameters:
        model: Trained ARIMA or SARIMA model.
        steps (int): Number of periods to forecast.

        Returns:
        np.array, np.array: Forecast values and confidence intervals.
        """
        forecast, conf_int = model.predict(n_periods=steps, return_conf_int=True)
        return forecast, conf_int

    def forecast_lstm(self, model, recent_data, steps=180):
        """
        Generate forecast using LSTM model.

        Parameters:
        model: Trained LSTM model.
        recent_data (np.array): Recent data to seed LSTM predictions.
        steps (int): Number of periods to forecast.

        Returns:
        np.array: Forecast values.
        """
        predictions = []
        data = recent_data.copy()
        
        for _ in range(steps):
            pred = model.predict(data.reshape(1, data.shape[0], 1))
            predictions.append(pred[0, 0])
            data = np.append(data[1:], pred[0, 0])  # Move window forward

        # Inverse transform to get actual prices
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions.flatten()

    def generate_forecast(self, steps=180):
        """
        Generate forecasts for all models.

        Parameters:
        steps (int): Number of periods to forecast.
        """
        try:
            # Ensure the data is sorted in ascending order
            if self.data.index.is_monotonic_decreasing:
                self.data = self.data.sort_index(ascending=True)
                self.logger.info("Data was in descending order. It has been sorted in ascending order.")
            else:
                self.logger.info("Data is already in ascending order.")
          
            # Prepare recent data for LSTM
            recent_data = np.array(self.data[self.column].values[-60:]).reshape(-1, 1)
          
            # Fit the scaler on the historical data column (if not already fitted)
            if not hasattr(self.scaler, 'scale_'):
                self.scaler.fit(self.data[self.column].values.reshape(-1, 1))

            # Transform the recent data
            recent_data = self.scaler.transform(recent_data)

            for model_name, model in self.models.items():
                if model_name in ['ARIMA', 'SARIMA']:
                    forecast, conf_int = self.forecast_arima_sarima(model, steps)
                    self.predictions[model_name] = forecast
                    self.conf_intervals[model_name] = conf_int
                elif model_name == 'LSTM':
                    lstm_forecast = self.forecast_lstm(model, recent_data, steps)
                    self.predictions['LSTM'] = lstm_forecast
            self.logger.info("Forecasts generated for all models")
      
        except Exception as e:
            self.logger.error(f"Error generating forecasts: {e}")
            raise ValueError("Forecast generation failed")

    def plot_forecast(self):
      """
      Plot the forecast results for each model alongside the actual historical data.
      Also, plot the 95% confidence intervals where available.
      """
      try:
          forecast_dates = pd.date_range(start=self.data.index[-1], periods=len(self.predictions['LSTM']) + 1, freq='D')[1:]
          # Combine historical dates with forecast dates
          all_dates = self.data.index.append(forecast_dates)

          # Set up the plot
          plt.figure(figsize=(15, 8))
          plt.plot(self.data.index, self.data[self.column], label='Actual', color='blue', linewidth=2)

          # ARIMA: Plot the forecast and confidence interval
          if 'ARIMA' in self.predictions:
              arima_forecast = self.predictions['ARIMA']
              arima_conf_int = self.conf_intervals['ARIMA']
              plt.plot(forecast_dates, arima_forecast, label='ARIMA Forecast', linestyle='--', color='blue')
              plt.fill_between(forecast_dates, arima_conf_int[:, 0], arima_conf_int[:, 1], color='blue', alpha=0.2, label='ARIMA 95% CI')

          # SARIMA: Plot the forecast and confidence interval
          if 'SARIMA' in self.predictions:
              sarima_forecast = self.predictions['SARIMA']
              sarima_conf_int = self.conf_intervals['SARIMA']
              plt.plot(forecast_dates, sarima_forecast, label='SARIMA Forecast', linestyle='--', color='green')
              plt.fill_between(forecast_dates, sarima_conf_int[:, 0], sarima_conf_int[:, 1], color='green', alpha=0.2, label='SARIMA 95% CI')

          # LSTM: Plot the forecast and estimate confidence interval as +/- 1.96*std
          if 'LSTM' in self.predictions:
              lstm_forecast = self.predictions['LSTM']
              # Estimating the confidence interval for LSTM as ± 1.96 standard deviations
              lstm_std = np.std(lstm_forecast)  # Standard deviation of LSTM forecast
              lstm_upper = lstm_forecast + 1.96 * lstm_std  # Upper bound of the 95% CI
              lstm_lower = lstm_forecast - 1.96 * lstm_std  # Lower bound of the 95% CI

              plt.plot(forecast_dates, lstm_forecast, label='LSTM Forecast', linestyle='--', color='green')
              plt.fill_between(forecast_dates, lstm_lower, lstm_upper, color='red', alpha=0.2, label='LSTM 95% CI')
          
          # Adjust the x-axis to show the full date range
          plt.xticks(all_dates[::365], all_dates.strftime('%Y')[::365], rotation=45)

          # Set plot labels and title
          plt.title("Forecast Comparison with 95% Confidence Intervals", fontsize=16)
          plt.xlabel("Date", fontsize=14)
          plt.ylabel(self.column, fontsize=14)
          plt.legend(loc='best')
          plt.xticks(rotation=45)  # Rotate date labels for better readability
          sns.set(style="whitegrid")
          plt.tight_layout()
          plt.show()
          self.logger.info("Forecasts with 95% confidence intervals plotted successfully.")

      except Exception as e:
          self.logger.error(f"Error in plotting forecasts: {e}")
          raise ValueError("Plotting forecasts failed")

    def analyze_forecast(self, threshold=0.05):
      """
      Analyze and interpret the forecast results, including trend, volatility, and market opportunities/risk.
      """
      analysis_results = {}

      self.logger.info("Starting forecast analysis.")
      
      for model_name, forecast in self.predictions.items():
          # Trend Analysis
          trend = "upward" if np.mean(np.diff(forecast)) > 0 else "downward"
          trend_magnitude = np.max(np.diff(forecast))
          self.logger.info(f"{model_name} forecast shows a {trend} trend.")

          # Volatility and Risk Analysis
          volatility = np.std(forecast)
          volatility_level = "High" if volatility > threshold else "Low"
          max_price = np.max(forecast)
          min_price = np.min(forecast)
          price_range = max_price - min_price
          volatility_analysis = self._volatility_risk_analysis(forecast, threshold)

          # Market Opportunities and Risks
          opportunities_risks = self._market_opportunities_risks(trend, volatility_level)
          
          # Store results in the analysis dictionary
          analysis_results[model_name] = {
              'Trend': trend,
              'Trend_Magnitude': trend_magnitude,
              'Volatility': volatility,
              'Volatility_Level': volatility_level,
              'Max_Price': max_price,
              'Min_Price': min_price,
              'Price_Range': price_range
          }
          print(f"  Volatility and Risk: {volatility_analysis}")
          print(f"  Market Opportunities/Risks: {opportunities_risks}")
          
          # Log the detailed analysis
          self.logger.info(f"{model_name} Analysis Results:")
          self.logger.info(f"  Trend: {trend}")
          self.logger.info(f"  Trend Magnitude: {trend_magnitude:.2f}")
          self.logger.info(f"  Volatility: {volatility:.2f}")
          self.logger.info(f"  Volatility Level: {volatility_level}")
          self.logger.info(f"  Max Price: {max_price:.2f}")
          self.logger.info(f"  Min Price: {min_price:.2f}")
          self.logger.info(f"  Price Range: {price_range:.2f}")
          self.logger.info(f"  Volatility and Risk: {volatility_analysis}")
          self.logger.info(f"  Market Opportunities/Risks: {opportunities_risks}")
      
      # Return the results in a DataFrame for easy viewing
      analysis_df = pd.DataFrame(analysis_results).T
      return analysis_df

    def _volatility_risk_analysis(self, forecast, threshold):
        """
        Analyze the volatility and risk based on confidence intervals and forecast.
        """
        volatility = np.std(forecast)
        volatility_level = "High" if volatility > threshold else "Low"

        # Highlight periods of increasing volatility
        increasing_volatility = any(np.diff(forecast) > np.mean(np.diff(forecast)))
        
        if increasing_volatility:
            return f"Potential increase in volatility, which could lead to market risk."
        else:
            return f"Stable volatility, lower risk."

    def _market_opportunities_risks(self, trend, volatility_level):
        """
        Identify market opportunities or risks based on forecast trends and volatility.
        """
        if trend == "upward":
            if volatility_level == "High":
                return "Opportunity with high risk due to increased volatility."
            else:
                return "Opportunity with moderate risk due to stable volatility."
        elif trend == "downward":
            if volatility_level == "High":
                return "Risk of decline with high uncertainty."
            else:
                return "Moderate risk of decline with low volatility."
        else:
            return "Stable market, with minimal risks."
