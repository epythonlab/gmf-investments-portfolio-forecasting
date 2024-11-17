import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


class ModelForecaster:
    def __init__(self, historical_csv, forecast_csv, logger=None, column='Close'):
        """
        Initialize the model forecaster.

        Parameters:
        model_paths (dict): Paths to the saved models.
        data (pd.DataFrame): Historical data for generating forecasts.
        logger (logging.Logger): Logger instance for logging information.
        column (str): The column to forecast.
        """
        self.historical_data = historical_csv
        self.forecast_csv = forecast_csv
        self.column = column
        self.logger = logger
        self.predictions = {}

        # Load data
        self._load_data()


    def _load_data(self):
        """
        Load historical and forecast data from CSV files.
        """
        try:
            # Load historical data
            if isinstance(self.historical_data, str):
                self.historical_data = pd.read_csv(self.historical_data, index_col=0, parse_dates=True)

            # Ensure the historical data contains the required column
            if self.column not in self.historical_data.columns:
                raise ValueError(f"Historical data must have a '{self.column}' column.")

            # Load forecast data
            self.forecast_data = pd.read_csv(self.forecast_csv, index_col=0, parse_dates=True)

            # Ensure the forecast data contains the required columns
            if 'forecast' not in self.forecast_data.columns:
                raise ValueError("Forecast CSV must have a 'forecast' column.")

            # Extract forecast values
            self.predictions['forecast'] = self.forecast_data['forecast'].values
            self.forecast_dates = self.forecast_data.index

            self.logger.info("Historical and forecast data loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise ValueError("Error loading data")



    def plot_forecast(self):
        """
        Plot the historical data alongside the forecast data with confidence intervals.
        """
        try:
            # Historical and forecast dates
            historical_dates = self.historical_data.index
            forecast_dates = self.forecast_dates

            # Set up the plot
            plt.figure(figsize=(15, 8))

            # Plot historical data
            plt.plot(historical_dates, self.historical_data[self.column], label='Actual', color='blue', linewidth=2)

            # Plot forecast data
            forecast = self.predictions['forecast']
            plt.plot(forecast_dates, forecast, label='Forecast', linestyle='--', color='red')

            # Plot confidence intervals
            plt.fill_between(
                forecast_dates,
                self.forecast_data['conf_lower'],
                self.forecast_data['conf_upper'],
                color='red', alpha=0.25, label='95% Confidence Interval'
            )

            # Set up labels and title
            plt.xticks(rotation=45)
            plt.title("Historical vs. Forecast Data with Confidence Intervals", fontsize=16)
            plt.xlabel("Date", fontsize=14)
            plt.ylabel(self.column, fontsize=14)
            plt.legend(loc='best')
            sns.set(style="whitegrid")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            self.logger.error(f"Error in plotting forecasts: {e}")
            raise ValueError("Error plotting forecasts")



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
        Analyze the volatility and risk based on forecast data.
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
