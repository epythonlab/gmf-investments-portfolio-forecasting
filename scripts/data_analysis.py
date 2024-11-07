import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

class DataAnalysis:
    def __init__(self, data_dir='../data/', logger=None):
        """
        Initializes the DataAnalysis class.

        Parameters:
        - data_dir (str): Directory containing the data files.
        - logger (logging.Logger, optional): Logger instance to log errors.
        """
        self.data_dir = data_dir
        self.logger = logger

    def _log_error(self, message):
        """Logs error messages to the log file."""
        if self.logger:
            self.logger.error(message)
        print(f"Error: {message}")

    

    def plot_percentage_change(self, data_dict):
        """Plots the daily percentage change in the closing price."""
        for symbol, df in data_dict.items():
            
            if df is None or df.empty:
                self._log_error(f"DataFrame for {symbol} is empty.")
                return
            try:
                df['Pct_Change'] = df['Close'].pct_change() * 100
                plt.figure(figsize=(10, 6))
                plt.plot(df.index, df['Pct_Change'], label=f'{symbol} Daily Percentage Change')
                plt.title(f"{symbol} Daily Percentage Change Over Time")
                plt.xlabel("Date")
                plt.ylabel("Percentage Change (%)")
                plt.legend()
                plt.grid(True)
                plt.show()
            except Exception as e:
                self._log_error(f"Error plotting percentage change for {symbol}: {str(e)}")

    def analyze_price_trend(self, data_dict, window_size=30):
        """Plots the closing price, rolling mean, and volatility (rolling std) over time for multiple symbols."""

        sns.set(style="whitegrid")  # Set Seaborn style for improved aesthetics

        for symbol, df in data_dict.items():
            try:
                if df is None or df.empty:
                    self._log_error(f"DataFrame for {symbol} is empty.")
                    continue

                if 'Close' not in df.columns:
                    self._log_error(f"'Close' column not found in DataFrame for {symbol}.")
                    continue

                # Calculate rolling mean and standard deviation (volatility)
                df['Rolling_Mean'] = df['Close'].rolling(window=window_size).mean()
                df['Rolling_Std'] = df['Close'].rolling(window=window_size).std()

                # Plotting each symbol in a separate figure for clarity
                plt.figure(figsize=(12, 6))
                
                # Closing price line
                sns.lineplot(data=df, x=df.index, y='Close', label=f'{symbol} Closing Price', color="blue", linestyle='solid')
                # Rolling mean line
                sns.lineplot(data=df, x=df.index, y='Rolling_Mean', label=f'{symbol} {window_size}-day Rolling Mean', color="orange", linestyle="--")
                # Rolling standard deviation (volatility) line
                sns.lineplot(data=df, x=df.index, y='Rolling_Std', label=f'{symbol} {window_size}-day Rolling Volatility', color="green", linestyle=":")

                # Titles and labels
                plt.title(f"Closing Price Trend, Rolling Mean and Volatility of {symbol} Over Time", fontsize=16)
                plt.xlabel("Date", fontsize=12)
                plt.ylabel("Value", fontsize=12)
                plt.legend(title="Symbols")
                plt.grid(True)

                # Optional y-axis ticks increment adjustment for readability
                y_max = int(plt.ylim()[1])
                plt.yticks(range(0, y_max + 50, 50))  # Adjusts y-ticks by 50 increments

                # Tight layout for clarity
                plt.tight_layout()
                plt.show()

            except Exception as e:
                self._log_error(f"Error plotting data for {symbol}: {str(e)}")

        

    def plot_unusual_daily_return(self, data_dict, threshold=2.5):
        """
        Calculates and plots daily returns with highlights on unusually high or low return days for each symbol.

        Parameters:
        - data_dict (dict): Dictionary with stock symbols as keys and their DataFrames as values.
        - threshold (float): Threshold (in terms of standard deviations) to define unusual returns.
        """
        sns.set(style="whitegrid")

        for symbol, df in data_dict.items():
            try:
                if df is None or df.empty:
                    print(f"DataFrame for {symbol} is empty.")
                    continue

                # Calculate daily returns
                df['Daily_Return'] = df['Close'].pct_change() * 100

                # Determine unusual returns using the standard deviation threshold
                mean_return = df['Daily_Return'].mean()
                std_dev = df['Daily_Return'].std()
                unusual_returns = df[(df['Daily_Return'] > mean_return + threshold * std_dev) |
                                     (df['Daily_Return'] < mean_return - threshold * std_dev)]

                # Plot daily returns
                plt.figure(figsize=(12, 6))
                sns.lineplot(x=df.index, y=df['Daily_Return'], label=f'{symbol} Daily Return', color='skyblue')

                # Highlight unusually high and low returns
                plt.scatter(unusual_returns.index, unusual_returns['Daily_Return'], color='red', 
                            label=f"Unusual Returns (±{threshold}σ)", s=50, marker='o')

                # Plot styling
                plt.title(f"Daily Returns with Unusual Days Highlighted - {symbol}", fontsize=16)
                plt.xlabel("Date", fontsize=12)
                plt.ylabel("Daily Return (%)", fontsize=12)
                plt.axhline(0, color='grey', linestyle='--')  # Optional: Horizontal line at 0%
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

            except Exception as e:
                print(f"Error plotting unusual daily returns for {symbol}: {str(e)}")
 

    
