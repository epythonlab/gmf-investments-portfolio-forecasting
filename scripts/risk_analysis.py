import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class RiskAnalysis:
    
    def __init__(self, data_dict, risk_free_rate=0.01, logger=None):
        """
        Initializes the RiskAnalysis class.

        Parameters:
        - data_dict (dict): Dictionary with stock symbols as keys and their DataFrames as values.
        - risk_free_rate (float): The risk-free rate for Sharpe Ratio calculation.
        - logger (logging.Logger): Optional logger for logging information and errors.
        """
        self.data_dict = data_dict
        self.risk_free_rate = risk_free_rate
        self.logger = logger

    def calculate_daily_return(self, df):
        """Calculate daily returns for the given DataFrame."""
        try:
            daily_returns = df['Close'].pct_change().dropna()
            self.logger.info("Calculated daily returns.")
            return daily_returns
        except KeyError:
            self.logger.error("The 'Close' column is missing in the data.")
            return pd.Series(dtype=float)

    def calculate_VaR(self, returns, confidence_level=0.95):
        """
        Calculate Value at Risk (VaR) at a specified confidence level.

        Parameters:
        - returns (pd.Series): Daily returns of the stock.
        - confidence_level (float): Confidence level for VaR calculation.

        Returns:
        - VaR (float): The Value at Risk for the specified confidence level.
        """
        try:
            VaR = np.percentile(returns, (1 - confidence_level) * 100)
            self.logger.info(f"Calculated VaR at {confidence_level * 100}% confidence: {VaR:.2%}")
            return VaR
        except ValueError as e:
            self.logger.error(f"Error calculating VaR: {e}")
            return np.nan

    def calculate_sharpe_ratio(self, returns):
        """
        Calculate the Sharpe Ratio for the given returns.

        Parameters:
        - returns (pd.Series): Daily returns of the stock.

        Returns:
        - Sharpe Ratio (float): The risk-adjusted return.
        """
        try:
            excess_returns = returns - (self.risk_free_rate / 252)
            sharpe_ratio = excess_returns.mean() / excess_returns.std()
            annualized_sharpe_ratio = sharpe_ratio * np.sqrt(252)
            self.logger.info(f"Calculated Sharpe Ratio: {annualized_sharpe_ratio:.2f}")
            return annualized_sharpe_ratio
        except ZeroDivisionError:
            self.logger.error("Standard deviation of returns is zero; cannot calculate Sharpe Ratio.")
            return np.nan
        except Exception as e:
            self.logger.error(f"Unexpected error in calculating Sharpe Ratio: {e}")
            return np.nan

    def analyze_risk_and_return(self, confidence_level=0.95):
        """Perform VaR and Sharpe Ratio calculations for each stock and plot results."""
        
        results = {}

        for symbol, df in self.data_dict.items():
            try:
                if df is None or df.empty:
                    self.logger.error(f"DataFrame for {symbol} is empty.")
                    continue

                # Calculate daily returns
                daily_returns = self.calculate_daily_return(df)

                # Calculate VaR
                VaR = self.calculate_VaR(daily_returns, confidence_level=confidence_level)
                self.logger.info(f"{symbol} VaR at {confidence_level * 100}% confidence: {VaR:.2%}")

                # Calculate Sharpe Ratio
                sharpe_ratio = self.calculate_sharpe_ratio(daily_returns)
                self.logger.info(f"{symbol} Sharpe Ratio: {sharpe_ratio:.2f}")

                # Store results
                results[symbol] = {"VaR": VaR, "Sharpe Ratio": sharpe_ratio}

                # Plot daily returns with VaR threshold
                try:
                    plt.figure(figsize=(12, 6))
                    plt.plot(daily_returns.index, daily_returns, label=f"{symbol} Daily Returns", color='skyblue')
                    plt.axhline(VaR, color='red', linestyle='--', label=f"VaR ({confidence_level * 100}%)")
                    plt.title(f"{symbol} Daily Returns with VaR Threshold")
                    plt.xlabel("Date")
                    plt.ylabel("Daily Return (%)")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    self.logger.error(f"Error plotting daily returns for {symbol}: {e}")

            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {str(e)}")

        return pd.DataFrame(results).T
