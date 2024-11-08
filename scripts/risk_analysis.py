import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class RiskAnalysis:
    
    def __init__(self, data_dict, risk_free_rate=0.01):
        """
        Initializes the RiskAnalysis class.

        Parameters:
        - data_dict (dict): Dictionary with stock symbols as keys and their DataFrames as values.
        - risk_free_rate (float): The risk-free rate for Sharpe Ratio calculation.
        """
        self.data_dict = data_dict
        self.risk_free_rate = risk_free_rate

    def calculate_daily_returns(self, df):
        """Calculate daily returns for the given DataFrame."""
        return df['Close'].pct_change().dropna()

    def calculate_VaR(self, returns, confidence_level=0.95):
        """
        Calculate Value at Risk (VaR) at a specified confidence level.

        Parameters:
        - returns (pd.Series): Daily returns of the stock.
        - confidence_level (float): Confidence level for VaR calculation.

        Returns:
        - VaR (float): The Value at Risk for the specified confidence level.
        """
        # Calculate the quantile based on the confidence level
        VaR = np.percentile(returns, (1 - confidence_level) * 100)
        return VaR

    def calculate_sharpe_ratio(self, returns):
        """
        Calculate the Sharpe Ratio for the given returns.

        Parameters:
        - returns (pd.Series): Daily returns of the stock.

        Returns:
        - Sharpe Ratio (float): The risk-adjusted return.
        """
        # Calculate excess returns
        excess_returns = returns - (self.risk_free_rate / 252)  # Adjust risk-free rate for daily
        # Sharpe Ratio calculation
        sharpe_ratio = excess_returns.mean() / excess_returns.std()
        return sharpe_ratio * np.sqrt(252)  # Annualize Sharpe Ratio

    def analyze_risk_and_return(self, confidence_level=0.95):
        """Perform VaR and Sharpe Ratio calculations for each stock and plot results."""
        
        results = {}

        for symbol, df in self.data_dict.items():
            try:
                if df is None or df.empty:
                    print(f"DataFrame for {symbol} is empty.")
                    continue
                
                # Calculate daily returns
                daily_returns = self.calculate_daily_returns(df)

                # Calculate VaR
                VaR = self.calculate_VaR(daily_returns, confidence_level=confidence_level)
                print(f"{symbol} VaR at {confidence_level*100}% confidence: {VaR:.2%}")

                # Calculate Sharpe Ratio
                sharpe_ratio = self.calculate_sharpe_ratio(daily_returns)
                print(f"{symbol} Sharpe Ratio: {sharpe_ratio:.2f}")

                # Store results
                results[symbol] = {"VaR": VaR, "Sharpe Ratio": sharpe_ratio}

                # Plot daily returns with VaR threshold
                plt.figure(figsize=(12, 6))
                plt.plot(daily_returns.index, daily_returns, label=f"{symbol} Daily Returns", color='skyblue')
                plt.axhline(VaR, color='red', linestyle='--', label=f"VaR ({confidence_level*100}%)")
                plt.title(f"{symbol} Daily Returns with VaR Threshold")
                plt.xlabel("Date")
                plt.ylabel("Daily Return (%)")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

            except Exception as e:
                print(f"Error analyzing {symbol}: {str(e)}")

        return pd.DataFrame(results).T


