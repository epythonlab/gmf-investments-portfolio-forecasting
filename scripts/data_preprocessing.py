import pynance as pn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
import os

class DataPreprocessor:
    """
    DataPreprocessor class for fetching, detecting, cleaning, and analyzing financial data from YFinance.
    """

    def __init__(self, data_dir="../data", logger=None):
        """
        Initializes the DataPreprocessor instance.

        Parameters:
        - symbols (list of str): List of stock symbols to fetch data for (e.g., ["TSLA", "BND", "SPY"]).
        - start_date (str): Start date for data fetching in 'YYYY-MM-DD' format.
        - end_date (str): End date for data fetching in 'YYYY-MM-DD' format.
        - data_dir (str): Directory to save downloaded data. Defaults to "data".
        - logger (logging.Logger): Optional logger for logging information and errors.
        """
        self.data_dir = data_dir
        self.logger = logger
        os.makedirs(self.data_dir, exist_ok=True)

    def get_data(self, start_date, end_date, symbols):
        """
        Fetches historical data for each symbol and saves it as a CSV.

        Returns:
        - dict: Dictionary with symbol names as keys and file paths of saved CSV files as values.
        """
        data_paths = {}
        
        for symbol in symbols:
            try:
                print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
                data = pn.data.get(symbol, start=start_date, end=end_date)
                # Save to CSV
                file_path = os.path.join(self.data_dir, f"{symbol}.csv")
                data.to_csv(file_path)
                data_paths[symbol] = file_path
                print(f"Data for {symbol} saved to '{file_path}'.")

            except ValueError as ve:
                error_message = f"Data format issue for {symbol}: {ve}"
                if self.logger:
                    self.logger.error(error_message)
                else:
                    print(error_message)

            except Exception as e:
                error_message = f"Failed to fetch data for {symbol}: {e}"
                if self.logger:
                    self.logger.error(error_message)
                else:
                    print(error_message)

        return data_paths
    
    def load_data(self, symbol):
        """
        Loads data from a CSV file for a specified symbol.

        Parameters:
        - symbol (str): Stock symbol to load data for (e.g., "TSLA").

        Returns:
        - pd.DataFrame: DataFrame with loaded data, or raises FileNotFoundError if missing.
        """
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")
        if os.path.exists(file_path):
            return pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
        else:
            error_message = f"Data file for symbol '{symbol}' not found. Run `get_data()` first."
            self._log_error(error_message)
            print("please check <a href='../logs/notebooks.log'>log</a>")
            raise FileNotFoundError(error_message)

    def inspect_data(self, data):
        """
        Inspects the data by checking data types, missing values, and duplicates.

        Parameters:
        - data (pd.DataFrame): DataFrame containing stock data for inspection.

        Returns:
        - dict: A dictionary containing the following inspection results:
          - Data types of the columns.
          - Missing values count.
          - Duplicate rows count.
        """
        inspection_results = {
            "data_types": data.dtypes,
            "missing_values": data.isnull().sum(),
            "duplicate_rows": data.duplicated().sum()
        }

        self._log_info(f"Data inspection results:\n{inspection_results}")
        return inspection_results

    def detect_outliers(self, data, method="iqr", z_threshold=3):
        """
        Detects outliers in the data using either the IQR or Z-score method.

        Parameters:
        - data (pd.DataFrame): DataFrame containing stock data.
        - method (str): Outlier detection method ('iqr' or 'z_score'). Default is 'iqr'.
        - z_threshold (int): Z-score threshold to classify an outlier. Default is 3 (only used if method is 'z_score').

        Returns:
        - pd.DataFrame: DataFrame containing boolean values indicating outliers.
        """
        outliers = pd.DataFrame(index=data.index)

        for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if col in data.columns:
                if method == "z_score":
                    z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                    outliers[col] = z_scores > z_threshold
                elif method == "iqr":
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers[col] = (data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))
        
        self._log_info("Outliers detected using {} method.".format(method.capitalize()))
        return outliers

    def plot_outliers(self, data, outliers, symbol):
        """
        Plots box plots to visualize outliers in the data.

        Parameters:
        - data (pd.DataFrame): DataFrame containing stock data.
        - outliers (pd.DataFrame): Boolean DataFrame indicating outliers.
        """
        columns_with_outliers = [col for col in data.columns if col in outliers.columns and outliers[col].any()]

        if not columns_with_outliers:
            self._log_info("No outliers detected in any columns.")
            return

        num_plots = len(columns_with_outliers)
        grid_size = math.ceil(math.sqrt(num_plots))  # Calculate grid dimensions

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12 * grid_size, 4 * grid_size))
        
        # Flatten axes to make sure we can iterate over it regardless of grid size
        if num_plots == 1:
            axes = [axes]
        else:
            axes = axes.ravel()

        for i, col in enumerate(columns_with_outliers):
            ax = axes[i]
            ax.plot(data.index, data[col], label=col, color="skyblue")  # Time series line
            ax.scatter(data.index[outliers[col]], data[col][outliers[col]], 
                       color='red', s=20, label="Outliers")  # Outliers as red dots

            ax.set_title(f"{col} - Time Series with Outliers of {symbol}")
            ax.set_xlabel("Date")
            ax.set_ylabel(col)
            ax.legend()

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

        # Confirm columns without outliers
        columns_without_outliers = [col for col in data.columns if col not in columns_with_outliers]
        if columns_without_outliers:
            self._log_info(f"No outliers detected in columns: {', '.join(columns_without_outliers)}")
 

    def handle_outliers(self, data_dict, outliers_dict):
        """
        Handles detected outliers by replacing them with NaN for later filling.

        Parameters:
        - data_dict (dict): Dictionary containing stock data as DataFrames for each symbol (e.g., {'TSLA': df_tsla, 'BND': df_bnd, 'SPY': df_spy}).
        - outliers_dict (dict): Dictionary containing boolean DataFrames indicating positions of outliers for each symbol.

        Returns:
        - dict: Dictionary with cleaned data for each symbol where outliers have been handled.
        """
        cleaned_data_dict = {}

        for symbol, data in data_dict.items():
            # Copy the data to avoid modifying the original
            cleaned_data = data.copy()
            
            # Check if outliers exist for the symbol and handle them
            if symbol in outliers_dict:
                outliers = outliers_dict[symbol]
                # Set outliers to NaN for interpolation
                cleaned_data[outliers] = np.nan
                
                # Interpolate NaN values and use forward/backward fill for any remaining
                cleaned_data.interpolate(method="time", inplace=True)
                cleaned_data.bfill(inplace=True)
                cleaned_data.ffill(inplace=True)

                print(f"Outliers handled for {symbol} by setting to NaN and filling with interpolation.")
            
            # Store the cleaned data
            cleaned_data_dict[symbol] = cleaned_data

        self._log_info("Outliers handled across all data sources.")
        return cleaned_data_dict


    def clean_data(self, data):
        """
        Cleans the loaded data by detecting and handling missing values and outliers.

        Parameters:
        - data (pd.DataFrame): DataFrame containing stock data to be cleaned.

        Returns:
        - pd.DataFrame: Cleaned DataFrame.
        """
        outliers = self.detect_outliers(data)
        data_cleaned = self.handle_outliers(data, outliers)
        return data_cleaned

    def normalize_data(self, data):
        """
        Normalizes the data columns (except 'Volume' and 'Date') using standard scaling.

        Parameters:
        - data (pd.DataFrame): DataFrame containing stock data to be normalized.

        Returns:
        - pd.DataFrame: DataFrame with normalized columns.
        """
        scaler = StandardScaler()
        columns_to_normalize = ["Open", "High", "Low", "Close", "Adj Close"]
        data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
        self._log_info("Data normalized using standard scaling.")
        return data

    def analyze_data(self, data):
        """
        Analyzes data by calculating basic statistics and checking for anomalies.

        Parameters:
        - data (pd.DataFrame): DataFrame containing stock data for analysis.

        Returns:
        - dict: Summary statistics including mean, median, standard deviation, and count of missing values.
        """
        analysis_results = {
            "mean": data.mean(),
            "median": data.median(),
            "std_dev": data.std(),
            "missing_values": data.isnull().sum()
        }
        self._log_info(f"Basic statistics calculated for data:\n{analysis_results}")
        return analysis_results



    def _log_info(self, message):
        """Logs informational messages."""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def _log_error(self, message):
        """Logs error messages."""
        if self.logger:
            self.logger.error(message)
        else:
            print(message)
