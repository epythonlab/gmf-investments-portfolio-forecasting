
# GMF Investments Portfolio Forecasting

## Overview
This project leverages time series forecasting models to enhance portfolio management strategies for **Guide Me in Finance (GMF) Investments**. Using historical data for key financial assets—Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY)—we aim to forecast market trends, optimize asset allocation, and manage risk. By incorporating ARIMA, SARIMA, and LSTM models, this project enables GMF Investments to offer clients data-driven investment recommendations.

## Project Goals
- **Market Trend Forecasting**: Predict future market trends for TSLA, BND, and SPY to anticipate asset performance.
- **Portfolio Optimization**: Use forecasted insights to optimize portfolio allocation, balancing high returns with risk management.
- **Risk Management**: Adjust portfolio strategy based on predicted volatility and market trends to minimize client exposure to risks.

## Data Sources
Data is sourced using the [YFinance](https://pypi.org/project/yfinance/) library, covering:
- **Tesla (TSLA)**: High-growth, high-volatility stock in the automotive sector.
- **Vanguard Total Bond Market ETF (BND)**: A bond ETF for stability and income.
- **S&P 500 ETF (SPY)**: An ETF representing U.S. market exposure.

Historical data includes Open, High, Low, Close, Volume, and Adjusted Close prices from **January 1, 2015, to December 31, 2024**.

## Technologies Used
- **Programming Language**: Python
- **Data Collection**: YFinance API
- **Time Series Models**: ARIMA, SARIMA, LSTM (Long Short-Term Memory)
- **Libraries**: `pandas`, `numpy`, `statsmodels`, `tensorflow`, `scikit-learn`

## Project Structure
- **/.github**: Contains GitHub Actions workflows for building, testing, and deploying the CI/CD pipeline

- **/notebooks**: Includes Jupyter notebooks for exploratory data analysis, model development, and portfolio optimization. Used primarily for prototyping and visualization.

- **/scripts**: Contains modularized Python scripts for core functionalities, such as data processing, model training, data analysis, seasonal analysis, risk analysis, feature forecasting, and portfolio optimization.

- **tests/**: Contains unit and integration tests for validating the functionality and robustness of modularized code before deployment.

- **requirements.txt**: Includes Jupyter notebooks for exploratory data analysis, model development, and portfolio optimization. Used primarily for prototyping and visualization.


## Getting Started

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/epythonlab/gmf-investments-portfolio-forecasting.git
   cd gmf-investments-portfolio-forecasting
   ```

2. **Install dependencies**:
   Ensure you have Python 3.7+ installed, then install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Data Collection
Use the `YFinance` library to collect historical financial data. To fetch data for each asset, use the following example for **Tesla (TSLA)**:

```python
import yfinance as yf

# Download data for Tesla
tsla_data = yf.download("TSLA", start="2015-01-01", end="2024-12-31")
```

### Usage

1. **Run Jupyter Notebooks**:
   - **Data Exploration**: Analyze trends and preprocess data.
   - **Modeling Notebooks**: ARIMA, SARIMA, and LSTM models for forecasting asset prices.
   - **Portfolio Optimization**: Adjust portfolio allocation based on forecasting results.

2. **Portfolio Optimization**:
   - Utilize forecasted trends to balance the portfolio with high-growth, stable, and diversified assets.
   - Visualize asset allocation to see how each asset contributes to the portfolio's risk-return profile.

## Methodology

### Data Preprocessing
- Clean and preprocess data by removing missing values and handling anomalies.
- Analyze volatility and trends for each asset to inform model choice and calibration.

### Model Development
- **ARIMA & SARIMA**: Traditional time series models for initial trend analysis and baseline comparison.
- **LSTM**: Deep learning model for capturing non-linear patterns and dependencies, particularly effective for high-volatility assets.

### Evaluation Metrics
- **Root Mean Square Error (RMSE)**
- **Mean Absolute Error (MAE)**

Use these metrics to assess and compare model performance, refining them based on results.

## Contributing
You are welcome to contributions to enhance this project! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and submission guidelines.

### Steps to Contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Create a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
