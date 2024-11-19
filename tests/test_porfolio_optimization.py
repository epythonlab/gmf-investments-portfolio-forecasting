import unittest
import pandas as pd
import numpy as np
import os
from scripts.portfolio_optimization import PortfolioOptimizer
import logging

class TestPortfolioOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock data for testing
        cls.tsla_data = pd.DataFrame({"forecast": np.random.normal(300, 20, 1000)})
        cls.bnd_data = pd.DataFrame({"forecast": np.random.normal(80, 5, 1000)})
        cls.spy_data = pd.DataFrame({"forecast": np.random.normal(400, 15, 1000)})
        
        # Save to temporary CSV files
        cls.tsla_csv = "test_tsla.csv"
        cls.bnd_csv = "test_bnd.csv"
        cls.spy_csv = "test_spy.csv"
        cls.tsla_data.to_csv(cls.tsla_csv)
        cls.bnd_data.to_csv(cls.bnd_csv)
        cls.spy_data.to_csv(cls.spy_csv)

        # Create logger
        cls.logger = logging.getLogger("PortfolioOptimizerTest")
        logging.basicConfig(level=logging.INFO)

        # Initialize PortfolioOptimizer instance
        cls.optimizer = PortfolioOptimizer(
            tsla_csv=cls.tsla_csv,
            bnd_csv=cls.bnd_csv,
            spy_csv=cls.spy_csv,
            logger=cls.logger
        )

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary files
        os.remove(cls.tsla_csv)
        os.remove(cls.bnd_csv)
        os.remove(cls.spy_csv)

    def test_data_loading(self):
        """Test that data loads correctly and contains all columns."""
        self.assertEqual(list(self.optimizer.df.columns), ['TSLA', 'BND', 'SPY'])
        self.assertEqual(len(self.optimizer.df), 1000)

    def test_annualized_returns(self):
        """Test calculation of annualized returns."""
        returns = self.optimizer.calculate_annual_returns()
        self.assertEqual(len(returns), 3)  # Should return for TSLA, BND, SPY
        self.assertTrue(all(returns >= -1))  # Returns should not be absurdly low

    def test_portfolio_statistics(self):
        """Test portfolio statistics with equal weights."""
        weights = np.array([1/3, 1/3, 1/3])
        stats = self.optimizer.portfolio_statistics(weights)
        self.assertEqual(len(stats), 3)  # Expect (return, risk, sharpe_ratio)
        self.assertTrue(stats[1] > 0)  # Risk (volatility) must be positive
        self.assertTrue(stats[2] > 0)  # Sharpe Ratio should be positive with mock data

    def test_optimize_portfolio(self):
        """Test portfolio optimization."""
        result = self.optimizer.optimize_portfolio()
        self.assertIn("weights", result)
        self.assertIn("return", result)
        self.assertIn("risk", result)
        self.assertIn("sharpe_ratio", result)
        self.assertAlmostEqual(sum(result["weights"]), 1, places=6)  # Weights should sum to 1

    def test_risk_metrics(self):
        """Test risk metrics calculation."""
        metrics = self.optimizer.risk_metrics()
        self.assertIn("volatility", metrics)
        self.assertIn("VaR_95", metrics)
        self.assertTrue(metrics["volatility"] > 0)  # Volatility must be positive

if __name__ == "__main__":
    unittest.main()
