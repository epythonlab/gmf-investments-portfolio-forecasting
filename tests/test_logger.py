import os
import unittest
import logging
from scripts.logger import SetupLogger  # Adjust the import path as necessary

class TestSetupLogger(unittest.TestCase):
    def setUp(self):
        # Define a test log file path
        self.log_file = 'logs/test_app.log'
        self.log_level = logging.DEBUG  # Test with DEBUG level
        # Remove the log file if it exists from a previous test
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def tearDown(self):
        # Clean up the created log file
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def test_logger_creation(self):
        # Initialize the logger
        logger_setup = SetupLogger(log_file=self.log_file, log_level=self.log_level)
        logger = logger_setup.get_logger()

        # Check if the logger is properly configured
        self.assertEqual(logger.level, self.log_level)
        self.assertTrue(any(isinstance(handler, logging.FileHandler) for handler in logger.handlers))

    def test_log_file_creation(self):
        # Initialize and use the logger
        logger_setup = SetupLogger(log_file=self.log_file, log_level=self.log_level)
        logger = logger_setup.get_logger()
        test_message = "This is a test log entry."
        logger.info(test_message)

        # Verify the log file is created and contains the log entry
        self.assertTrue(os.path.exists(self.log_file))
        with open(self.log_file, 'r') as f:
            log_content = f.read()
        self.assertIn(test_message, log_content)

if __name__ == '__main__':
    unittest.main()
