import pandas as pd
import numpy as np
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import save_model

class ModelTrainer:
    def __init__(self, data, logger=None, column='Close'):
        """
        Initialize the model trainer.

        Parameters:
        data (pd.DataFrame): Time series data.
        logger (logging.Logger): Logger instance for logging information.
        column (str): The column name to forecast.
        """
        self.df = data
        self.column = column
        self.logger = logger
        self.train = None
        self.test = None
        self.model = {}
        self.prediction = {}
        self.metric = {}
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Initialize the scaler

    def prepare_data(self, train_size=0.8):
        """
        Prepare the data for training and testing, and apply scaling.
        """
        try:
            self.df.index = pd.to_datetime(self.df.index)
            self.df = self.df.resample('D').ffill().dropna()

            split_idx = int(len(self.df) * train_size)
            self.train, self.test = self.df[:split_idx], self.df[split_idx:]
            self.logger.info(f"Data split: {len(self.train)} train, {len(self.test)} test")

            scaler = MinMaxScaler(feature_range=(0, 1))
            train_scaled = scaler.fit_transform(self.train[[self.column]])
            test_scaled = scaler.transform(self.test[[self.column]])

            self.train.loc[:, self.column] = train_scaled
            self.test.loc[:, self.column] = test_scaled
        except Exception as e:
            self.logger.error(f"Error in preparing data: {e}")
            raise ValueError("Data preparation failed")
        
    def train_arima(self):
        """Train ARIMA model using auto_arima."""
        try:
            self.logger.info("Training ARIMA model")
            model = pm.auto_arima(self.train[self.column], seasonal=False, trace=True, error_action='ignore',
                                  suppress_warnings=True, stepwise=True)
            self.model['ARIMA'] = model
            print(model.summary())
            self.logger.info(f"ARIMA model trained with parameters: {model.get_params()}")
        except Exception as e:
            self.logger.error(f"Error in ARIMA training: {e}")
            raise ValueError("ARIMA model training failed")

    def train_sarima(self, seasonal_period=5):
        """Train SARIMA model using auto_arima."""
        try:
            self.logger.info("Training SARIMA model")
            model = pm.auto_arima(self.train[self.column], seasonal=True, m=seasonal_period,
                                  start_p=0, start_q=0, max_p=3, max_q=3, d=1, D=1,
                                  trace=True, error_action='ignore', suppress_warnings=True)
            self.model['SARIMA'] = model
            print(model.summary())
            self.logger.info(f"SARIMA model trained with parameters: {model.get_params()}")
        except Exception as e:
            self.logger.error(f"Error in SARIMA training: {e}")
            raise ValueError("SARIMA model training failed")

       

    def _create_sequence(self, data, seq_length=60):
        """Create sequences of data for LSTM."""
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i + seq_length]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def train_lstm(self, seq_length=60, epochs=50, batch_size=32):
        """Train an LSTM model on the data."""
        try:
            self.logger.info("Training LSTM model")
            data = self.train[self.column].values.reshape(-1, 1)

            # Create sequences
            X_train, y_train = self._create_sequence(data, seq_length)
            
            model = Sequential()
            model.add(Input(shape=(seq_length, 1)))
            model.add(LSTM(50, activation='relu', return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(50, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(1))

            model.compile(optimizer='adam', loss='mse')
            model.summary()  # Print model summary
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            history = model.fit(
                X_train, y_train, epochs=epochs, batch_size=batch_size,
                validation_split=0.1, callbacks=[early_stopping], verbose=1
            )

            self.model['LSTM'] = {'model': model, 'history': history, 'seq_length': seq_length}
            self.logger.info("LSTM model training completed")

            # Plot training and validation loss
            self.plot_training_history(history)

        except Exception as e:
            self.logger.error(f"Error in LSTM training: {e}")
            raise ValueError("LSTM model training failed")

    def plot_training_history(self, history):
        """Plot the training and validation loss."""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            self.logger.info("Training history plotted successfully")
        except Exception as e:
            self.logger.error(f"Error in plotting training history: {e}")
            raise ValueError("Plotting training history failed")

    def make_prediction(self):
        """Generate predictions using all trained models."""
        try:
            for model_name, model_data in self.model.items():
                if model_name == 'ARIMA' or model_name == 'SARIMA':
                    self.prediction[model_name] = model_data.predict(n_periods=len(self.test))
                elif model_name == 'LSTM':
                    model = model_data['model']
                    seq_length = model_data['seq_length']
                    data = np.array(self.train[self.column].values[-seq_length:].reshape(-1, 1))
                    predictions = []
                    for i in range(len(self.test)):
                        # Predict and reshape for next iteration
                        pred = model.predict(data.reshape(1, seq_length, 1))
                        predictions.append(pred[0, 0])
                        data = np.append(data[1:], pred[0, 0]).reshape(-1, 1)
                    self.prediction['LSTM'] = np.array(predictions)
            self.logger.info("Predictions generated for all models")
        except Exception as e:
            self.logger.error(f"Error in making predictions: {e}")
            raise ValueError("Prediction generation failed")

    def evaluate_model(self):
        """Evaluate all models and log metrics."""
        try:
            metric_data = []
            for model_name, model in self.model.items():
                predictions = self.prediction.get(model_name)
                if predictions is None:
                    self.logger.error(f"No predictions for model {model_name}")
                    continue

                # Flatten the test data
                test_data = self.test[self.column].values
                mae = mean_absolute_error(test_data, predictions)
                rmse = np.sqrt(mean_squared_error(test_data, predictions))
                mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100

                self.metric[model_name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
                self.logger.info(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

                metric_data.append([model_name, mae, rmse, mape])

            # Display metrics in DataFrame
            metric_df = pd.DataFrame(metric_data, columns=["Model", "MAE", "RMSE", "MAPE"])
            print("\nModel Evaluation Metrics:\n", metric_df)
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {e}")
            raise ValueError("Model evaluation failed")

    def plot_result(self):
        """Plot the actual vs predicted results for all models."""
        try:
            plt.figure(figsize=(15, 8))
            plt.plot(self.test.index, self.test[self.column], label='Actual', linewidth=2)

            for model_name, predictions in self.prediction.items():
                plt.plot(self.test.index, predictions, label=f'{model_name} Prediction', linestyle='--')

            plt.title('Model Predictions Comparison')
            plt.xlabel('Date')
            plt.ylabel(self.column)
            plt.legend()
            plt.show()
            self.logger.info("Results plotted successfully")
        except Exception as e:
            self.logger.error(f"Error in plotting results: {e}")
            raise ValueError("Plotting results failed")
    
    def save_best_model(self, model_name='LSTM'):
        """Save the best model for future use."""
        try:
            if model_name in self.model:
                model_data = self.model[model_name]
                if model_name == 'LSTM':
                    # Save the LSTM model
                    model = model_data['model']
                    model.save(f'{model_name}_best_model.h5')
                    self.logger.info(f"{model_name} model saved successfully.")
                else:
                    # Save the ARIMA or SARIMA model using joblib
                    joblib.dump(model_data, f'{model_name}_best_model.pkl')
                    self.logger.info(f"{model_name} model saved successfully.")
            else:
                self.logger.error(f"{model_name} model not found for saving.")
        except Exception as e:
            self.logger.error(f"Error saving {model_name} model: {e}")
            raise ValueError(f"Model saving failed for {model_name}")