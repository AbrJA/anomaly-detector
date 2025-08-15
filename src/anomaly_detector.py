import os
import pandas as pd
import numpy as np
import logging
import pickle
from sklearn.model_selection import train_test_split
from ngboost import NGBRegressor
from ngboost.distns import Normal, LogNormal, Exponential
from scipy import stats
from src.utils import evaluate

class AnomalyDetector:
    """
    Anomalies in time-series data using a probabilistic regression model.
    """

    def __init__(self, train_file: str, test_file: str, pred_file: str):
        """
        Initializes the AnomalyDetector with file paths.

        Args:
            train_file: Path to the CSV file containing training data.
            test_file: Path to the CSV file containing test data.
            pred_file: Path to save the anomaly report as a CSV.
        """
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training file not found: {train_file}")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")

        self.train_file = train_file
        self.test_file = test_file
        self.pred_file = pred_file
        self.model = None
        logging.info("AnomalyDetector class initialized successfully.")

    def load(self, load_model_path: str = None):
        """
        Loads a pre-trained model from the specified path.

        Args:
            load_model_path: The file path to load the pre-trained model.
        """
        if load_model_path and os.path.exists(load_model_path):
            with open(load_model_path, "rb") as f:
                self.model = pickle.load(f)
            logging.info(f"Model loaded from '{load_model_path}'.")
        else:
            logging.warning("No pre-trained model found or specified.")

    def _load_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Loads and validates data, the function expects the CSV to have 'timestamp'
        and 'value' columns.

        Args:
            file_path: The path to the CSV data file.
        """
        try:
            df = pd.read_csv(file_path)
            if "timestamp" not in df.columns or "value" not in df.columns:
                raise ValueError("Missing required 'timestamp' or 'value' columns.")
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["minute"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
            logging.info(f"Data loaded from '{file_path}'. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Data file not found at '{file_path}'")
        except (pd.errors.ParserError, ValueError) as e:
            raise ValueError(f"Error parsing data from '{file_path}': {e}")

    def train(self, dist: str = "normal",
              n_estimators: int = 500,
              learning_rate: float = 0.01,
              minibatch_frac: float = 1.0,
              col_sample: float = 1.0):
        """
        Trains the NGBoost model using the data from the train_file.

        Args:
            dist: The distribution to use for NGBoost (e.g., "normal", "lognormal").
            n_estimators: The number of boosting iterations.
            learning_rate: The learning rate for the boosting algorithm.
            minibatch_frac: The fraction of data to use for each minibatch.
            col_sample: The fraction of columns to sample for each tree.
            save_path: (Optional) The file path to save the trained model.
        """
        logging.info("Trainig phase...")

        if self.model is None:
            distributions = {"normal": Normal, "lognormal": LogNormal, "exponential": Exponential}
            dist = distributions[dist]
            df_train = self._load_dataset(self.train_file)

            X = df_train.minute.to_numpy().reshape(-1, 1)
            y = df_train.value.to_numpy()

            self.model = NGBRegressor(
                Dist=dist,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                minibatch_frac=minibatch_frac,
                col_sample=col_sample
            )
            logging.info("Start training")
            self.model.fit(X, y)
            y_pred = self.model.predict(X)
            evaluate(y, y_pred)
        else:
            logging.info("Using pre-trained model.")

    def predict(self, alpha: float):
        """
        Detects anomalies in the test data using the trained model.
        Anomalies are identified as data points that fall outside the
        `1 - alpha` prediction interval of the model's distribution.

        Args:
            alpha: The significance level for the confidence interval.
        """
        logging.info("Predicting phase...")
        if self.model is None:
            raise RuntimeError("Model is not trained. Please call .train() first.")

        df_test = self._load_dataset(self.test_file)
        X_test = df_test.minute.to_numpy().reshape(-1, 1)
        y_test = df_test.value.to_numpy()
        logging.info("Start predicting")
        y_dist = self.model.pred_dist(X_test)
        quantile_lower = [stats.norm.ppf(q=alpha/2, **dist.params) for dist in y_dist]
        quantile_upper = [stats.norm.ppf(q=1-alpha/2, **dist.params) for dist in y_dist]
        anomaly = np.where((y_test >= quantile_lower) & (y_test <= quantile_upper), False, True)
        metrics = (
            f"Anomaly detection metrics.\n"
            f"Anomalies: {np.sum(anomaly):.2f}\n"
            f"Percentage: {(np.sum(anomaly) / len(anomaly)) * 100:.2f}"
        )
        logging.info(metrics)
        df_test["anomaly"] = anomaly
        self._save_dataset(df_test, self.pred_file)

    def save(self, save_model_path: str = None):
        """
        Saves the trained model to the specified path.

        Args:
            save_model_path: The file path to save the trained model.
        """
        if save_model_path is not None:
            with open(save_model_path, "wb") as f:
                pickle.dump(self.model, f)
            logging.info(f"Model saved to '{save_model_path}'")
        else:
            logging.warning("No save path specified. Model not saved.")

    def _save_dataset(self, df: pd.DataFrame, pred_file: str):
        """
        Saves a file to CSV format.

        Args:
            df: DataFrame to save.
            pred_file: The file path to save the DataFrame.
        """
        df.to_csv(pred_file, index=False)
        logging.info(f"Data saved to '{pred_file}'")