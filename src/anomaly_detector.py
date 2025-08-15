import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from ngboost import NGBRegressor
from ngboost.distns import Normal, LogNormal, Exponential
from scipy import stats
from src.utils import evaluate

class AnomalyDetector:
    """
    Anomalies in time-series data using a probabilistic regression model.
    """

    def __init__(self, file_train: str, file_test: str, file_output: str):
        """
        Initializes the AnomalyDetector with file paths.

        Args:
            file_train: Path to the CSV file containing training data.
            file_test: Path to the CSV file containing test data.
            file_output: Path to save the anomaly report as a CSV.
        """
        if not os.path.exists(file_train):
            raise FileNotFoundError(f"Training file not found: {file_train}")
        if not os.path.exists(file_test):
            raise FileNotFoundError(f"Test file not found: {file_test}")

        self.file_train = file_train
        self.file_test = file_test
        self.file_output = file_output
        self.model = None
        logging.info("AnomalyDetector class initialized successfully.")

    def _load(self, file_path: str) -> pd.DataFrame:
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

    def train(self, dist: str, n_estimators: int, learning_rate: float,
              minibatch_frac: float, col_sample: float):
        """
        Trains the NGBoost model using the data from the file_train.

        Args:
            dist: The distribution to use for NGBoost (e.g., "normal", "lognormal").
            n_estimators: The number of boosting iterations.
            learning_rate: The learning rate for the boosting algorithm.
            minibatch_frac: The fraction of data to use for each minibatch.
            col_sample: The fraction of columns to sample for each tree.
            save_path: (Optional) The file path to save the trained model.
        """
        logging.info("Trainig phase...")
        distributions = {"normal": Normal, "lognormal": LogNormal, "exponential": Exponential}
        dist = distributions[dist]
        df_train = self._load(self.file_train)

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
        logging.info("Training metrics")
        evaluate(y, y_pred)

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

        df_test = self._load(self.file_test)
        X_test = df_test.minute.to_numpy().reshape(-1, 1)
        y_test = df_test.value.to_numpy()
        logging.info("Start predicting")
        y_dist = self.model.pred_dist(X_test)
        quantile_lower = [stats.norm.ppf(q=alpha/2, **dist.params) for dist in y_dist]
        quantile_upper = [stats.norm.ppf(q=1-alpha/2, **dist.params) for dist in y_dist]
        anomaly = np.where((y_test >= quantile_lower) & (y_test <= quantile_upper), False, True)
        metrics = (
            f"Anomalies: {np.sum(anomaly):.4f}\n"
            f"Percentage: {(np.sum(anomaly) / len(anomaly)) * 100:.4f}"
        )
        print(metrics)
        df_test["anomaly"] = anomaly
        self._save(df_test, self.file_output)

    def _save(self, df: pd.DataFrame, file_output: str):
        """
        Saves a file to CSV format.

        Args:
            df: DataFrame to save.
            file_output: The file path to save the DataFrame.
        """
        df.to_csv(file_output, index=False)
        logging.info(f"Data saved to '{file_output}'")