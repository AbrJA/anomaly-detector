import pandas as pd
import numpy as np
import argparse
import os
import sys
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from ngboost import NGBRegressor
from ngboost.distns import Normal, LogNormal, Exponential
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

distributions = {
    "normal": Normal,
    "lognormal": LogNormal,
    "exponential": Exponential
}

class AnomalyDetector:
    """
    This class encapsulates the logic for training a model on univariate
    time serie data and test it in a new dataset.
    """

    def __init__(self, file_train: str, file_test: str, file_output: str):
        """
        Initializes the AnomalyDetector with configuration parameters.

        Args:

        """
        if not os.path.exists(file_train):
            raise FileNotFoundError(f"Training file not found: {file_train}")
        if not os.path.exists(file_test):
            raise FileNotFoundError(f"Test file not found: {file_test}")

        self.file_train = file_train
        self.file_test = file_test
        self.file_output = file_output
        self.model = None
        logger.info("AnomalyDetector class initialized successfully.")

    def _load(self, file_path: str) -> pd.DataFrame:
        """Loads and validates sensor data from a CSV file."""
        try:
            df = pd.read_csv(file_path)
            if "timestamp" not in df.columns or "value" not in df.columns:
                raise ValueError("Missing required 'timestamp' or 'value' columns.")
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["minute"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
            logger.info(f"Data loaded from '{file_path}'. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Data file not found at '{file_path}'")
        except (pd.errors.ParserError, ValueError) as e:
            raise ValueError(f"Error parsing data from '{file_path}': {e}")

    def train(self, dist: str, n_estimators: int, learning_rate: float,
              minibatch_frac: float, col_sample: float):
        """
        Trains the detector by establishing a baseline from training data.

        Args:
            file_train_path: Path to the CSV file with training data.
        """
        logger.info("Trainig phase...")
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
        logger.info("Start training")
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        logger.info("Training metrics")
        metrics = (
            f"MSE: {mean_squared_error(y, y_pred):.4f}\n"
            f"MAE: {mean_absolute_error(y, y_pred):.4f}\n"
            f"R2: {r2_score(y, y_pred):.4f}\n"
            f"Median AE: {median_absolute_error(y, y_pred):.4f}"
        )
        print(metrics)

    def predict(self, alpha: float):
        """
        Detects anomalies in the test data using the established baseline.

        Args:
            test_data_path: Path to the CSV file with test data.

        Returns:
            A DataFrame containing the detected anomalies.
        """
        logger.info("Predicting phase...")
        if self.model is None:
            raise RuntimeError("Model is not trained. Please call .train() first.")

        df_test = self._load(self.file_test)
        X_test = df_test.minute.to_numpy().reshape(-1, 1)
        y_test = df_test.value.to_numpy()
        logger.info("Start predicting")
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
        Saves the detected anomalies to a CSV report.
        """
        df.to_csv(file_output, index=False)
        logger.info(f"Data saved to '{file_output}'")

def main():
    parser = argparse.ArgumentParser(
        description="Detect anomalies in factory sensor data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--file_train",
        type=str,
        required=True,
        help="Path to the training data CSV file."
    )
    parser.add_argument(
        "--file_test",
        type=str,
        required=True,
        help="Path to the test data CSV file."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for anomaly detection (default: 0.05)."
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=500,
        help="Number of boosting iterations (default: 500)."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for boosting (default: 0.01)."
    )
    parser.add_argument(
        "--minibatch_frac",
        type=float,
        default=1.0,
        help="Fraction of data to use for each minibatch (default: 1.0)."
    )
    parser.add_argument(
        "--col_sample",
        type=float,
        default=1.0,
        help="Fraction of columns to sample for each tree (default: 1.0)."
    )
    parser.add_argument(
        "--distribution",
        type=str,
        choices=["normal", "lognormal", "exponential"],
        default="normal",
        help="Distribution to use for NGBoost (default: normal)."
    )
    parser.add_argument(
        "--file_output",
        type=str,
        default="anomalies.csv",
        help="Output file path for the anomaly report (default: anomalies.csv)."
    )

    args = parser.parse_args()

    try:
        detector = AnomalyDetector(args.file_train, args.file_test, args.file_output)
        detector.train(dist = args.distribution,
                       n_estimators=args.n_estimators,
                       learning_rate=args.learning_rate,
                       minibatch_frac=args.minibatch_frac,
                       col_sample=args.col_sample)
        detector.predict(args.alpha)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
