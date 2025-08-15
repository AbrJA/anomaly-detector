import argparse
import sys
import logging
from src.anomaly_detector import AnomalyDetector

logger = logging.getLogger(__name__)

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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
