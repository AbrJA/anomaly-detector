import argparse
import logging
import os
import sys
import yaml
from src.anomaly_detector import AnomalyDetector

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Detect anomalies in factory sensor data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--file_config",
        type=str,
        required=True,
        help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    try:
        if not os.path.exists(args.file_config):
            raise FileNotFoundError(f"Configuration file not found: {args.file_config}")
        with open(args.file_config, "r") as f:
            config = yaml.safe_load(f)
        file_train = config.get("file_train")
        file_test = config.get("file_test")
        file_output = config.get("file_output")
        if not file_train or not file_test or not file_output:
            raise ValueError("Configuration file must specify 'file_train', 'file_test', and 'file_output'.")
        detector = AnomalyDetector(file_train, file_test, file_output)
        detector.train(dist = config.get("distribution", "normal"),
                       n_estimators=config.get("n_estimators", 500),
                       learning_rate=config.get("learning_rate", 0.01),
                       minibatch_frac=config.get("minibatch_frac", 1.0),
                       col_sample=config.get("col_sample", 1.0))
        detector.predict(config.get("alpha", 0.01))
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
