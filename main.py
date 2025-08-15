import argparse
import logging
import os
import sys
import yaml
from src.anomaly_detector import AnomalyDetector

def main():
    parser = argparse.ArgumentParser(
        description="Detect anomalies in factory sensor data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    try:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Configuration file not found: {args.config}")
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        train_file = config.get("train_file")
        test_file = config.get("test_file")
        pred_file = config.get("pred_file")
        if not train_file or not test_file or not pred_file:
            raise ValueError("Configuration file must specify 'train_file', 'test_file', and 'pred_file'.")
        detector = AnomalyDetector(train_file, test_file, pred_file)
        detector.load(load_model_path=config.get("load_model_path", None))
        detector.train(dist = config.get("distribution", "normal"),
                       n_estimators=config.get("n_estimators", 500),
                       learning_rate=config.get("learning_rate", 0.01),
                       minibatch_frac=config.get("minibatch_frac", 1.0),
                       col_sample=config.get("col_sample", 1.0))
        detector.predict(config.get("alpha", 0.01))
        detector.save(save_model_path=config.get("save_model_path", None))
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
