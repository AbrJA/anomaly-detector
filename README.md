# Anomaly Detector

This project provides a robust, production-ready solution for detecting anomalies in univariate time series data. It uses the NGBoost (Natural Gradient Boosting) library to model the data distribution and identify values that fall outside of a statistically significant range. The application is built with a modular structure, using a configuration file for easy parameter management and a clear folder layout for maintainability.

## Features

- Configurable Parameters: All model and file path parameters are managed via a config.json file, eliminating the need for long command-line arguments.

- Modular Design: The project is split into logical modules (main.py, anomaly_detector.py, utils.py) for improved readability, reusability, and easier testing.

- Model Persistence: The trained model can be saved and reloaded, allowing for efficient prediction on new data without re-training.

- Structured Logging: Uses Python's built-in logging module for clear, timestamped output and error handling.

## Prerequisites

Before you begin, ensure you have Python 3.8 or higher installed. The required libraries are listed in the requirements.txt file.

## Installation

1 Clone the repository:

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2 Create and activate a virtual environment (recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3 Install the dependencies:

```
    pip install -r requirements.txt
```

4 Project Structure

```
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   └── ngboost_model.joblib
├── src/
│   ├── anomaly_detector.py
│   └── utils.py
├── config.json
├── main.py
└── README.md
```

## Usage

Prepare your data: Place your training data (train.csv) and test data (test.csv) inside the data/ folder. Ensure both files have timestamp and value columns.

Configure the config.json file: Update the file paths and model parameters in config.json to match your needs.

    file_train: Path to the training data.

    file_test: Path to the test data.

    file_output: Path where the output CSV with anomalies will be saved.

    model_path: Path to save or load the trained model.

    alpha: Significance level (e.g., 0.05 for a 95% confidence interval).

    distribution: The statistical distribution for the NGBoost model (e.g., "normal", "lognormal").

Run the application: Execute the main script with the --config argument pointing to your configuration file.

```
python main.py --config config.json
```

## Output

The script will generate two main outputs:

    Console Logs: Detailed logs will be printed to the console, showing the training process, model metrics, and the anomaly detection summary.

    Output File: A new CSV file (as specified in file_output in config.json) will be created, containing the test data with a new anomaly column (True for anomalies, False otherwise).

## Model Management

The application is designed to be efficient for production use.

    Training and Saving: If the model_path specified in config.json does not exist, a new model will be trained on the data from file_train and saved to model_path.

    Loading and Predicting: If a model already exists at model_path, the script will automatically load it and skip the training phase, proceeding directly to anomaly detection on the test data.

## Contributing

Feel free to open issues or submit pull requests. All contributions are welcome
