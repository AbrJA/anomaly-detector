# Anomaly Detector

This project provides a robust, production-ready solution for detecting anomalies in univariate time series data. It uses the **NGBoost (Natural Gradient Boosting)** library to model the data distribution and identify values that fall outside of a statistically significant range. The application is built with a modular structure, using a configuration file for easy parameter management and a clear folder layout for maintainability.

## Features

- **Configurable Parameters**: All model and file path parameters are managed via a config.json file, eliminating the need for long command-line arguments.

- **Modular Design**: The project is split into logical modules (main.py, anomaly_detector.py, utils.py) for improved readability, reusability, and easier testing.

- **Model Persistence**: The trained model can be saved and reloaded, allowing for efficient prediction on new data without re-training.

- **Structured Logging**: Uses Python's built-in logging module for clear, timestamped output and error handling.

### Project Structure

    ├── data/
    │   ├── sensor_data_train.csv
    │   └── sensor_data_test.csv
    ├── models/
    │   └── model.pickle
    ├── src/
    │   ├── anomaly_detector.py
    │   └── utils.py
    ├── config.json
    ├── main.py
    └── README.md

## Prerequisites

**Clone the repository**:

    git clone https://github.com/AbrJA/anomaly-detector.git
    cd your-repo-name

## Installation

### Local

Before you begin, ensure you have Python 3.12 installed. The required libraries are listed in the requirements.txt file.

**Create and activate a virtual environment (recommended)**:

    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

**Install the dependencies**:

    pip install -r requirements.txt

**Run the application**:

    python main.py --config config.json

### Docker

You can run the application in a containerized environment using Docker.

**Build the Docker image:**

    [sudo] docker build -t anomaly-detector .

**Run the container:**

    [sudo] docker run  -v "$(pwd)/cache":/detector --name anomaly-detector-cont anomaly-detector

*Note:* Make sure to update your `Dockerfile` to copy your training and test data files into the image.

## Usage

- **Prepare your data**: Place your training data and test data in some place, for instance, inside the datasets/ folder. Ensure both files have timestamp and value columns.

- **Configure the `config.json` file**: Update the file paths and model parameters in config.json to match your needs.

```
train_file: Path to the training data.
test_file: Path to the test data.
pred_file: Path where the output CSV with anomalies will be saved.
load_model_path: Path to load the trained model.
save_model_path: Path to save the trained model.
alpha: Significance level (e.g., 0.05 for a 95% confidence interval).
distribution: The statistical distribution for the NGBoost model (e.g., "normal").
n_estimators: Number of boosting rounds (trees).
learning_rate: Step size shrinkage used in update to prevent overfitting.
minibatch_frac: Fraction of data to use for each boosting iteration.
col_sample: Fraction of features to use for each boosting iteration.
```

## Output

The script will generate two main outputs:

- **Console Logs**: Detailed logs will be printed to the console, showing the training process, model metrics, and the anomaly detection summary.

- **Output File**: A new CSV file (as specified in pred_file in config.json) will be created, containing the test data with a new anomaly column (True for anomalies, False otherwise).

## Model Management

The application is designed to be efficient for production use.

- **Training and Saving**: If the `load_model_path` specified in `config.json` does not exist, a new model will be trained on the data from train_file and saved to save_model_path if it's specified.

- **Loading and Predicting**: If a model already exists at `load_model_path`, the script will automatically load it and skip the training phase, proceeding directly to anomaly detection on the test data.

## Contributing

Feel free to open issues or submit pull requests. All contributions are welcome
