import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

"""Utility functions for evaluating model performance."""
def evaluate(y, y_pred, type: str = "Training"):
    metrics = (
        f"{type} metrics:\n"
        f"RMSE: {root_mean_squared_error(y, y_pred):.4f}\n"
        f"MSE: {mean_squared_error(y, y_pred):.4f}\n"
        f"MAE: {mean_absolute_error(y, y_pred):.4f}\n"
        f"R2: {r2_score(y, y_pred):.4f}"
    )
    logging.info(metrics)
