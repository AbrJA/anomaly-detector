from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

def evaluate(y, y_pred):
    metrics = (
        f"MSE: {mean_squared_error(y, y_pred):.4f}\n"
        f"MAE: {mean_absolute_error(y, y_pred):.4f}\n"
        f"R2: {r2_score(y, y_pred):.4f}\n"
        f"Median AE: {median_absolute_error(y, y_pred):.4f}"
    )
    print(metrics)