"""Batch, stochastic, mini-batch, momentum, and Nesterov descent."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.tools.logger import configure_logger, logger
from src.tools.paths import data_file, result_file


def descend(feature_matrix: np.ndarray, target_values: np.ndarray, mode: str,
            epochs: int, learning_rate: float, seed: int = 78):
    """Minimize half MSE; the first design column holds the bias."""
    weights = np.zeros(feature_matrix.shape[1])
    velocity = np.zeros_like(weights)
    rng = np.random.default_rng(seed)
    batch_size = 1 if mode == "sgd" else 24 if mode == "mini_batch" else len(feature_matrix)
    for _ in range(epochs):
        if batch_size < len(feature_matrix):
            order = rng.permutation(len(feature_matrix))
        else:
            order = np.arange(len(feature_matrix))
        for indices in np.array_split(order, max(1, int(np.ceil(len(order) / batch_size)))):
            batch = feature_matrix[indices]
            targets = target_values[indices]
            lookahead = weights + 0.8 * velocity if mode == "nesterov" else weights
            gradient = batch.T @ (batch @ lookahead - targets) / len(indices)
            if mode in ("momentum", "nesterov"):
                velocity = 0.8 * velocity - learning_rate * gradient
                weights += velocity
            else:
                weights -= learning_rate * gradient
    return weights


def main() -> None:
    configure_logger(result_file("gradient_descent"))
    logger.info("Gradient descent: started")
    data = pd.read_csv(data_file("gradient_descent", "regression.csv"))
    features = data[[name for name in data if name != "target"]].to_numpy()
    targets = data["target"].to_numpy()
    train_features, test_features, train_targets, test_targets = train_test_split(
        features, targets, test_size=0.25, random_state=78
    )
    design_train = np.column_stack((np.ones(len(train_features)), train_features))
    design_test = np.column_stack((np.ones(len(test_features)), test_features))
    reference = LinearRegression().fit(train_features, train_targets)
    reference_mse = np.mean((reference.predict(test_features) - test_targets) ** 2)
    logger.log("RESULT", "Gradient descent; scikit-learn test MSE: {:.8f}", reference_mse)
    for mode, epochs, rate in (("batch", 250, 0.05), ("sgd", 35, 0.01),
                               ("mini_batch", 120, 0.04), ("momentum", 120, 0.04),
                               ("nesterov", 120, 0.04)):
        logger.info("Gradient descent: {}", mode)
        weights = descend(design_train, train_targets, mode, epochs, rate)
        train_loss = np.mean((design_train @ weights - train_targets) ** 2) / 2
        test_mse = np.mean((design_test @ weights - test_targets) ** 2)
        logger.log(
            "RESULT", "Manual {}: epochs={}, learning rate={:.4f}, train half-MSE={:.8f}, test MSE={:.8f}, weights={}",
            mode, epochs, rate, train_loss, test_mse, np.round(weights, 8),
        )
    quadratic = pd.read_csv(data_file("gradient_descent", "quadratic.csv"))
    curvature = quadratic["curvature"].to_numpy()
    initial = quadratic["initial"].to_numpy()
    for mode in ("batch", "momentum", "nesterov"):
        point = initial.copy()
        velocity = np.zeros_like(point)
        for _ in range(120):
            lookahead = point + 0.8 * velocity if mode == "nesterov" else point
            gradient = curvature * lookahead
            if mode == "batch":
                point -= 0.1 * gradient
            else:
                velocity = 0.8 * velocity - 0.1 * gradient
                point += velocity
        logger.log("RESULT", "Quadratic Manual {}: objective={:.8f}, point={}",
                   mode, 0.5 * np.sum(curvature * point ** 2), np.round(point, 8))
    logger.info("Gradient descent: completed")


if __name__ == "__main__":
    main()
