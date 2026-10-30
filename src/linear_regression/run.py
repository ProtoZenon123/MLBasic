"""Fit linear regression with the pseudoinverse using NumPy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.tools.logger import configure_logger, logger
from src.tools.timing import measure_time

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "generated_data" / "linear_regression"
OUTPUT_DIR = ROOT / "outputs" / "linear_regression"


class LinearRegressionScratch:
    def __init__(self) -> None:
        self.intercept_: float | None = None
        self.coef_: np.ndarray | None = None

    def fit(self, feature_matrix: np.ndarray, target_values: np.ndarray) -> LinearRegressionScratch:
        """Find the bias and feature weights by least squares."""
        design_matrix = np.column_stack((np.ones(len(feature_matrix)), feature_matrix))
        weights = np.linalg.pinv(design_matrix) @ target_values
        self.intercept_ = float(weights[0])
        self.coef_ = weights[1:]
        return self

    def predict(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Predict target values using the fitted weights."""
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Call fit before predict")
        return feature_matrix @ self.coef_ + self.intercept_


def main() -> None:
    configure_logger(OUTPUT_DIR / "results.txt")
    logger.info("Linear regression: started")
    logger.log("RESULT", "Linear regression results")

    for dataset_name in ("clean", "outliers"):
        logger.info("{}: loading data", dataset_name)
        data = pd.read_csv(DATA_DIR / f"linear_regression_{dataset_name}.csv")
        feature_names = [name for name in data.columns if name not in ("target", "is_outlier")]
        feature_matrix = data[feature_names].to_numpy()
        target_values = data["target"].to_numpy()

        logger.log("RESULT", "\nDataset: {}", dataset_name)
        for model_name, model in (
            ("Manual", LinearRegressionScratch()),
            ("scikit-learn", LinearRegression()),
        ):
            logger.info("{}: fitting {}", dataset_name, model_name)
            model, fit_seconds = measure_time(model.fit, feature_matrix, target_values)
            predictions, predict_seconds = measure_time(model.predict, feature_matrix)
            mean_squared_error = np.mean((target_values - predictions) ** 2)

            logger.log("RESULT", "{}", model_name)
            logger.log("RESULT", "  bias: {:.8f}", model.intercept_)
            for feature_name, weight in zip(feature_names, model.coef_):
                logger.log("RESULT", "  {}: {:.8f}", feature_name, weight)
            logger.log("RESULT", "  training MSE: {:.8f}", mean_squared_error)
            logger.log("RESULT", "  fit time: {:.6f} s", fit_seconds)
            logger.log("RESULT", "  predict time: {:.6f} s", predict_seconds)
        train_features, test_features, train_targets, test_targets = train_test_split(
            feature_matrix, target_values, test_size=0.25, random_state=7523
        )
        logger.log("RESULT", "  Holdout evaluation (separate fit on training rows)")
        for model_name, model in (("Manual", LinearRegressionScratch()),
                                  ("scikit-learn", LinearRegression())):
            model.fit(train_features, train_targets)
            holdout_mse = np.mean((test_targets - model.predict(test_features)) ** 2)
            logger.log("RESULT", "  {} holdout test MSE: {:.8f}", model_name, holdout_mse)
        logger.info("{}: completed", dataset_name)

    logger.info("Linear regression: completed; results saved to {}", OUTPUT_DIR)


if __name__ == "__main__":
    main()
