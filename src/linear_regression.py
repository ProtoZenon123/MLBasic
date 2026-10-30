"""Fit linear regression with the pseudoinverse using NumPy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tools.plotting import save_scatter_plot
from tools.timing import measure_time

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "generated_data"
PLOT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "plots"


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
    for dataset_name in ("clean", "outliers"):
        data = pd.read_csv(DATA_DIR / f"linear_regression_{dataset_name}.csv")
        feature_names = [name for name in data.columns if name not in ("target", "is_outlier")]
        feature_matrix = data[feature_names].to_numpy()
        target_values = data["target"].to_numpy()
        model, fit_seconds = measure_time(
            LinearRegressionScratch().fit, feature_matrix, target_values
        )
        predictions, predict_seconds = measure_time(model.predict, feature_matrix)
        mean_squared_error = np.mean((target_values - predictions) ** 2)
        plot_path = PLOT_DIR / f"linear_regression_{dataset_name}.png"
        save_scatter_plot(
            target_values,
            predictions,
            plot_path,
            x_label="Actual target",
            y_label="Predicted target",
            title=f"Linear regression ({dataset_name})",
            show_identity_line=True,
        )

        print(f"\n{dataset_name}:")
        print(f"  bias: {model.intercept_:.3f}")
        for feature_name, weight in zip(feature_names, model.coef_):
            print(f"  {feature_name}: {weight:.3f}")
        print(f"  training MSE: {mean_squared_error:.3f}")
        print(f"  fit time: {fit_seconds:.6f} s")
        print(f"  predict time: {predict_seconds:.6f} s")
        print(f"  plot: {plot_path}")


if __name__ == "__main__":
    main()
