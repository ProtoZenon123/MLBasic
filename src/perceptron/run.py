"""Perceptron learning rule with an explicit update count."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.tools.logger import configure_logger, logger
from src.tools.paths import data_file, result_file


class PerceptronScratch:
    def __init__(self, max_epochs: int = 100):
        self.max_epochs = max_epochs

    def fit(self, feature_matrix: np.ndarray, target_values: np.ndarray):
        self.coef_ = np.zeros(feature_matrix.shape[1])
        self.intercept_ = 0.0
        self.updates_ = 0
        labels = 2 * target_values - 1
        for epoch in range(self.max_epochs):
            errors = 0
            for row, label in zip(feature_matrix, labels):
                if label * (row @ self.coef_ + self.intercept_) <= 0:
                    self.coef_ += label * row
                    self.intercept_ += label
                    self.updates_ += 1
                    errors += 1
            if errors == 0:
                break
        self.n_iter_ = epoch + 1
        return self

    def predict(self, feature_matrix: np.ndarray):
        return (feature_matrix @ self.coef_ + self.intercept_ >= 0).astype(int)


def main() -> None:
    configure_logger(result_file("perceptron"))
    logger.info("Perceptron: started")
    logger.log("RESULT", "Perceptron results")
    for case in ("separable", "noisy"):
        data = pd.read_csv(data_file("perceptron", f"{case}.csv"))
        features = data[[name for name in data if name != "target"]].to_numpy()
        targets = data["target"].to_numpy()
        train_features, test_features, train_targets, test_targets = train_test_split(
            features, targets, test_size=0.3, random_state=91, stratify=targets
        )
        logger.info("Perceptron: {}", case)
        logger.log("RESULT", "\nDataset: {}", case)
        for name, model in (("Manual", PerceptronScratch()),
                            ("scikit-learn", Perceptron(max_iter=100, tol=None, shuffle=False, eta0=1))):
            model.fit(train_features, train_targets)
            predictions = model.predict(test_features)
            logger.log("RESULT", "{}: test accuracy={:.8f}, epochs={}, updates={}", name,
                       accuracy_score(test_targets, predictions), model.n_iter_,
                       getattr(model, "updates_", "not exposed"))
            logger.log("RESULT", "{}: bias={:.8f}, weights={}", name,
                       float(np.ravel(model.intercept_)[0]), np.round(np.ravel(model.coef_), 8))
    logger.info("Perceptron: completed")


if __name__ == "__main__":
    main()
