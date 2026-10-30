"""Binary logistic regression trained with NumPy gradient descent."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from src.tools.logger import configure_logger, logger
from src.tools.paths import data_file, result_file


class LogisticRegressionScratch:
    def fit(self, feature_matrix: np.ndarray, target_values: np.ndarray):
        design = np.column_stack((np.ones(len(feature_matrix)), feature_matrix))
        weights = np.zeros(design.shape[1])
        for _ in range(2500):
            probabilities = 1 / (1 + np.exp(-np.clip(design @ weights, -35, 35)))
            weights -= 0.2 * (design.T @ (probabilities - target_values)) / len(target_values)
        self.intercept_ = weights[0]
        self.coef_ = weights[1:]
        return self

    def predict_proba(self, feature_matrix: np.ndarray):
        logits = feature_matrix @ self.coef_ + self.intercept_
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -35, 35)))
        return np.column_stack((1 - probabilities, probabilities))

    def predict(self, feature_matrix: np.ndarray):
        return (self.predict_proba(feature_matrix)[:, 1] >= 0.5).astype(int)


def main() -> None:
    configure_logger(result_file("logistic_regression"))
    logger.info("Logistic regression: started")
    data = pd.read_csv(data_file("logistic_regression", "binary.csv"))
    features = data[[name for name in data if name != "target"]].to_numpy()
    targets = data["target"].to_numpy()
    train_features, test_features, train_targets, test_targets = train_test_split(
        features, targets, test_size=0.3, random_state=101, stratify=targets
    )
    logger.log("RESULT", "Logistic regression results")
    for name, model in (("Manual", LogisticRegressionScratch()),
                        ("scikit-learn", LogisticRegression(C=np.inf, max_iter=2500))):
        logger.info("Logistic regression: fitting {}", name)
        model.fit(train_features, train_targets)
        probabilities = model.predict_proba(test_features)
        predictions = model.predict(test_features)
        logger.log("RESULT", "{}: test log loss={:.8f}, accuracy={:.8f}, bias={:.8f}, weights={}",
                   name, log_loss(test_targets, probabilities), accuracy_score(test_targets, predictions),
                   float(np.ravel(model.intercept_)[0]), np.round(np.ravel(model.coef_), 8))
    logger.info("Logistic regression: completed")


if __name__ == "__main__":
    main()
