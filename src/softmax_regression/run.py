"""Vectorized multinomial softmax trained with NumPy."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from src.tools.logger import configure_logger, logger
from src.tools.paths import data_file, result_file


class SoftmaxScratch:
    def fit(self, feature_matrix: np.ndarray, target_values: np.ndarray):
        design = np.column_stack((np.ones(len(feature_matrix)), feature_matrix))
        self.classes_ = np.unique(target_values)
        targets = (target_values[:, None] == self.classes_[None, :]).astype(float)
        weights = np.zeros((design.shape[1], len(self.classes_)))
        for _ in range(3500):
            scores = design @ weights
            scores -= scores.max(axis=1, keepdims=True)
            exponentials = np.exp(scores)
            probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
            weights -= 0.12 * design.T @ (probabilities - targets) / len(targets)
        self.intercept_ = weights[0]
        self.coef_ = weights[1:].T
        return self

    def predict_proba(self, feature_matrix: np.ndarray):
        scores = feature_matrix @ self.coef_.T + self.intercept_
        exponentials = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def predict(self, feature_matrix: np.ndarray):
        return self.classes_[self.predict_proba(feature_matrix).argmax(axis=1)]


def main() -> None:
    configure_logger(result_file("softmax_regression"))
    logger.info("Softmax regression: started")
    data = pd.read_csv(data_file("softmax_regression", "classes.csv"))
    features = data[[name for name in data if name != "target"]].to_numpy()
    targets = data["target"].to_numpy()
    train_features, test_features, train_targets, test_targets = train_test_split(
        features, targets, test_size=0.3, random_state=131, stratify=targets
    )
    logger.log("RESULT", "Softmax regression results")
    for name, model in (("Manual", SoftmaxScratch()), ("scikit-learn", LogisticRegression(C=np.inf, max_iter=3500))):
        logger.info("Softmax regression: fitting {}", name)
        model.fit(train_features, train_targets)
        probabilities = model.predict_proba(test_features)
        logger.log("RESULT", "{}: test cross-entropy={:.8f}, accuracy={:.8f}", name,
                   log_loss(test_targets, probabilities), accuracy_score(test_targets, model.predict(test_features)))
        logger.log("RESULT", "{} bias={}; weights={}", name, np.round(model.intercept_, 8), np.round(model.coef_, 8))
    logger.info("Softmax regression: completed")


if __name__ == "__main__":
    main()
