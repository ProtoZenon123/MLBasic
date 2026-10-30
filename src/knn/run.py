"""K-nearest neighbors with vectorized distances and majority voting."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.tools.logger import configure_logger, logger
from src.tools.paths import data_file, result_file
from src.tools.timing import measure_time


class KNNScratch:
    def __init__(self, neighbors: int = 5):
        self.neighbors = neighbors

    def fit(self, feature_matrix: np.ndarray, target_values: np.ndarray):
        self.feature_matrix = feature_matrix
        self.target_values = target_values
        return self

    def predict(self, feature_matrix: np.ndarray) -> np.ndarray:
        if self.neighbors > len(self.feature_matrix):
            raise ValueError("neighbors exceeds training rows")
        distances = np.sum((feature_matrix[:, None, :] - self.feature_matrix[None, :, :]) ** 2, axis=2)
        nearest = np.argsort(distances, axis=1, kind="stable")[:, :self.neighbors]
        return np.array([np.bincount(row, minlength=int(self.target_values.max()) + 1).argmax()
                         for row in self.target_values[nearest]])


def main() -> None:
    configure_logger(result_file("knn"))
    logger.info("KNN: started")
    data = pd.read_csv(data_file("knn", "classification.csv"))
    features = data[[name for name in data if name != "target"]].to_numpy()
    targets = data["target"].to_numpy()
    train_features, test_features, train_targets, test_targets = train_test_split(
        features, targets, test_size=0.3, random_state=61, stratify=targets
    )
    logger.log("RESULT", "KNN results (k=5, test rows: {})", len(test_targets))
    for name, model in (("Manual", KNNScratch()), ("scikit-learn", KNeighborsClassifier(n_neighbors=5))):
        logger.info("KNN: fitting {}", name)
        model, fit_seconds = measure_time(model.fit, train_features, train_targets)
        predictions, predict_seconds = measure_time(model.predict, test_features)
        logger.log("RESULT", "{}: accuracy={:.8f}, fit={:.6f} s, predict={:.6f} s", name,
                   accuracy_score(test_targets, predictions), fit_seconds, predict_seconds)
        logger.log("RESULT", "{} confusion matrix:\n{}", name, confusion_matrix(test_targets, predictions))
    logger.info("KNN: completed")


if __name__ == "__main__":
    main()
