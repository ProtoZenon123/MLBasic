"""NumPy Fisher projection and scikit-learn LDA on the same split."""

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.tools.logger import configure_logger, logger
from src.tools.paths import data_file, result_file


class LDAScratch:
    def fit(self, feature_matrix: np.ndarray, target_values: np.ndarray):
        self.classes_ = np.unique(target_values)
        overall_mean = feature_matrix.mean(axis=0)
        within = np.zeros((feature_matrix.shape[1], feature_matrix.shape[1]))
        between = np.zeros_like(within)
        for label in self.classes_:
            rows = feature_matrix[target_values == label]
            mean = rows.mean(axis=0)
            centered = rows - mean
            within += centered.T @ centered
            difference = mean - overall_mean
            between += len(rows) * np.outer(difference, difference)
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(within) @ between)
        order = np.argsort(eigenvalues.real)[::-1][:len(self.classes_) - 1]
        self.components_ = eigenvectors[:, order].real
        projected = feature_matrix @ self.components_
        self.centroids_ = np.array([projected[target_values == label].mean(axis=0) for label in self.classes_])
        self.fisher_ratio_ = float(eigenvalues.real[order].sum())
        return self

    def transform(self, feature_matrix: np.ndarray):
        return feature_matrix @ self.components_

    def predict(self, feature_matrix: np.ndarray):
        projected = self.transform(feature_matrix)
        distances = np.sum((projected[:, None, :] - self.centroids_[None, :, :]) ** 2, axis=2)
        return self.classes_[distances.argmin(axis=1)]


def main() -> None:
    configure_logger(result_file("lda"))
    logger.info("LDA: started")
    data = pd.read_csv(data_file("lda", "classes.csv"))
    features = data[[name for name in data if name != "target"]].to_numpy()
    targets = data["target"].to_numpy()
    train_features, test_features, train_targets, test_targets = train_test_split(
        features, targets, test_size=0.3, random_state=291, stratify=targets
    )
    logger.log("RESULT", "LDA results")
    for name, model in (("Manual", LDAScratch()),
                        ("scikit-learn", LinearDiscriminantAnalysis(n_components=2))):
        logger.info("LDA: fitting {}", name)
        model.fit(train_features, train_targets)
        logger.log("RESULT", "{}: projected shape={}, test accuracy={:.8f}", name,
                   model.transform(test_features).shape, accuracy_score(test_targets, model.predict(test_features)))
        if name == "Manual":
            logger.log("RESULT", "Manual Fisher eigenvalue sum={:.8f}", model.fisher_ratio_)
    logger.info("LDA: completed")


if __name__ == "__main__":
    main()
