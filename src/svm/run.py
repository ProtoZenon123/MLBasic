"""SVM progression: QP, kernel SVC, and vectorized multiclass hinge loss."""

import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC

from src.tools.logger import configure_logger, logger
from src.tools.paths import data_file, result_file


class DualLinearSVM:
    """Explicit SVM dual passed to the CVXOPT QP solver."""

    def __init__(self, penalty: float | None):
        self.penalty = penalty

    def fit(self, feature_matrix: np.ndarray, target_values: np.ndarray):
        labels = 2 * target_values - 1
        count = len(labels)
        gram = (feature_matrix @ feature_matrix.T) * np.outer(labels, labels)
        inequalities = -np.eye(count) if self.penalty is None else np.vstack((-np.eye(count), np.eye(count)))
        bounds = np.zeros(count) if self.penalty is None else np.r_[np.zeros(count), np.full(count, self.penalty)]
        solvers.options["show_progress"] = False
        solution = solvers.qp(
            matrix(gram + 1e-9 * np.eye(count)), matrix(-np.ones(count)),
            matrix(inequalities), matrix(bounds),
            matrix(labels.astype(float), (1, count)), matrix(0.0),
        )
        if solution["status"] != "optimal":
            raise RuntimeError(f"CVXOPT QP status: {solution['status']}")
        multipliers = np.array(solution["x"]).ravel()
        self.coef_ = (multipliers * labels) @ feature_matrix
        margin = multipliers > 1e-5
        if self.penalty is not None:
            margin &= multipliers < self.penalty - 1e-5
        if not margin.any():
            margin = multipliers > 1e-5
        self.intercept_ = np.mean(labels[margin] - feature_matrix[margin] @ self.coef_)
        self.support_count_ = int(np.sum(multipliers > 1e-5))
        return self

    def predict(self, feature_matrix: np.ndarray):
        return (feature_matrix @ self.coef_ + self.intercept_ >= 0).astype(int)


class MulticlassHingeScratch:
    def fit(self, feature_matrix: np.ndarray, target_values: np.ndarray):
        design = np.column_stack((np.ones(len(feature_matrix)), feature_matrix))
        self.classes_ = np.unique(target_values)
        class_indices = np.searchsorted(self.classes_, target_values)
        weights = np.zeros((design.shape[1], len(self.classes_)))
        for _ in range(1500):
            scores = design @ weights
            margins = np.maximum(0, scores - scores[np.arange(len(scores)), class_indices, None] + 1)
            margins[np.arange(len(scores)), class_indices] = 0
            active = (margins > 0).astype(float)
            active[np.arange(len(scores)), class_indices] = -active.sum(axis=1)
            gradient = design.T @ active / len(design)
            gradient[1:] += 0.01 * weights[1:]
            weights -= 0.05 * gradient
        self.intercept_ = weights[0]
        self.coef_ = weights[1:].T
        return self

    def predict(self, feature_matrix: np.ndarray):
        return self.classes_[(feature_matrix @ self.coef_.T + self.intercept_).argmax(axis=1)]


def main() -> None:
    configure_logger(result_file("svm"))
    logger.info("SVM: started")
    logger.log("RESULT", "SVM results")
    for case in ("hard", "soft", "kernel", "multiclass"):
        logger.info("SVM: {}", case)
        data = pd.read_csv(data_file("svm", f"{case}.csv"))
        features = data[[name for name in data if name != "target"]].to_numpy()
        targets = data["target"].to_numpy()
        train_features, test_features, train_targets, test_targets = train_test_split(
            features, targets, test_size=0.3, random_state=191, stratify=targets
        )
        if case in ("hard", "soft"):
            penalty = None if case == "hard" else 1.0
            models = (("CVXOPT QP", DualLinearSVM(penalty)),
                      ("scikit-learn SVC", SVC(kernel="linear", C=1e6 if penalty is None else penalty)))
        elif case == "kernel":
            models = (("scikit-learn RBF SVC", SVC(kernel="rbf", C=10.0, gamma="scale")),)
        else:
            models = (("Manual", MulticlassHingeScratch()),
                      ("scikit-learn LinearSVC", LinearSVC(C=1.0, max_iter=5000)))
        setting = {"hard": "hard margin", "soft": "C=1.0", "kernel": "C=10.0, RBF",
                   "multiclass": "linear hinge"}[case]
        logger.log("RESULT", "\nDataset: {}; {}", case, setting)
        for name, model in models:
            model.fit(train_features, train_targets)
            accuracy = accuracy_score(test_targets, model.predict(test_features))
            logger.log("RESULT", "{}: test accuracy={:.8f}, support vectors={}", name, accuracy,
                       getattr(model, "support_count_", len(model.support_) if hasattr(model, "support_") else "n/a"))
    logger.info("SVM: completed")


if __name__ == "__main__":
    main()
