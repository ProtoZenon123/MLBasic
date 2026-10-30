"""Framework-only binary, one-vs-rest, and one-vs-one demonstrations."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

from src.tools.logger import configure_logger, logger
from src.tools.paths import data_file, result_file


def main() -> None:
    configure_logger(result_file("binary_classifiers"))
    logger.info("Binary classifiers: started")
    data = pd.read_csv(data_file("binary_classifiers", "classes.csv"))
    features = data[[name for name in data if name != "target"]].to_numpy()
    targets = data["target"].to_numpy()
    logger.log("RESULT", "Binary classifiers results (scikit-learn)")
    for case, case_features, case_targets in (
        ("binary", features[targets != 2], targets[targets != 2]),
        ("multiclass", features, targets),
    ):
        train_features, test_features, train_targets, test_targets = train_test_split(
            case_features, case_targets, test_size=0.3, random_state=121, stratify=case_targets
        )
        models = (("binary logistic", LogisticRegression(C=np.inf)),) if case == "binary" else (
            ("one-vs-rest", OneVsRestClassifier(LogisticRegression(C=np.inf))),
            ("one-vs-one", OneVsOneClassifier(LogisticRegression(C=np.inf))),
        )
        for name, model in models:
            logger.info("Binary classifiers: {} {}", case, name)
            model.fit(train_features, train_targets)
            predictions = model.predict(test_features)
            logger.log("RESULT", "{} {}: accuracy={:.8f}; confusion={}", case, name,
                       accuracy_score(test_targets, predictions), confusion_matrix(test_targets, predictions).tolist())
    logger.info("Binary classifiers: completed")


if __name__ == "__main__":
    main()
