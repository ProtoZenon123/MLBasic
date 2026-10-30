"""Generate binary and multiclass labels, scores, and predictions."""

import numpy as np
import pandas as pd

from loguru import logger

from src.tools.paths import data_file


def main() -> None:
    logger.info("Classification Evaluation: generating data")
    rng = np.random.default_rng(331)
    binary_targets = rng.integers(0, 2, 180)
    binary_scores = 1 / (1 + np.exp(-(2 * binary_targets - 1 + rng.normal(0, 1.0, 180))))
    pd.DataFrame({"target": binary_targets, "prediction": (binary_scores >= 0.5).astype(int),
                  "score": binary_scores}).to_csv(data_file("classification_evaluation", "binary.csv"), index=False)
    multiclass_targets = rng.integers(0, 3, 180)
    scores = rng.normal(size=(180, 3)) + 1.5 * np.eye(3)[multiclass_targets]
    scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    scores /= scores.sum(axis=1, keepdims=True)
    data = pd.DataFrame(scores, columns=[f"score_{index}" for index in range(3)])
    data["target"] = multiclass_targets
    data["prediction"] = scores.argmax(axis=1)
    data.to_csv(data_file("classification_evaluation", "multiclass.csv"), index=False)

    logger.info("Classification Evaluation: data saved")


if __name__ == "__main__":
    main()
