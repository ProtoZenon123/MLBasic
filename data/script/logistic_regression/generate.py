"""Generate logistic probabilities, then sample binary labels."""

import numpy as np
import pandas as pd

from loguru import logger

from src.tools.paths import data_file


def main() -> None:
    logger.info("Logistic Regression: generating data")
    rng = np.random.default_rng(101)
    features = rng.normal(size=(500, 3))
    logits = features @ np.array([1.4, -1.2, 0.8]) + 0.35
    probabilities = 1 / (1 + np.exp(-logits))
    targets = rng.binomial(1, probabilities)
    data = pd.DataFrame(features, columns=[f"feat_{i}" for i in range(1, 4)])
    data["target"] = targets
    data.to_csv(data_file("logistic_regression", "binary.csv"), index=False)

    logger.info("Logistic Regression: data saved")


if __name__ == "__main__":
    main()
