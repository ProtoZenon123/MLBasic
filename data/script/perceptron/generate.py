"""Generate separable and label-noisy binary classification data."""

import numpy as np
import pandas as pd

from loguru import logger

from src.tools.paths import data_file
from src.tools.synthetic import gaussian_classes


def main() -> None:
    logger.info("Perceptron: generating data")
    features, labels = gaussian_classes(91, [(-3.0, -3.0), (3.0, 3.0)], 100, 0.55)
    for case in ("separable", "noisy"):
        targets = labels.copy()
        if case == "noisy":
            rng = np.random.default_rng(92)
            targets[rng.choice(len(targets), 22, replace=False)] ^= 1
        data = pd.DataFrame(features, columns=["feat_1", "feat_2"])
        data["target"] = targets
        data.to_csv(data_file("perceptron", f"{case}.csv"), index=False)

    logger.info("Perceptron: data saved")


if __name__ == "__main__":
    main()
