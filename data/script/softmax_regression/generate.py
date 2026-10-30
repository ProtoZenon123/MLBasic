"""Generate overlapping three-class features for softmax regression."""

import pandas as pd

from loguru import logger

from src.tools.paths import data_file
from src.tools.synthetic import gaussian_classes


def main() -> None:
    logger.info("Softmax Regression: generating data")
    features, labels = gaussian_classes(131, [(-2, -1), (2, -1), (0, 2)], 140, 1.2)
    data = pd.DataFrame(features, columns=["feat_1", "feat_2"])
    data["target"] = labels
    data.to_csv(data_file("softmax_regression", "classes.csv"), index=False)

    logger.info("Softmax Regression: data saved")


if __name__ == "__main__":
    main()
