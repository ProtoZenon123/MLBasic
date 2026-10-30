"""Generate several classes for binary and multiclass wrappers."""

import pandas as pd

from loguru import logger

from src.tools.paths import data_file
from src.tools.synthetic import gaussian_classes


def main() -> None:
    logger.info("Binary Classifiers: generating data")
    features, labels = gaussian_classes(121, [(-2.5, -1), (2.5, -1), (0, 3)], 110, 0.95)
    data = pd.DataFrame(features, columns=["feat_1", "feat_2"])
    data["target"] = labels
    data.to_csv(data_file("binary_classifiers", "classes.csv"), index=False)

    logger.info("Binary Classifiers: data saved")


if __name__ == "__main__":
    main()
