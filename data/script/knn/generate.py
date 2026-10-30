"""Generate a three-class dataset for nearest-neighbor classification."""

import pandas as pd

from loguru import logger

from src.tools.paths import data_file
from src.tools.synthetic import gaussian_classes


def main() -> None:
    logger.info("KNN: generating data")
    features, labels = gaussian_classes(61, [(-2.5, 0), (2.5, 0), (0, 3)], 100, 1.0)
    data = pd.DataFrame(features, columns=["feat_1", "feat_2"])
    data["target"] = labels
    data.to_csv(data_file("knn", "classification.csv"), index=False)

    logger.info("KNN: data saved")


if __name__ == "__main__":
    main()
