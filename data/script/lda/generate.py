"""Generate three classes in four dimensions for discriminant analysis."""

import pandas as pd

from loguru import logger

from src.tools.paths import data_file
from src.tools.synthetic import gaussian_classes


def main() -> None:
    logger.info("LDA: generating data")
    features, labels = gaussian_classes(291, [(-2, 0, 0, 0), (2, 0, 1, 0), (0, 2, -1, 1)], 120, 1.0)
    data = pd.DataFrame(features, columns=[f"feat_{i}" for i in range(1, 5)])
    data["target"] = labels
    data.to_csv(data_file("lda", "classes.csv"), index=False)

    logger.info("LDA: data saved")


if __name__ == "__main__":
    main()
