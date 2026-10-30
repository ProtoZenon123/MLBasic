"""Generate noisy samples from a cubic polynomial."""

import numpy as np
import pandas as pd

from loguru import logger

from src.tools.paths import data_file


def main() -> None:
    logger.info("Overfitting: generating data")
    rng = np.random.default_rng(151)
    inputs = rng.uniform(-1.0, 1.0, 120)
    targets = 0.3 + 1.7 * inputs - 2.2 * inputs ** 2 + 1.4 * inputs ** 3 + rng.normal(0, 0.18, len(inputs))
    pd.DataFrame({"feat_1": inputs, "target": targets}).to_csv(data_file("overfitting", "polynomial.csv"), index=False)

    logger.info("Overfitting: data saved")


if __name__ == "__main__":
    main()
