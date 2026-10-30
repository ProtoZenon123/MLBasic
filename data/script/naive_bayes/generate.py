"""Generate word-count features with class-dependent frequencies."""

import numpy as np
import pandas as pd

from loguru import logger

from src.tools.paths import data_file


def main() -> None:
    logger.info("Naive Bayes: generating data")
    rng = np.random.default_rng(321)
    rates = np.array([[3.5, 2.8, 0.3, 0.5, 1.0, 0.5],
                      [0.4, 0.8, 3.2, 2.5, 0.4, 1.0],
                      [0.7, 0.4, 0.5, 0.7, 3.5, 2.8]])
    targets = np.repeat(np.arange(3), 120)
    counts = rng.poisson(rates[targets])
    data = pd.DataFrame(counts, columns=[f"feat_{i}" for i in range(1, 7)])
    data["target"] = targets
    data.to_csv(data_file("naive_bayes", "word_counts.csv"), index=False)

    logger.info("Naive Bayes: data saved")


if __name__ == "__main__":
    main()
