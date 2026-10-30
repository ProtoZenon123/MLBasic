"""Generate six correlated features from two latent variables."""

import numpy as np
import pandas as pd

from loguru import logger

from src.tools.paths import data_file


def main() -> None:
    logger.info("PCA: generating data")
    rng = np.random.default_rng(271)
    latent = rng.normal(size=(300, 2)) * [2.5, 0.9]
    projection = rng.normal(size=(2, 6))
    features = latent @ projection + rng.normal(0, 0.08, (300, 6))
    pd.DataFrame(features, columns=[f"feat_{i}" for i in range(1, 7)]).to_csv(
        data_file("pca", "correlated.csv"), index=False
    )

    logger.info("PCA: data saved")


if __name__ == "__main__":
    main()
