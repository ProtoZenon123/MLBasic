"""Generate a regression dataset and a quadratic objective."""

import numpy as np
import pandas as pd

from loguru import logger

from src.tools.paths import data_file


def main() -> None:
    logger.info("Gradient Descent: generating data")
    rng = np.random.default_rng(78)
    features = rng.normal(size=(240, 3))
    targets = 1.2 + features @ np.array([2.0, -1.5, 0.7]) + rng.normal(0, 0.25, 240)
    data = pd.DataFrame(features, columns=[f"feat_{i}" for i in range(1, 4)])
    data["target"] = targets
    data.to_csv(data_file("gradient_descent", "regression.csv"), index=False)
    pd.DataFrame({"curvature": [1.0, 8.0], "initial": [5.0, -3.0]}).to_csv(
        data_file("gradient_descent", "quadratic.csv"), index=False
    )

    logger.info("Gradient Descent: data saved")


if __name__ == "__main__":
    main()
