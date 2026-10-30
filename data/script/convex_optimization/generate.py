"""Generate positive coefficients for a feasible geometric program."""

import numpy as np
import pandas as pd

from loguru import logger

from src.tools.paths import data_file


def main() -> None:
    logger.info("Convex Optimization: generating data")
    rng = np.random.default_rng(171)
    coefficients = rng.uniform(1, 4, 2)
    pd.DataFrame({"coefficient": coefficients, "budget": [10.0, 10.0]}).to_csv(
        data_file("convex_optimization", "geometric_program.csv"), index=False
    )

    logger.info("Convex Optimization: data saved")


if __name__ == "__main__":
    main()
