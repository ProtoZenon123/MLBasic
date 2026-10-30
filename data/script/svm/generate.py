"""Generate separable, overlapping, nonlinear, and multiclass SVM data."""

import numpy as np
import pandas as pd

from loguru import logger

from src.tools.paths import data_file
from src.tools.synthetic import gaussian_classes


def write_case(name: str, features: np.ndarray, labels: np.ndarray) -> None:
    data = pd.DataFrame(features, columns=[f"feat_{i}" for i in range(1, features.shape[1] + 1)])
    data["target"] = labels
    data.to_csv(data_file("svm", f"{name}.csv"), index=False)


def main() -> None:
    logger.info("SVM: generating data")
    for name, seed, centers, scale in (
        ("hard", 191, [(-2, -2), (2, 2)], 0.45),
        ("soft", 201, [(-1, -1), (1, 1)], 1.25),
        ("multiclass", 221, [(-2.5, -1), (2.5, -1), (0, 3)], 0.95),
    ):
        write_case(name, *gaussian_classes(seed, centers, 80, scale))
    rng = np.random.default_rng(211)
    angles = rng.uniform(0, np.pi, 120)
    first = np.column_stack((np.cos(angles), np.sin(angles)))
    second = np.column_stack((1 - np.cos(angles), 0.5 - np.sin(angles)))
    features = np.vstack((first, second)) + rng.normal(0, 0.07, (240, 2))
    write_case("kernel", features, np.repeat([0, 1], 120))

    logger.info("SVM: data saved")


if __name__ == "__main__":
    main()
