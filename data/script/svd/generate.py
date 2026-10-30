"""Generate a noisy low-rank matrix for SVD checks."""

import numpy as np

from loguru import logger

from src.tools.paths import data_file


def generate_matrix(seed: int = 261) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(40, 3)) @ rng.normal(size=(3, 25))
    matrix += rng.normal(0, 0.03, matrix.shape)
    return matrix


def main() -> None:
    logger.info("SVD: generating data")
    np.savez(data_file("svd", "matrix.npz"), matrix=generate_matrix())

    logger.info("SVD: data saved")


if __name__ == "__main__":
    main()
