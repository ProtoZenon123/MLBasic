"""Generate sparse ratings from a planted latent-factor model."""

from loguru import logger

from src.tools.paths import data_file
from src.tools.synthetic import sparse_ratings


def main() -> None:
    logger.info("Matrix Factorization CF: generating data")
    sparse_ratings(251).to_csv(data_file("matrix_factorization_cf", "ratings.csv"), index=False)

    logger.info("Matrix Factorization CF: data saved")


if __name__ == "__main__":
    main()
