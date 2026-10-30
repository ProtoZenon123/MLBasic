"""Generate sparse explicit user-item ratings."""

from loguru import logger

from src.tools.paths import data_file
from src.tools.synthetic import sparse_ratings


def main() -> None:
    logger.info("Neighborhood CF: generating data")
    sparse_ratings(241).to_csv(data_file("neighborhood_cf", "ratings.csv"), index=False)

    logger.info("Neighborhood CF: data saved")


if __name__ == "__main__":
    main()
