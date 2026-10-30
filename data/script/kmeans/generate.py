"""Generate cluster points and a tiny RGB image with planted regions."""

import numpy as np
import pandas as pd

from loguru import logger

from src.tools.paths import data_file
from src.tools.synthetic import gaussian_classes


def main() -> None:
    logger.info("K-means: generating data")
    points, labels = gaussian_classes(41, [(-4.0, -3.0), (0.0, 4.0), (4.0, -2.0)], 90, 0.75)
    pd.DataFrame({"feat_1": points[:, 0], "feat_2": points[:, 1], "region": labels}).to_csv(
        data_file("kmeans", "clusters.csv"), index=False
    )
    rng = np.random.default_rng(42)
    row, col = np.indices((24, 24))
    region = np.where(row < 12, 0, np.where(col < 12, 1, 2)).ravel()
    palette = np.array([[220, 45, 40], [35, 180, 75], [45, 70, 210]])
    pixels = np.clip(palette[region] + rng.normal(0, 12, (len(region), 3)), 0, 255)
    pd.DataFrame({"red": pixels[:, 0], "green": pixels[:, 1], "blue": pixels[:, 2], "region": region}).to_csv(
        data_file("kmeans", "pixels.csv"), index=False
    )

    logger.info("K-means: data saved")


if __name__ == "__main__":
    main()
