"""Generate categorical weather-like observations from a planted rule."""

import numpy as np
import pandas as pd

from loguru import logger

from src.tools.paths import data_file


def main() -> None:
    logger.info("ID3: generating data")
    rng = np.random.default_rng(341)
    count = 360
    outlook = rng.choice(["sunny", "overcast", "rainy"], count)
    humidity = rng.choice(["high", "normal"], count)
    wind = rng.choice(["weak", "strong"], count)
    temperature = rng.choice(["cool", "mild", "hot"], count)
    target = np.where(outlook == "overcast", "yes",
                      np.where(outlook == "sunny", np.where(humidity == "normal", "yes", "no"),
                               np.where(wind == "weak", "yes", "no")))
    data = pd.DataFrame({"outlook": outlook, "humidity": humidity, "wind": wind,
                         "temperature": temperature, "target": target})
    data.to_csv(data_file("id3", "categories.csv"), index=False)

    logger.info("ID3: data saved")


if __name__ == "__main__":
    main()
