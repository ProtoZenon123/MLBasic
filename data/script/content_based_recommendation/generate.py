"""Generate item genre profiles and user ratings from planted preferences."""

import numpy as np
import pandas as pd

from loguru import logger

from src.tools.paths import data_file


def main() -> None:
    logger.info("Content Based Recommendation: generating data")
    rng = np.random.default_rng(231)
    genres = (rng.uniform(size=(42, 6)) < 0.4).astype(int)
    empty_items = genres.sum(axis=1) == 0
    genres[empty_items, 0] = 1
    items = pd.DataFrame(genres, columns=[f"genre_{i}" for i in range(1, 7)])
    items.insert(0, "item_id", np.arange(len(items)))
    items.to_csv(data_file("content_based_recommendation", "items.csv"), index=False)
    preferences = rng.normal(size=(18, 6))
    records = []
    for user_id in range(len(preferences)):
        chosen = rng.choice(len(items), 22, replace=False)
        for position, item_id in enumerate(chosen):
            rating = 3 + 0.65 * (genres[item_id] @ preferences[user_id]) + rng.normal(0, 0.18)
            records.append((user_id, int(item_id), float(np.clip(rating, 1, 5)),
                            "test" if position < 5 else "train"))
    pd.DataFrame(records, columns=["user_id", "item_id", "rating", "split"]).to_csv(
        data_file("content_based_recommendation", "ratings.csv"), index=False
    )

    logger.info("Content Based Recommendation: data saved")


if __name__ == "__main__":
    main()
