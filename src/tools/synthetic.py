"""Small reproducible data recipes used by several classification lessons."""

import numpy as np
import pandas as pd


def gaussian_classes(seed: int, centers: list[tuple[float, ...]], samples: int, scale: float):
    """Return rows of features and integer labels for Gaussian class clouds."""
    rng = np.random.default_rng(seed)
    feature_matrix = np.vstack(
        [rng.normal(center, scale, size=(samples, len(center))) for center in centers]
    )
    target_values = np.repeat(np.arange(len(centers)), samples)
    order = rng.permutation(len(target_values))
    return feature_matrix[order], target_values[order]


def sparse_ratings(seed: int) -> pd.DataFrame:
    """Create explicit ratings with held-out observations for each user."""
    rng = np.random.default_rng(seed)
    user_factors = rng.normal(size=(24, 4))
    item_factors = rng.normal(size=(36, 4))
    records = []
    for user_id in range(len(user_factors)):
        chosen_items = rng.choice(len(item_factors), 22, replace=False)
        for position, item_id in enumerate(chosen_items):
            value = 3 + 0.45 * (user_factors[user_id] @ item_factors[item_id]) + rng.normal(0, 0.2)
            records.append((user_id, int(item_id), float(np.clip(value, 1, 5)),
                            "test" if position < 5 else "train"))
    return pd.DataFrame(records, columns=["user_id", "item_id", "rating", "split"])
