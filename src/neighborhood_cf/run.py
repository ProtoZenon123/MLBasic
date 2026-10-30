"""User-user and item-item rating prediction against Surprise KNNWithMeans."""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.tools.logger import configure_logger, logger
from src.tools.paths import ROOT, data_file, result_file

os.environ.setdefault("SURPRISE_DATA_FOLDER", str(ROOT / "data" / "generated_data" / ".surprise_cache"))
from surprise import Dataset, KNNWithMeans, Reader  # noqa: E402


class NeighborhoodScratch:
    def __init__(self, user_based: bool, neighbors: int = 10):
        self.user_based = user_based
        self.neighbors = neighbors

    def fit(self, ratings: pd.DataFrame):
        self.values = np.full((ratings["user_id"].max() + 1, ratings["item_id"].max() + 1), np.nan)
        self.values[ratings["user_id"], ratings["item_id"]] = ratings["rating"]
        axis = 1 if self.user_based else 0
        self.means = np.nanmean(self.values, axis=axis)
        baselines = self.means[:, None] if self.user_based else self.means[None, :]
        centered = np.where(np.isnan(self.values), 0, self.values - baselines)
        vectors = centered if self.user_based else centered.T
        norms = np.linalg.norm(vectors, axis=1)
        denominator = np.outer(norms, norms)
        self.similarity = np.divide(
            vectors @ vectors.T, denominator, out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        np.fill_diagonal(self.similarity, 0)
        return self

    def predict_one(self, user_id: int, item_id: int) -> float:
        if self.user_based:
            candidates = np.flatnonzero(~np.isnan(self.values[:, item_id]))
            candidates = candidates[candidates != user_id]
            similarity = self.similarity[user_id, candidates]
            residuals = self.values[candidates, item_id] - self.means[candidates]
            baseline = self.means[user_id]
        else:
            candidates = np.flatnonzero(~np.isnan(self.values[user_id]))
            candidates = candidates[candidates != item_id]
            similarity = self.similarity[item_id, candidates]
            residuals = self.values[user_id, candidates] - self.means[candidates]
            baseline = self.means[item_id]
        order = np.argsort(-similarity)[:self.neighbors]
        weights = np.maximum(similarity[order], 0)
        adjustment = np.dot(weights, residuals[order]) / weights.sum() if weights.sum() else 0
        return float(np.clip(baseline + adjustment, 1, 5))


def main() -> None:
    configure_logger(result_file("neighborhood_cf"))
    logger.info("Neighborhood CF: started")
    ratings = pd.read_csv(data_file("neighborhood_cf", "ratings.csv"))
    train = ratings[ratings["split"] == "train"]
    test = ratings[ratings["split"] == "test"]
    surprise_train = Dataset.load_from_df(
        train[["user_id", "item_id", "rating"]], Reader(rating_scale=(1, 5)),
    ).build_full_trainset()
    logger.log("RESULT", "Neighborhood CF results; k=10, train={}, test={}", len(train), len(test))
    for user_based in (True, False):
        case = "user-user" if user_based else "item-item"
        for name, model in (("Manual", NeighborhoodScratch(user_based)),
                            ("Surprise", KNNWithMeans(
                                k=10, sim_options={"name": "pearson", "user_based": user_based}, verbose=False,
                            ))):
            logger.info("Neighborhood CF: {} {}", case, name)
            model.fit(train if name == "Manual" else surprise_train)
            predictions = [model.predict_one(int(row.user_id), int(row.item_id)) if name == "Manual"
                           else model.predict(int(row.user_id), int(row.item_id)).est for row in test.itertuples()]
            logger.log("RESULT", "{} {}: test MAE={:.8f}, RMSE={:.8f}", case, name,
                       mean_absolute_error(test["rating"], predictions),
                       np.sqrt(mean_squared_error(test["rating"], predictions)))
    logger.info("Neighborhood CF: completed")


if __name__ == "__main__":
    main()
