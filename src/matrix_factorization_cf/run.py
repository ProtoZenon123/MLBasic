"""Manual latent factors with SGD compared to Surprise SVD."""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.tools.logger import configure_logger, logger
from src.tools.paths import ROOT, data_file, result_file

os.environ.setdefault("SURPRISE_DATA_FOLDER", str(ROOT / "data" / "generated_data" / ".surprise_cache"))
from surprise import Dataset, Reader, SVD  # noqa: E402


class MatrixFactorizationScratch:
    def __init__(self, factors: int = 4, epochs: int = 80):
        self.factors = factors
        self.epochs = epochs

    def fit(self, ratings: pd.DataFrame):
        rng = np.random.default_rng(251)
        users = int(ratings["user_id"].max()) + 1
        items = int(ratings["item_id"].max()) + 1
        self.user_factors = rng.normal(0, 0.1, (users, self.factors))
        self.item_factors = rng.normal(0, 0.1, (items, self.factors))
        self.user_bias = np.zeros(users)
        self.item_bias = np.zeros(items)
        self.global_mean = float(ratings["rating"].mean())
        records = ratings[["user_id", "item_id", "rating"]].to_numpy()
        for _ in range(self.epochs):
            for index in rng.permutation(len(records)):
                user_id, item_id, rating = records[index]
                user_id, item_id = int(user_id), int(item_id)
                error = rating - self.predict_one(user_id, item_id, clip=False)
                old_user = self.user_factors[user_id].copy()
                self.user_bias[user_id] += 0.02 * (error - 0.03 * self.user_bias[user_id])
                self.item_bias[item_id] += 0.02 * (error - 0.03 * self.item_bias[item_id])
                self.user_factors[user_id] += 0.02 * (error * self.item_factors[item_id] - 0.03 * old_user)
                self.item_factors[item_id] += 0.02 * (error * old_user - 0.03 * self.item_factors[item_id])
        return self

    def predict_one(self, user_id: int, item_id: int, clip: bool = True) -> float:
        value = self.global_mean + self.user_bias[user_id] + self.item_bias[item_id]
        value += self.user_factors[user_id] @ self.item_factors[item_id]
        return float(np.clip(value, 1, 5) if clip else value)


def main() -> None:
    configure_logger(result_file("matrix_factorization_cf"))
    logger.info("Matrix factorization CF: started")
    ratings = pd.read_csv(data_file("matrix_factorization_cf", "ratings.csv"))
    train = ratings[ratings["split"] == "train"]
    test = ratings[ratings["split"] == "test"]
    surprise_train = Dataset.load_from_df(
        train[["user_id", "item_id", "rating"]], Reader(rating_scale=(1, 5)),
    ).build_full_trainset()
    logger.log(
        "RESULT", "Matrix factorization CF results; factors=4, epochs=80, learning rate=0.02, regularization=0.03; train={}, test={}",
        len(train), len(test),
    )
    for name, model in (("Manual", MatrixFactorizationScratch()),
                        ("Surprise", SVD(n_factors=4, n_epochs=80, lr_all=0.02,
                                          reg_all=0.03, random_state=251))):
        logger.info("Matrix factorization CF: fitting {}", name)
        model.fit(train if name == "Manual" else surprise_train)
        train_predictions = [model.predict_one(int(row.user_id), int(row.item_id)) if name == "Manual"
                             else model.predict(int(row.user_id), int(row.item_id)).est for row in train.itertuples()]
        test_predictions = [model.predict_one(int(row.user_id), int(row.item_id)) if name == "Manual"
                            else model.predict(int(row.user_id), int(row.item_id)).est for row in test.itertuples()]
        logger.log("RESULT", "{}: train MSE={:.8f}, test MAE={:.8f}, test RMSE={:.8f}", name,
                   mean_squared_error(train["rating"], train_predictions),
                   mean_absolute_error(test["rating"], test_predictions),
                   np.sqrt(mean_squared_error(test["rating"], test_predictions)))
    logger.info("Matrix factorization CF: completed")


if __name__ == "__main__":
    main()
