"""TF-IDF item profiles and one Ridge rating model per user."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.tools.logger import configure_logger, logger
from src.tools.paths import data_file, result_file


def main() -> None:
    configure_logger(result_file("content_based_recommendation"))
    logger.info("Content-based recommendation: started")
    items = pd.read_csv(data_file("content_based_recommendation", "items.csv"))
    ratings = pd.read_csv(data_file("content_based_recommendation", "ratings.csv"))
    profile_matrix = TfidfTransformer().fit_transform(items.drop(columns="item_id").to_numpy()).toarray()
    train = ratings[ratings["split"] == "train"]
    test = ratings[ratings["split"] == "test"]
    predictions = np.empty(len(test))
    logger.info("Content-based recommendation: fitting {} user profiles", train["user_id"].nunique())
    for user_id in train["user_id"].unique():
        known = train[train["user_id"] == user_id]
        unknown = test[test["user_id"] == user_id]
        model = Ridge(alpha=0.1).fit(profile_matrix[known["item_id"]], known["rating"])
        predictions[test.index.get_indexer(unknown.index)] = model.predict(profile_matrix[unknown["item_id"]])
    predictions = np.clip(predictions, 1, 5)
    logger.log("RESULT", "Content-based recommendation (scikit-learn TF-IDF + Ridge, alpha=0.1)")
    logger.log("RESULT", "train ratings={}, test ratings={}, profile shape={}",
               len(train), len(test), profile_matrix.shape)
    logger.log("RESULT", "test MAE={:.8f}, RMSE={:.8f}", mean_absolute_error(test["rating"], predictions),
               np.sqrt(mean_squared_error(test["rating"], predictions)))
    logger.info("Content-based recommendation: completed")


if __name__ == "__main__":
    main()
