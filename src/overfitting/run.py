"""Framework-only model complexity, validation, and ridge comparison."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from src.tools.logger import configure_logger, logger
from src.tools.paths import data_file, result_file


def main() -> None:
    configure_logger(result_file("overfitting"))
    logger.info("Overfitting: started")
    data = pd.read_csv(data_file("overfitting", "polynomial.csv"))
    features = data[["feat_1"]].to_numpy()
    targets = data["target"].to_numpy()
    build_features, test_features, build_targets, test_targets = train_test_split(
        features, targets, test_size=0.5, random_state=151
    )
    train_features, validation_features, train_targets, validation_targets = train_test_split(
        build_features, build_targets, test_size=0.5, random_state=152
    )
    logger.log("RESULT", "Polynomial overfitting results (scikit-learn)")
    for degree in (1, 3, 8, 16):
        logger.info("Overfitting: polynomial degree {}", degree)
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(train_features, train_targets)
        errors = [mean_squared_error(values, model.predict(rows)) for rows, values in (
            (train_features, train_targets), (validation_features, validation_targets), (test_features, test_targets)
        )]
        logger.log("RESULT", "degree={}: train MSE={:.8f}, validation MSE={:.8f}, test MSE={:.8f}", degree, *errors)
    folds = KFold(n_splits=5, shuffle=True, random_state=151)
    scores = {}
    for degree in (1, 2, 3, 4, 8, 16):
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        scores[degree] = -cross_val_score(model, build_features, build_targets,
                                          cv=folds, scoring="neg_mean_squared_error").mean()
    selected_degree = min(scores, key=scores.get)
    selected_model = make_pipeline(PolynomialFeatures(selected_degree), LinearRegression())
    selected_model.fit(build_features, build_targets)
    ridge = make_pipeline(PolynomialFeatures(16), Ridge(alpha=0.01)).fit(build_features, build_targets)
    logger.log("RESULT", "CV selected degree={}; CV MSE={:.8f}; test MSE={:.8f}", selected_degree,
               scores[selected_degree], mean_squared_error(test_targets, selected_model.predict(test_features)))
    logger.log("RESULT", "degree=16 Ridge: test MSE={:.8f}",
               mean_squared_error(test_targets, ridge.predict(test_features)))
    logger.info("Overfitting: completed")


if __name__ == "__main__":
    main()
