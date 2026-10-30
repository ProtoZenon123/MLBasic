"""PCA reconstruction and its singular-value relationship."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from src.tools.logger import configure_logger, logger
from src.tools.paths import data_file, result_file


def main() -> None:
    configure_logger(result_file("pca"))
    logger.info("PCA: started")
    features = pd.read_csv(data_file("pca", "correlated.csv")).to_numpy()
    train_features, test_features = train_test_split(features, test_size=0.3, random_state=271)
    singular_values = np.linalg.svd(train_features - train_features.mean(axis=0), compute_uv=False)
    logger.log("RESULT", "PCA results (scikit-learn); original shape={}", features.shape)
    for components in (1, 2, 3, 4):
        logger.info("PCA: {} components", components)
        model = PCA(n_components=components).fit(train_features)
        reconstructed = model.inverse_transform(model.transform(test_features))
        logger.log(
            "RESULT", "components={}: explained variance={:.8f}, test reconstruction MSE={:.8f}, SVD singular gap={:.8f}",
            components, model.explained_variance_ratio_.sum(),
            np.mean((test_features - reconstructed) ** 2),
            np.max(np.abs(model.singular_values_ - singular_values[:components])),
        )
    logger.info("PCA: completed")


if __name__ == "__main__":
    main()
