"""Compare a NumPy K-means implementation with scikit-learn."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.tools.logger import configure_logger, logger
from src.tools.paths import data_file, result_file
from src.tools.timing import measure_time


class KMeansScratch:
    def __init__(self, initial_centers: np.ndarray, max_iter: int = 100):
        self.cluster_centers_ = initial_centers.copy()
        self.max_iter = max_iter
        self.n_iter_ = 0

    def fit(self, feature_matrix: np.ndarray):
        for iteration in range(self.max_iter):
            labels = self.predict(feature_matrix)
            centers = self.cluster_centers_.copy()
            for cluster in range(len(centers)):
                members = feature_matrix[labels == cluster]
                if len(members):
                    centers[cluster] = members.mean(axis=0)
            self.n_iter_ = iteration + 1
            converged = np.allclose(centers, self.cluster_centers_)
            self.cluster_centers_ = centers
            if converged:
                break
        return self

    def predict(self, feature_matrix: np.ndarray) -> np.ndarray:
        distances = np.sum((feature_matrix[:, None, :] - self.cluster_centers_[None, :, :]) ** 2, axis=2)
        return np.argmin(distances, axis=1)


def main() -> None:
    configure_logger(result_file("kmeans"))
    logger.info("K-means: started")
    logger.log("RESULT", "K-means results")
    data = pd.read_csv(data_file("kmeans", "clusters.csv"))
    feature_matrix = data[[name for name in data if name not in ("region", "target", "is_outlier")]].to_numpy()
    initial_centers = feature_matrix[[0, 90, 180]].copy()
    for name, model in (("Manual", KMeansScratch(initial_centers)),
                        ("scikit-learn", KMeans(n_clusters=3, init=initial_centers, n_init=1, random_state=41))):
        logger.info("K-means: fitting {}", name)
        model, fit_seconds = measure_time(model.fit, feature_matrix)
        labels = model.predict(feature_matrix)
        inertia = np.sum((feature_matrix - model.cluster_centers_[labels]) ** 2)
        logger.log("RESULT", "{}: inertia={:.8f}, iterations={}, fit time={:.6f} s",
                   name, inertia, model.n_iter_, fit_seconds)
        logger.log("RESULT", "{} centers={}", name, np.round(model.cluster_centers_, 8))

    pixels = pd.read_csv(data_file("kmeans", "pixels.csv"))
    pixel_matrix = pixels[["red", "green", "blue"]].to_numpy()
    model = KMeans(n_clusters=3, n_init=10, random_state=42).fit(pixel_matrix)
    reconstruction = model.cluster_centers_[model.labels_]
    logger.log("RESULT", "scikit-learn image quantization MSE: {:.8f}", np.mean((pixel_matrix - reconstruction) ** 2))
    logger.info("K-means: completed")


if __name__ == "__main__":
    main()
