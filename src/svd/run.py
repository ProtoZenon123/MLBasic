"""SVD orthogonality, reconstruction, and truncated-rank error."""

import numpy as np

from src.tools.logger import configure_logger, logger
from src.tools.paths import data_file, result_file


def main() -> None:
    configure_logger(result_file("svd"))
    logger.info("SVD: started")
    feature_matrix = np.load(data_file("svd", "matrix.npz"))["matrix"]
    left, singular_values, right_transposed = np.linalg.svd(feature_matrix, full_matrices=False)
    reconstruction = (left * singular_values) @ right_transposed
    logger.log("RESULT", "SVD results (NumPy); shape={}", feature_matrix.shape)
    left_error = np.linalg.norm(left.T @ left - np.eye(len(singular_values)))
    right_error = np.linalg.norm(right_transposed @ right_transposed.T - np.eye(len(singular_values)))
    logger.log("RESULT", "left orthogonality error={:.8f}", left_error)
    logger.log("RESULT", "right orthogonality error={:.8f}", right_error)
    logger.log("RESULT", "full reconstruction Frobenius error={:.8f}", np.linalg.norm(feature_matrix - reconstruction))
    for rank in (1, 2, 3, 4):
        approximation = (left[:, :rank] * singular_values[:rank]) @ right_transposed[:rank]
        logger.log("RESULT", "rank {} approximation Frobenius error={:.8f}", rank,
                   np.linalg.norm(feature_matrix - approximation))
    logger.info("SVD: completed")


if __name__ == "__main__":
    main()
