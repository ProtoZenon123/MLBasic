"""Behavior checks for reproducibility, edge cases, and numerical agreement."""

import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from data.script.svd.generate import generate_matrix
from src.classification_evaluation.metrics import confusion, precision_recall_f1, roc_auc
from src.id3.run import ID3Scratch
from src.kmeans.run import KMeansScratch
from src.knn.run import KNNScratch
from src.tools.synthetic import gaussian_classes, sparse_ratings


class PracticeTests(unittest.TestCase):
    def test_generators_are_reproducible_and_ratings_holdout_is_disjoint(self):
        first_features, first_labels = gaussian_classes(17, [(0, 0), (1, 1)], 8, 0.5)
        second_features, second_labels = gaussian_classes(17, [(0, 0), (1, 1)], 8, 0.5)
        np.testing.assert_array_equal(first_features, second_features)
        np.testing.assert_array_equal(first_labels, second_labels)
        ratings = sparse_ratings(17)
        pd.testing.assert_frame_equal(ratings, sparse_ratings(17))
        train = set(map(tuple, ratings.loc[ratings["split"] == "train", ["user_id", "item_id"]].to_numpy()))
        test = set(map(tuple, ratings.loc[ratings["split"] == "test", ["user_id", "item_id"]].to_numpy()))
        self.assertFalse(train & test)
        self.assertTrue((ratings[ratings["split"] == "train"].groupby("user_id").size() >= 1).all())

    def test_knn_rejects_too_many_neighbors_and_resolves_vote_ties(self):
        model = KNNScratch(neighbors=3).fit(np.array([[0.0], [1.0]]), np.array([1, 0]))
        with self.assertRaises(ValueError):
            model.predict(np.array([[0.2]]))
        model.neighbors = 2
        self.assertEqual(model.predict(np.array([[0.5]])).item(), 0)

    def test_kmeans_empty_cluster_keeps_finite_center(self):
        model = KMeansScratch(np.array([[0.0], [10.0], [20.0]])).fit(np.array([[0.0], [10.0]]))
        self.assertTrue(np.isfinite(model.cluster_centers_).all())

    def test_metrics_match_sklearn_with_ties_and_missing_predictions(self):
        targets = np.array([0, 1, 0, 1, 2, 2])
        predictions = np.array([0, 0, 0, 1, 0, 1])
        matrix = confusion(targets, predictions, 3)
        np.testing.assert_array_equal(matrix, confusion_matrix(targets, predictions, labels=[0, 1, 2]))
        for average in ("micro", "macro"):
            expected = precision_recall_fscore_support(targets, predictions, average=average, zero_division=0)[:3]
            np.testing.assert_allclose(precision_recall_f1(matrix, average), expected)
        binary_targets = np.array([0, 1, 0, 1, 1, 0])
        scores = np.array([0.2, 0.8, 0.2, 0.5, 0.5, 0.8])
        self.assertAlmostEqual(roc_auc(binary_targets, scores), roc_auc_score(binary_targets, scores))
        with self.assertRaises(ValueError):
            roc_auc(np.ones(3, dtype=int), np.array([0.1, 0.2, 0.3]))

    def test_id3_uses_majority_for_unseen_category(self):
        features = pd.DataFrame({"color": ["red", "red", "blue"], "shape": ["round", "round", "round"]})
        targets = np.array(["yes", "yes", "no"])
        model = ID3Scratch().fit(features, targets)
        prediction = model.predict(pd.DataFrame({"color": ["green"], "shape": ["round"]}))
        self.assertEqual(prediction.item(), "yes")

    def test_svd_generated_matrix_is_nearly_rank_three(self):
        feature_matrix = generate_matrix()
        np.testing.assert_array_equal(feature_matrix, generate_matrix())
        left, singular_values, right = np.linalg.svd(feature_matrix, full_matrices=False)
        rank_two = (left[:, :2] * singular_values[:2]) @ right[:2]
        rank_three = (left[:, :3] * singular_values[:3]) @ right[:3]
        self.assertLess(np.linalg.norm(feature_matrix - rank_three),
                        0.1 * np.linalg.norm(feature_matrix - rank_two))


if __name__ == "__main__":
    unittest.main()
