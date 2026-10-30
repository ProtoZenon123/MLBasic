"""Categorical ID3 with information gain; compare a CART reference."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.tools.logger import configure_logger, logger
from src.tools.paths import data_file, result_file


def entropy(target_values: np.ndarray) -> float:
    counts = np.unique(target_values, return_counts=True)[1]
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


class ID3Scratch:
    def fit(self, feature_frame: pd.DataFrame, target_values: np.ndarray):
        self.tree_ = self._build(feature_frame, target_values, list(feature_frame.columns))
        return self

    def _build(self, features: pd.DataFrame, targets: np.ndarray, available: list[str]):
        values, counts = np.unique(targets, return_counts=True)
        majority = values[counts.argmax()]
        node = {"majority": majority}
        if len(values) == 1 or not available:
            return node
        parent_entropy = entropy(targets)
        gains = {}
        for feature in available:
            weighted = 0.0
            for value in features[feature].unique():
                mask = (features[feature] == value).to_numpy()
                weighted += mask.mean() * entropy(targets[mask])
            gains[feature] = parent_entropy - weighted
        selected = max(gains, key=gains.get)
        if gains[selected] <= 1e-12:
            return node
        node["feature"] = selected
        node["children"] = {}
        for value in features[selected].unique():
            mask = (features[selected] == value).to_numpy()
            node["children"][value] = self._build(features.loc[mask], targets[mask],
                                                   [name for name in available if name != selected])
        return node

    def predict(self, feature_frame: pd.DataFrame):
        predictions = []
        for _, row in feature_frame.iterrows():
            node = self.tree_
            while "feature" in node:
                node = node["children"].get(row[node["feature"]], {"majority": node["majority"]})
            predictions.append(node["majority"])
        return np.array(predictions)


def count_nodes(node: dict) -> tuple[int, int]:
    if "children" not in node:
        return 1, 1
    child_counts = [count_nodes(child) for child in node["children"].values()]
    return 1 + sum(count for count, _ in child_counts), sum(leaves for _, leaves in child_counts)


def main() -> None:
    configure_logger(result_file("id3"))
    logger.info("ID3: started")
    data = pd.read_csv(data_file("id3", "categories.csv"))
    feature_names = [name for name in data if name != "target"]
    features = data[feature_names]
    targets = data["target"].to_numpy()
    train_features, test_features, train_targets, test_targets = train_test_split(
        features, targets, test_size=0.3, random_state=341, stratify=targets
    )
    manual = ID3Scratch().fit(train_features, train_targets)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train_features)
    cart = DecisionTreeClassifier(criterion="entropy", random_state=341).fit(
        encoder.transform(train_features), train_targets
    )
    logger.log("RESULT", "ID3 results (scikit-learn reference uses CART, not ID3)")
    logger.log("RESULT", "Manual ID3: test accuracy={:.8f}, nodes={}, leaves={}",
               accuracy_score(test_targets, manual.predict(test_features)), *count_nodes(manual.tree_))
    logger.log("RESULT", "scikit-learn CART: test accuracy={:.8f}, depth={}, leaves={}",
               accuracy_score(test_targets, cart.predict(encoder.transform(test_features))),
               cart.get_depth(), cart.get_n_leaves())
    logger.info("ID3: completed")


if __name__ == "__main__":
    main()
