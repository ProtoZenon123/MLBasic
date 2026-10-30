"""Multinomial and Bernoulli Naive Bayes framework demonstrations."""

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from src.tools.logger import configure_logger, logger
from src.tools.paths import data_file, result_file


def main() -> None:
    configure_logger(result_file("naive_bayes"))
    logger.info("Naive Bayes: started")
    data = pd.read_csv(data_file("naive_bayes", "word_counts.csv"))
    features = data[[name for name in data if name != "target"]].to_numpy()
    targets = data["target"].to_numpy()
    train_features, test_features, train_targets, test_targets = train_test_split(
        features, targets, test_size=0.3, random_state=321, stratify=targets
    )
    logger.log("RESULT", "Naive Bayes results (scikit-learn)")
    for name, model, binary in (("MultinomialNB", MultinomialNB(), False),
                                ("BernoulliNB", BernoulliNB(), True)):
        logger.info("Naive Bayes: {}", name)
        prepared_train = (train_features > 0).astype(int) if binary else train_features
        prepared_test = (test_features > 0).astype(int) if binary else test_features
        model.fit(prepared_train, train_targets)
        predictions = model.predict(prepared_test)
        logger.log("RESULT", "{}: test accuracy={:.8f}; confusion={}", name,
                   accuracy_score(test_targets, predictions), confusion_matrix(test_targets, predictions).tolist())
    logger.info("Naive Bayes: completed")


if __name__ == "__main__":
    main()
