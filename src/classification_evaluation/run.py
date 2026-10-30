"""Compare hand-computed classification metrics with scikit-learn."""

import numpy as np
import pandas as pd
from sklearn import metrics as sklearn_metrics

from src.classification_evaluation.metrics import confusion, precision_recall_f1, roc_auc
from src.tools.logger import configure_logger, logger
from src.tools.paths import data_file, result_file


def main() -> None:
    configure_logger(result_file("classification_evaluation"))
    logger.info("Classification evaluation: started")
    logger.log("RESULT", "Classification evaluation results")
    for case, classes in (("binary", 2), ("multiclass", 3)):
        data = pd.read_csv(data_file("classification_evaluation", f"{case}.csv"))
        targets = data["target"].to_numpy()
        predictions = data["prediction"].to_numpy()
        own_confusion = confusion(targets, predictions, classes)
        reference_confusion = sklearn_metrics.confusion_matrix(targets, predictions, labels=np.arange(classes))
        logger.info("Classification evaluation: {}", case)
        logger.log("RESULT", "\n{}", case)
        for name, matrix in (("Manual", own_confusion), ("scikit-learn", reference_confusion)):
            logger.log("RESULT", "{} accuracy={:.8f}; confusion={}", name,
                       np.trace(matrix) / matrix.sum(), matrix.tolist())
            for average in ("micro", "macro"):
                if name == "Manual":
                    values = precision_recall_f1(matrix, average)
                else:
                    values = sklearn_metrics.precision_recall_fscore_support(
                        targets, predictions, average=average, zero_division=0,
                    )[:3]
                logger.log("RESULT", "{} {} precision={:.8f}, recall={:.8f}, F1={:.8f}", name, average, *values)
        if case == "binary":
            scores = data["score"].to_numpy()
            logger.log("RESULT", "Manual ROC AUC={:.8f}; scikit-learn ROC AUC={:.8f}",
                       roc_auc(targets, scores), sklearn_metrics.roc_auc_score(targets, scores))
    logger.info("Classification evaluation: completed")


if __name__ == "__main__":
    main()
