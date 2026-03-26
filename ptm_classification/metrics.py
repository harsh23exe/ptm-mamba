from typing import Dict

import numpy as np
from sklearn import metrics


def compute_binary_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute standard binary classification metrics.

    y_true: 1-D array-like of 0/1.
    y_prob: 1-D array-like of predicted probability for class 1.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    out: Dict[str, float] = {}

    # Basic metrics
    out["accuracy"] = float(metrics.accuracy_score(y_true, y_pred))
    out["precision"] = float(metrics.precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(metrics.recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(metrics.f1_score(y_true, y_pred, zero_division=0))
    out["mcc"] = float(metrics.matthews_corrcoef(y_true, y_pred))

    # AUROC can fail if there is only one class in y_true
    try:
        out["auroc"] = float(metrics.roc_auc_score(y_true, y_prob))
    except Exception:
        out["auroc"] = float("nan")

    try:
        out["aupr"] = float(metrics.average_precision_score(y_true, y_prob))
    except Exception:
        out["aupr"] = float("nan")

    return out


