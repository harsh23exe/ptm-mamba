from typing import Dict, List, Optional

import numpy as np
from sklearn import metrics


def compute_binary_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    """
    Binary metrics including MCC, F1, AUROC, and AUPR (average_precision).

    y_true: 1-D array-like of 0/1.
    y_prob: 1-D array-like of predicted probability for class 1.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    out: Dict[str, float] = {}
    out["accuracy"] = float(metrics.accuracy_score(y_true, y_pred))
    out["precision"] = float(metrics.precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(metrics.recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(metrics.f1_score(y_true, y_pred, zero_division=0))
    out["mcc"] = float(metrics.matthews_corrcoef(y_true, y_pred))

    try:
        out["auroc"] = float(metrics.roc_auc_score(y_true, y_prob))
    except Exception:
        out["auroc"] = float("nan")

    try:
        out["aupr"] = float(metrics.average_precision_score(y_true, y_prob))
    except Exception:
        out["aupr"] = float("nan")

    return out


def metrics_per_ptm(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ptm_ids: np.ndarray,
    ptm_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Slice flat arrays by PTM id and compute binary metrics for each slice.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    ptm_ids = np.asarray(ptm_ids).astype(int)
    out: Dict[str, Dict[str, float]] = {}
    for pid in sorted(np.unique(ptm_ids)):
        mask = ptm_ids == pid
        name = ptm_names[pid] if ptm_names is not None else str(pid)
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        yp = y_prob[mask]
        if len(np.unique(yt)) < 2:
            entry = {
                "n": float(mask.sum()),
                "note": "single_class_in_subset",
                "mcc": float("nan"),
                "f1": float("nan"),
                "aupr": float("nan"),
                "auroc": float("nan"),
            }
            out[name] = entry
            continue
        m = compute_binary_metrics(yt, yp)
        m["n"] = float(mask.sum())
        out[name] = m
    return out


def aggregate_macro(
    per_ptm: Dict[str, Dict[str, float]], keys: List[str]
) -> Dict[str, float]:
    """Macro average over PTM groups for numeric keys (ignores NaN)."""
    agg: Dict[str, float] = {}
    names = list(per_ptm.keys())
    for k in keys:
        vals = []
        for name in names:
            v = per_ptm[name].get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append(float(v))
        agg[f"macro_{k}"] = float(np.mean(vals)) if vals else float("nan")
    return agg
