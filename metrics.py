from sklearn.metrics import roc_auc_score,log_loss
import numpy as np



def DaTClassifierMetrics(y_true, y_prob):
    """
    Compute classification metrics for a binary DaTScan classifier.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels (0 or 1).

    y_prob : array-like
        Predicted probabilities for the positive class (class 1).
        Values are expected to be in the range [0, 1].

    Returns
    -------
    dict
        Dictionary containing:
        - "log_loss": Binary cross-entropy (logarithmic loss).
        - "roc_auc": Area Under the Receiver Operating Characteristic Curve.
    """

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    auc = roc_auc_score(
        y_true,
        y_prob
    )

    loss = log_loss(
        y_true,
        y_prob,
        labels=[0, 1]
    )

    return {
        "log_loss": loss,
        "roc_auc": auc
    }
    
