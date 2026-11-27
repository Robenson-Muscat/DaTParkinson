from sklearn.metrics import roc_auc_score,log_loss, accuracy_score
import numpy as np




def compute_binary_metrics(y_true, probabilities):

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

    probabilities = np.clip(probabilities, 1e-7,1 - 1e-7)
    predictions = (probabilities >= 0.5).astype(int)


    return {

        "AUROC": roc_auc_score(y_true,probabilities),

        "LogLoss": log_loss(y_true, probabilities,labels=[0, 1]),

        "Accuracy": accuracy_score(y_true, predictions)
    }
