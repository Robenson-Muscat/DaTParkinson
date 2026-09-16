from sklearn.metrics import roc_auc_score,log_loss
import numpy as np



def DaTClassifierMetrics(y_true,y_prob):

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    auc = roc_auc_score(y_true, y_prob)
    loss = log_loss(y_true, y_prob, labels=[0,1]
    
return {"log_loss": loss,
    "roc_auc": auc}