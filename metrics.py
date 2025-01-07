import config
from utils import *
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, confusion_matrix

def f1_score_and_auc_score(y_true, y_proba, threshold=0.5):
    y_pred = [1 if p>=threshold else 0 for p in y_proba]
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[1]==1 or cm[1][1]==0:
        precision, recall, f1 = 0, 0, 0
        auc = None
    else:
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = round(roc_auc_score(y_true, y_pred), 6)
    result = {
        "metric": "f1_score",
        "f1_score": round(f1,6),
        "auc": auc,
        "precision": round(precision,6),
        "recall": round(recall,6),
    }
    return result

_all_evaluators = {
    "default": "f1_score",
    "f1_score": f1_score_and_auc_score,
}
_evaluate = _all_evaluators[_all_evaluators["default"]]

def set_evaluator(name=None):
    global _evaluate
    if name in _all_evaluators:
        _evaluate = _all_evaluators[name]
    return

def evaluate(y_true, y_proba, show_detail=False, alias=""):
    result = _evaluate(y_true, y_proba)
    if show_detail==True: 
        printf("Evaluation%s: "%(" (%s)"%(alias) if len(alias)>0 else ""),\
                        ", ".join(["%s=%s"%(k,v) for k,v in result.items() if k!="metric"]))
    return result[result["metric"]]
