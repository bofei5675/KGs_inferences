import numpy as np
import os
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

def eval(scores_dir):
    y_preds = []
    y_trues = []

    for fn in os.listdir(scores_dir):
        print(fn)
        y_pred, y_true = np.loadtxt(os.path.join(scores_dir, fn), delimiter=',', usecols=(3, 4), unpack=True)  ##, max_rows=10
        y_preds.extend(y_pred)
        y_trues.extend(y_true)
    y_preds = np.array(y_preds)
    y_trues = np.array(y_trues)
    iso_reg = IsotonicRegression().fit(y_preds, y_trues)
    y_probs = iso_reg.predict(y_preds)
    auc_roc = roc_auc_score(y_trues, y_probs)
    auc_pr = average_precision_score(y_trues, y_probs)

    y_predicts = np.where(y_probs > 0.5, 1.0, 0.0)
    accuracy = accuracy_score(y_trues, y_predicts)
    f1 = f1_score(y_trues, y_predicts)
    print(f'total length is {len(y_trues)} \n auc score for roc is {auc_roc} and the auc for pr is {auc_pr}, f1 score is {f1}, accuracy is {accuracy}')
    res = np.vstack((y_preds, y_trues, y_probs)).T
    np.savetxt('/home/yh1844/inference-2019/eval/eval.txt', res)


def eval_multiclass(scores_dir):
    y_preds = []
    y_trues = []
    class_ids = []

    for fn in os.listdir(scores_dir):
        class_id, y_pred, y_true = np.loadtxt(os.path.join(scores_dir, fn), delimiter=',', usecols=(1, 3, 4),
                                    unpack=True)  ##, max_rows=10
        y_preds.extend(y_pred)
        y_trues.extend(y_true)
        class_ids.extend(class_id)
    y_preds = np.array(y_preds)
    y_trues = np.array(y_trues)
    class_ids = np.array(class_ids)
    auc_roc_avg = []
    auc_pr_avg = []
    for _, ele in np.ndenumerate(np.unique(class_ids)):
        condlist = [class_ids == ele]
        y_pred = y_preds[condlist[0]]
        y_true = y_trues[condlist[0]]
        if np.sum(y_true) == 0:
            print(ele)
            continue
        iso_reg = IsotonicRegression().fit(y_pred, y_true)
        y_prob = iso_reg.predict(y_pred)
        auc_roc = roc_auc_score(y_true, y_prob)
        auc_pr = average_precision_score(y_true, y_prob)
        auc_roc_avg.append(auc_roc)
        auc_pr_avg.append(auc_pr)
    auc_roc_avg = np.average(np.array(auc_roc_avg))
    auc_pr_avg = np.average(np.array(auc_pr_avg))
    print(
        f'total length is {len(y_trues)} \nauc score for roc is {auc_roc_avg} and the auc for pr is {auc_pr_avg}')


scores_dir = '/scratch/bz1030/relationPrediction/checkpoints/deepddi_tanh/output_scores'
eval_multiclass(scores_dir)
