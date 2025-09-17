from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def compute_eer(scores, labels):
    """
    Compute Equal Error Rate (EER).

    Args:
        scores (torch.Tensor or array-like): Score values.
        labels (list or array-like): Ground truth labels (1: target, 0: non-target).

    Returns:
        float: EER value (in percentage).
    """
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    # Interpolate between FPR and TPR to compute EER
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer * 100


def compute_min_dcf(scores, labels, p_target=0.05, c_miss=1, c_fa=1):
    """
    Compute the minimum Detection Cost Function (DCF).

    Args:
        scores (iterable): Score values.
        labels (iterable): Ground truth labels (1: target, 0: non-target).
        p_target (float): Target prior.
        c_miss (float): Miss cost.
        c_fa (float): False alarm cost.

    Returns:
        float: Minimum DCF value.
    """
    from operator import itemgetter

    # Sort scores along with their indices.
    sorted_items = sorted(enumerate(scores), key=lambda x: x[1])
    sorted_indexes, thresholds = zip(*sorted_items)
    sorted_labels = [labels[i] for i in sorted_indexes]

    fnrs = []
    fprs = []
    for i, label in enumerate(sorted_labels):
        if i == 0:
            fnrs.append(label)
            fprs.append(1 - label)
        else:
            fnrs.append(fnrs[i - 1] + label)
            fprs.append(fprs[i - 1] + (1 - label))

    total_misses = sum(sorted_labels)
    total_correct = len(sorted_labels) - total_misses

    # Compute false negative rates (FNR) and false positive rates (FPR)
    fnrs = [x / float(total_misses) for x in fnrs]
    fprs = [1.0 - (x / float(total_correct)) for x in fprs]

    min_c_det = float("inf")
    for threshold, fnr, fpr in zip(thresholds, fnrs, fprs):
        c_det = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det

    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def

    return min_dcf
