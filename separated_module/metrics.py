"""Calibration metrics (Expected Calibration Error)."""

import numpy as np


# ----------------------------------------------------------------------
# Calibration diagnostics
# ----------------------------------------------------------------------

def calculate_ece(probs, labels, n_bins=10):
    """
    Computes the Expected Calibration Error (ECE).
    
    ECE bins predictions by confidence and measures the average gap 
    between confidence and actual accuracy within each bin.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float64)
    
    # Assign each confidence to a bin index (1 to n_bins)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_assignments = np.digitize(confidences, bin_edges) - 1
    
    # Handled the exact 1.0 confidence edge case (maps it to the last bin)
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)
    
    ece = 0.0
    n = len(labels)
    
    for i in range(n_bins):
        mask = (bin_assignments == i)
        bin_size = mask.sum()
        
        if bin_size == 0:
            continue
            
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (bin_size / n) * abs(bin_acc - bin_conf)
        
    return float(ece)


def expected_calibration_error(probs, labels, n_bins=10):
    """Expected Calibration Error (ECE) of predicted probabilities.

    Bins samples by confidence and returns the weighted average gap between confidence and accuracy.
    `labels` may be class indices or one-hot.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)

    n, k = probs.shape
    if labels.ndim == 2 and labels.shape[-1] != k:
        print(f"[INFO] labels has {labels.shape[1]} classes, probs has {k}")
        return 1.0

    preds = probs.argmax(axis=1)
    confidences = probs[np.arange(n), preds]

    if labels.ndim == 2:
        correct = labels[np.arange(n), preds].astype(np.float64)
    else:
        correct = (preds == labels).astype(np.float64)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    # right=True gives the standard (lo, hi] bins
    bin_ids = np.clip(np.digitize(confidences, bin_edges, right=True) - 1, 0, n_bins - 1)

    # vectorized per-bin stats, no Python loop
    counts = np.bincount(bin_ids, minlength=n_bins)
    acc_sum = np.bincount(bin_ids, weights=correct, minlength=n_bins)
    conf_sum = np.bincount(bin_ids, weights=confidences, minlength=n_bins)

    nonempty = counts > 0
    gap = np.abs(acc_sum[nonempty] - conf_sum[nonempty]) / n   # = (count/n) * |acc - conf|
    return float(gap.sum())
