"""SNN confidence utilities: softmax probabilities, temperature scaling, Monte-Carlo confidence."""

import numpy as np


# ----------------------------------------------------------------------
# 1. Basic softmax confidence
# ----------------------------------------------------------------------
 
def softmax_probs(logits, temperature=1.0):
    """logits: (B, C) -> probs: (B, C), rows sum to 1."""
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def predict_with_confidence(logits, temperature=1.0, class_names=None):
    """
    logits: (B, C) raw network output.
    Returns a list of per-sample dicts: predicted class (index and/or
    name), its probability, the full probability vector, and the
    distribution's entropy (in nats) as an uncertainty measure -- entropy
    is 0 for a fully confident one-hot prediction and ln(C) for a
    uniform/maximally-uncertain one.
    """
    probs = softmax_probs(logits, temperature)
    B, C = probs.shape
    max_ent = np.log(C)
    results = []
    for i in range(B):
        p = probs[i]
        pred_idx = int(np.argmax(p))
        entropy = float(-np.sum(p * np.log(p + 1e-12)))
        results.append({
            "pred_index": pred_idx,
            "pred_name": class_names.get(pred_idx, "Unknown") if class_names is not None else pred_idx,
            "confidence": float(p[pred_idx]),
            "probs": p,
            "entropy": entropy,
            "entropy_frac": entropy / max_ent,   # 0 (certain) .. 1 (uniform/uncertain)
        })
    return results


# ----------------------------------------------------------------------
# 2. Temperature scaling (post-hoc calibration)
# ----------------------------------------------------------------------
 
class TemperatureScaler:
    """
    Fits a single scalar temperature T > 0 that divides the logits before
    softmax: probs = softmax(logits / T). This is Guo et al. (2017)'s
    temperature scaling -- the simplest, most robust post-hoc calibration
    method, which cannot change *accuracy* (T doesn't change argmax) but
    substantially improves how trustworthy the reported confidence is.
 
    T > 1 softens (reduces) over-confident probabilities (most common case
    for networks trained to near-zero training loss, like a memorizing/
    overfit SNN). T < 1 sharpens under-confident probabilities.
 
    Fit on a held-out VALIDATION set (never the training set -- a network
    is closest to perfectly "calibrated" on data it memorized, which would
    give a useless T ~ 1 even for a badly overconfident model on new data).
    """
 
    def __init__(self):
        self.temperature = 1.0
 
    def fit(self, val_logits, val_labels, lr=0.01, n_steps=200):
        """
        Simple 1-D gradient descent on negative log-likelihood w.r.t. T,
        implemented directly (no autograd needed for a single scalar).
        """
        T = 1.0
        B = val_logits.shape[0]
        labels = np.asarray(val_labels)
 
        for _ in range(n_steps):
            probs = softmax_probs(val_logits, temperature=T)
            # d(NLL)/dT for softmax(logits/T): see Guo et al. derivation;
            # computed here via finite difference for simplicity/robustness.
            eps = 1e-3
            nll_plus = self._nll(val_logits, labels, T + eps)
            nll_minus = self._nll(val_logits, labels, max(T - eps, 1e-3))
            grad = (nll_plus - nll_minus) / (2 * eps)
            T = max(T - lr * grad, 1e-3)
 
        self.temperature = float(T)
        return self.temperature
 
    @staticmethod
    def _nll(logits, labels, T):
        probs = softmax_probs(logits, temperature=T)
        B = logits.shape[0]
        return -np.mean(np.log(probs[np.arange(B), labels] + 1e-12))
 
    def apply(self, logits):
        """Return softmax probabilities of `logits` divided by the fitted temperature."""
        return softmax_probs(logits, temperature=self.temperature)


# ----------------------------------------------------------------------
# SNN-specific: Monte-Carlo confidence via encoding/dropout stochasticity
# ----------------------------------------------------------------------
 
def mc_confidence(net, encoder, x, n_samples=20, temperature=1.0,
                   enable_dropout=True, class_names=None):
    """
    Unlike a standard feed-forward net, an SNN fed by a stochastic encoder
    (e.g. PoissonEncoder) produces a *different* spike train every time
    you re-encode the same input. Re-running the forward pass several
    times and averaging gives you:
      - a smoother, more reliable mean confidence than any single
        stochastic pass (the single-pass confidence is noisier than it
        looks, since it depends on which random spike train was drawn)
      - a genuine uncertainty estimate: the spread (std) of predicted
        probabilities across passes tells you how sensitive the
        prediction is to encoding noise -- a proxy for how "on the edge"
        this input is, similar in spirit to Monte-Carlo Dropout
        (Gal & Ghahramani, 2016).
 
    x : (B, F) static input (same array fed to `encoder.encode()`).
    Returns a list of per-sample dicts with mean confidence, std across
    samples, mean entropy, and the full per-pass probability array.
    """
    was_training = net.training
    if enable_dropout:
        net.train()   # keep SpikeDropout active for extra MC variability
    else:
        net.eval()
 
    all_probs = []
    for _ in range(n_samples):
        spikes = encoder.encode(x)
        logits = net.forward(spikes)
        all_probs.append(softmax_probs(logits, temperature))
    all_probs = np.stack(all_probs, axis=0)   # (n_samples, B, C)
 
    if was_training:
        net.train()
    else:
        net.eval()
 
    mean_probs = all_probs.mean(axis=0)        # (B, C)
    std_probs = all_probs.std(axis=0)          # (B, C)
    B, C = mean_probs.shape
    max_ent = np.log(C)
 
    results = []
    for i in range(B):
        p = mean_probs[i]
        pred_idx = int(np.argmax(p))
        entropy = float(-np.sum(p * np.log(p + 1e-12)))
        results.append({
            "pred_index": pred_idx,
            "pred_name": class_names.get(pred_idx, 'Unknown') if class_names is not None else pred_idx,
            "confidence": float(p[pred_idx]),
            "confidence_std": float(std_probs[i, pred_idx]),
            "probs": p,
            "probs_std": std_probs[i],
            "entropy": entropy,
            "entropy_frac": entropy / max_ent,
            "n_samples": n_samples,
        })
    return results


# ----------------------------------------------------------------------
# Printing
# ----------------------------------------------------------------------
 
def print_confidence_report(results, top_k=3, max_print_threshold=50):
    """readable printout for a list of dicts from
    predict_with_confidence() or mc_confidence()."""
    pred_cache = []
    for i, r in enumerate(results):
        order = np.argsort(-r["probs"])[:top_k]
        top_str = ", ".join(f"{cls}:{r['probs'][cls]*100:.1f}%" for cls in order)
        std_str = f"  (±{r['confidence_std']*100:.1f}%)" if "confidence_std" in r else ""
        pred_cache.append(r['pred_name'])
        if i <= max_print_threshold:
            print(f"[>] Sample {i:3d} -> predicted={r['pred_name']}  "
                f"|| Confidence={r['confidence']*100:.1f}%{std_str}  "
                f"|| Entropy={r['entropy_frac']*100:.0f}%  "
                f"|| Top-{top_k}: [{top_str}]")
    return pred_cache
