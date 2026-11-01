"""SNN diagnostics: firing/weight/gradient statistics, health checks, reports and DiagnosticsTracker."""

import numpy as np
from .snn_layers import LIFDenseLayer, ReadoutLayer


def layer_firing_stats(layer, dead_thresh=1e-4, saturated_thresh=0.99):
    """
    Per-neuron firing-rate breakdown for one LIFDenseLayer, using the spike
    history recorded during its most recent forward() call.
 
    Returns a dict with mean/min/max/std rate, per-neuron rate array, and
    counts of "dead" (never fires) and "saturated" (fires almost every
    step) neurons -- the two failure modes discussed for LIF layers.
    """
    if not isinstance(layer, LIFDenseLayer):
        raise TypeError("layer_firing_stats expects a LIFDenseLayer")
    if layer._cache is None:
        raise RuntimeError("layer has no forward-pass cache yet; call forward() first")
 
    spikes = layer._cache["spike_hist"]              # (T, B, F_out)
    per_neuron_rate = spikes.mean(axis=(0, 1))        # (F_out,)
 
    return {
        "mean_rate": float(per_neuron_rate.mean()),
        "min_rate": float(per_neuron_rate.min()),
        "max_rate": float(per_neuron_rate.max()),
        "std_rate": float(per_neuron_rate.std()),
        "per_neuron_rate": per_neuron_rate,
        "n_neurons": per_neuron_rate.shape[0],
        "n_dead": int(np.sum(per_neuron_rate < dead_thresh)),
        "n_saturated": int(np.sum(per_neuron_rate > saturated_thresh)),
        "threshold": float(layer.threshold),
    }


def weight_stats(layer):
    """Basic weight-matrix health: mean, std, L2 norm, and max abs value."""
    W = layer.W
    return {
        "mean": float(W.mean()),
        "std": float(W.std()),
        "norm": float(np.linalg.norm(W)),
        "max_abs": float(np.max(np.abs(W))),
    }


def grad_norm_stats(layer):
    """L2 norm of the most recently computed weight gradient (call after
    backward(), before the optimizer's zero_grad() clears it)."""
    if getattr(layer, "grad_W", None) is None:
        return {"grad_W_norm": None}
    out = {"grad_W_norm": float(np.linalg.norm(layer.grad_W))}
    if getattr(layer, "grad_b", None) is not None:
        out["grad_b_norm"] = float(np.linalg.norm(layer.grad_b))
    return out


def network_report(net, dead_thresh=1e-4, saturated_thresh=0.99):
    """
    One-shot health snapshot across every layer in an SNNNetwork, right
    after a forward()/backward() call. Returns a list of per-layer dicts
    (LIFDenseLayer entries include firing stats; ReadoutLayer entries only
    include weight/grad stats, since it never spikes).
    """
    report = []
    for i, layer in enumerate(net.layers):
        entry = {"index": i, "type": type(layer).__name__}
        if isinstance(layer, LIFDenseLayer):
            entry.update(layer_firing_stats(layer, dead_thresh, saturated_thresh))
            entry.update(weight_stats(layer))
            entry.update(grad_norm_stats(layer))
        elif isinstance(layer, ReadoutLayer):
            entry.update(weight_stats(layer))
            entry.update(grad_norm_stats(layer))
        else:
            continue
        report.append(entry)
    return report


def print_report(report):
    """Pretty-print the output of network_report()."""
    for entry in report:
        if "mean_rate" in entry:
            print(f"[{entry['index']}] {entry['type']:14s} "
                  f"rate(mean/min/max/std)={entry['mean_rate']:.3f}/"
                  f"{entry['min_rate']:.3f}/{entry['max_rate']:.3f}/{entry['std_rate']:.3f}  "
                  f"dead={entry['n_dead']}/{entry['n_neurons']}  "
                  f"saturated={entry['n_saturated']}/{entry['n_neurons']}  "
                  f"thresh={entry['threshold']:.3f}  "
                  f"|W|={entry['norm']:.3f}  |gW|={entry['grad_W_norm']}")
        else:
            print(f"[{entry['index']}] {entry['type']:14s} "
                  f"|W|={entry['norm']:.3f}  |gW|={entry['grad_W_norm']}")


def check_health(report, target_rate=0.15, rate_tolerance=0.5, max_weight_norm=None):
    """
    Turns a network_report() into a list of plain-English warnings, e.g.
    dead/saturated neurons, firing rate far from a target, or weight/grad
    norms that look like they're exploding. Returns [] if nothing stands out.
    """
    warnings = []
    for entry in report:
        tag = f"layer {entry['index']} ({entry['type']})"
        if "n_dead" in entry and entry["n_dead"] > 0:
            warnings.append(f"{tag}: {entry['n_dead']}/{entry['n_neurons']} neurons are dead (never fire)")
        if "n_saturated" in entry and entry["n_saturated"] > 0:
            warnings.append(f"{tag}: {entry['n_saturated']}/{entry['n_neurons']} neurons are saturated (fire almost every step)")
        if "mean_rate" in entry:
            lo, hi = target_rate * (1 - rate_tolerance), target_rate * (1 + rate_tolerance)
            if not (lo <= entry["mean_rate"] <= hi):
                warnings.append(f"{tag}: mean firing rate {entry['mean_rate']:.3f} is outside "
                                 f"[{lo:.3f}, {hi:.3f}] around target {target_rate}")
        if entry.get("grad_W_norm") and not np.isfinite(entry["grad_W_norm"]):
            warnings.append(f"{tag}: grad_W contains NaN/Inf")
        if max_weight_norm is not None and entry["norm"] > max_weight_norm:
            warnings.append(f"{tag}: weight norm {entry['norm']:.2f} exceeds {max_weight_norm}")
    return warnings


# ----------------------------------------------------------------------
# Multi-epoch tracking for SNN
# ----------------------------------------------------------------------
 
class DiagnosticsTracker:
    """
    Call `.record(net, epoch, loss=..., test_acc=...)` once per epoch (right
    after a forward/backward pass, before zero_grad) to accumulate a
    per-epoch, per-layer history. `.summary()` prints a compact trajectory
    table; `.history` gives raw arrays for custom plotting.
    """
 
    def __init__(self):
        self.epochs = []
        self.loss = []
        self.test_acc = []
        self.per_layer = {}   # layer_index -> dict of lists
 
    def record(self, net, epoch, loss=None, test_acc=None):
        """Record loss / test accuracy for an epoch and append the per-layer firing, weight and gradient statistics from `network_report(net)`."""
        self.epochs.append(epoch)
        self.loss.append(loss)
        self.test_acc.append(test_acc)
 
        report = network_report(net)
        for entry in report:
            idx = entry["index"]
            bucket = self.per_layer.setdefault(idx, {
                "type": entry["type"], "mean_rate": [], "threshold": [],
                "weight_norm": [], "grad_norm": [], "n_dead": [], "n_saturated": [],
            })
            bucket["mean_rate"].append(entry.get("mean_rate"))
            bucket["threshold"].append(entry.get("threshold"))
            bucket["weight_norm"].append(entry["norm"])
            bucket["grad_norm"].append(entry.get("grad_W_norm"))
            bucket["n_dead"].append(entry.get("n_dead"))
            bucket["n_saturated"].append(entry.get("n_saturated"))
 
    def summary(self, last_n=5):
        """Print loss/accuracy plus the last `last_n` epochs of per-layer
        firing rate & threshold, so you can see the trend at a glance."""
        print(f"[=] Epochs recorded: {len(self.epochs)}")
        if self.loss and self.loss[-1] is not None:
            print(f"[=] loss: {self.loss[0]:.4f} -> {self.loss[-1]:.4f}")
        if self.test_acc and self.test_acc[-1] is not None:
            print(f"[=] test_acc: {self.test_acc[0]*100:.1f}% -> {self.test_acc[-1]*100:.1f}%")
        for idx, bucket in self.per_layer.items():
            if bucket["type"] != "LIFDenseLayer":
                continue
            rates = [r for r in bucket["mean_rate"][-last_n:] if r is not None]
            thresh = [t for t in bucket["threshold"][-last_n:] if t is not None]
            print(f"[=] layer {idx} last {len(rates)} epochs -- "
                  f"rate: {['%.3f' % r for r in rates]}  "
                  f"threshold: {['%.3f' % t for t in thresh]}")
