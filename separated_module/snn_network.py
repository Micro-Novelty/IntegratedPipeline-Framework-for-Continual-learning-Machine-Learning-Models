"""SNN loss helpers and the SNNNetwork container (forward / backward / train-eval switching)."""

import numpy as np
from .snn_layers import ReadoutLayer
from .snn_utility import SNNUtility


def softmax_cross_entropy(logits, labels):
    """
    logits : (B, C) ndarray
    labels : (B,) int array of class indices, OR (B, C) one-hot array
    Returns (loss: float, grad_logits: (B, C))
    """
    B = logits.shape[0]
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)

    if labels.ndim == 1:
        one_hot_labels = np.zeros_like(logits)
        one_hot_labels[np.arange(B), labels] = 1.0
    else:
        one_hot_labels = labels

    log_probs = shifted - np.log(exp.sum(axis=1, keepdims=True))
    loss = -np.sum(one_hot_labels * log_probs) / B
    grad_logits = (probs - one_hot_labels) / B
    return loss, grad_logits


def mse_loss(pred, target):
    """pred, target: same shape. Returns (loss, grad_pred)."""
    diff = pred - target
    loss = np.mean(diff * diff)
    grad = 2.0 * diff / diff.size
    return loss, grad


class SNNNetwork:
    """Stack of spiking layers ending in a ReadoutLayer.

    `forward` runs spikes through all layers and collapses the readout membrane trace ('mean', 'last'
    or 'max') into logits; `backward` runs surrogate-gradient BPTT in reverse.
    """
    def __init__(self, layers, readout_reduction="mean", input_noise_std=0.0,
                 rng=None):
        """
        layers : list of LIFDenseLayer / SpikeDropout objects, ending in a
                 ReadoutLayer.
        readout_reduction : {"mean", "last", "max"}
            How the readout's (T, B, C) membrane trace is collapsed into
            (B, C) logits for the loss, and correspondingly how the loss
            gradient is redistributed back across time.
        input_noise_std : float
            Std-dev of Gaussian jitter added to the encoded input spike
            train during training only -- a cheap robustness/regularization
            measure (akin to input-noise data augmentation).
        """
        assert isinstance(layers[-1], ReadoutLayer), "last layer must be a ReadoutLayer"
        self.layers = layers
        self.readout_reduction = readout_reduction
        self.input_noise_std = input_noise_std
        self.rng = rng or np.random.default_rng()
        self.training = True
        self.utility = SNNUtility()

    # ------------------------------------------------------------------
    def train(self):
        """Switch the network and all layers into training mode (enables dropout/noise)."""
        self.training = True
        for l in self.layers:
            if hasattr(l, "training"):
                l.training = True

    def eval(self):
        """Switch the network and all layers into evaluation mode."""
        self.training = False
        for l in self.layers:
            if hasattr(l, "training"):
                l.training = False

    # ------------------------------------------------------------------
    def _reduce_readout(self, mem_hist):
        """mem_hist: (T, B, C) -> logits (B, C)"""
        if self.readout_reduction == "mean":
            return mem_hist.mean(axis=0)
        elif self.readout_reduction == "last":
            return mem_hist[-1]
        elif self.readout_reduction == "max":
            return mem_hist.max(axis=0)
        else:
            raise ValueError(f"[=] unknown readout_reduction {self.readout_reduction}")

    def _expand_grad(self, grad_logits, mem_hist):
        """grad_logits (B, C) -> grad_mem_hist (T, B, C), matching the
        reduction used in _reduce_readout."""
        T = mem_hist.shape[0]
        if self.readout_reduction == "mean":
            return np.tile(grad_logits[None, :, :] / T, (T, 1, 1))
        elif self.readout_reduction == "last":
            g = np.zeros_like(mem_hist)
            g[-1] = grad_logits
            return g
        elif self.readout_reduction == "max":
            g = np.zeros_like(mem_hist)
            idx = np.argmax(mem_hist, axis=0)  # (B, C)
            B, C = idx.shape
            b_idx, c_idx = np.meshgrid(np.arange(B), np.arange(C), indexing="ij")
            g[idx, b_idx, c_idx] = grad_logits
            return g

    # ------------------------------------------------------------------
    def forward(self, input_spikes):
        """input_spikes: (T, B, F_in). Returns logits (B, C)."""
        x = input_spikes
        if self.training and self.input_noise_std > 0:
            x = x + self.rng.normal(0.0, self.input_noise_std, size=x.shape)
        for layer in self.layers[:-1]:
            x = layer.forward(x)
        mem_hist = self.layers[-1].forward(x)
        self._last_mem_hist = mem_hist
        return self._reduce_readout(mem_hist)

    def backward(self, grad_logits):
        """Back-propagate d(loss)/d(logits) through the readout reduction and every layer in reverse order."""
        grad = self._expand_grad(grad_logits, self._last_mem_hist)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    # ------------------------------------------------------------------
    def train_step(self, input_spikes, labels, optimizer, loss_fn=softmax_cross_entropy,
                    max_grad_norm=None, l2_lambda=0.0):
        """One full forward + loss + backward + optimizer.step call."""
        self.train()
        optimizer.zero_grad(self.layers)

        logits = self.forward(input_spikes)
        loss, grad_logits = loss_fn(logits, labels)

        if l2_lambda > 0:
            reg = 0.0
            for layer in self.layers:
                params, _ = layer.parameters()
                if "W" in params:
                    reg += np.sum(params["W"] ** 2)
            loss = loss + 0.5 * l2_lambda * reg

        self.backward(grad_logits)

        if l2_lambda > 0:
            for layer in self.layers:
                params, grads = layer.parameters()
                if "W" in params:
                    grads["W"] += l2_lambda * params["W"]

        if max_grad_norm is not None:
            self.utility.clip_grad_norm(self.layers, max_grad_norm)

        optimizer.step(self.layers)
        return loss, logits

    def predict(self, input_spikes, task='classification'):
        """Run in eval mode and return (predictions, logits): argmax classes, or the first logit column for regression."""
        self.eval()
        logits = self.forward(input_spikes)
        if task == 'regression':
            return logits[:, 0], logits
        return np.argmax(logits, axis=1), logits

    def firing_rates(self):
        """Mean firing rate per layer (for monitoring / homeostasis)."""
        rates = []
        for layer in self.layers[:-1]:
            if hasattr(layer, "_cache") and layer._cache is not None:
                rates.append(layer._cache["spike_hist"].mean())
        return rates
