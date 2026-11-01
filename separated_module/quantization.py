"""Quantisation-aware-training primitives: QuantConfig, fake weight quantisation and activation observers."""

import numpy as np
from dataclasses import dataclass


# ============================================================================
#  QAT PRIMITIVES  \
#============================================================================

@dataclass
class QuantConfig:
    """Quantisation-aware-training settings: enabled flag, weight/activation bit widths and per-channel option."""
    enabled: bool = True
    w_bits: int = 8
    a_bits: int = 8
    per_channel: bool = False      # weights: one scale per output unit
    act_momentum: float = 0.9      # EMA momentum of the activation range observer


def fake_quant_weight(W, bits, per_channel=False):
    """Symmetric uniform fake-quant. Returns the dequantized weights (float).

    Straight-through estimator: the callers uses the *unquantized* W for the
    gradient (dW is computed from the quantized forward, applied to float W).
    Because the scale is max-absolute, nothing is clipped, so the STE mask is all-ones.
    """
    eps = 1e-11
    qmax = 2 ** (bits - 1) - 1
    axis = 0 if per_channel else None            # W is (in, out): reduce over `in`
    amax = np.max(np.abs(W), axis=axis, keepdims=per_channel)
    scale = np.maximum(amax, eps) / qmax
    q = np.clip(np.round(W / scale), -qmax, qmax)
    return q * scale


class ActObserver:
    """Tracks an EMA of min/max and fake-quantizes asymmetrically to `bits`."""

    def __init__(self, bits, momentum=0.9):
        self.bits = bits
        self.momentum = momentum
        self.lo = None
        self.hi = None

    def reset(self, bits=None):
        """Clear the tracked min/max (optionally changing the bit width)."""
        if bits is not None:
            self.bits = bits
        self.lo = self.hi = None

    def _update(self, x):
        lo, hi = float(x.min()), float(x.max())
        if self.lo is None:
            self.lo, self.hi = lo, hi
        else:
            m = self.momentum
            self.lo = m * self.lo + (1 - m) * lo
            self.hi = m * self.hi + (1 - m) * hi

    def fake_quant(self, x, update, tuned_eps=1e-12):
        """Fake-quantise activations x to `bits` using the tracked EMA min/max.

        When `update` is true (training) the running range is updated first. Zero is always representable.
        """
        if update or self.lo is None:
            self._update(x)
        lo, hi = min(self.lo, 0.0), max(self.hi, 0.0)     # zero must be representable
        levels = 2 ** self.bits - 1
        scale = max(hi - lo, tuned_eps) / levels
        zp = np.round(-lo / scale)
        q = np.clip(np.round(x / scale) + zp, 0, levels)
        out = (q - zp) * scale
        mask = ((x >= lo) & (x <= hi)).astype(x.dtype)     # STE clip mask
        return out, mask
