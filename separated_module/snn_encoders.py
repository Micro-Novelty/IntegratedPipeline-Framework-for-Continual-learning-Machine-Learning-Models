"""Spike encoders for the spiking neural network (Poisson, rate, latency)."""

import numpy as np


class PoissonEncoder:
    """
    Stochastic rate coding: at every timestep, neuron i fires with
    probability proportional to its (normalized, in [0,1]) input
    intensity, independently sampled each step -> an inhomogeneous
    Poisson-like spike train. This is the most common SNN input encoding
    and is naturally noise-robust since the same input yields a
    different, but statistically similar, spike train on every forward
    pass.
    """

    def __init__(self, n_steps, max_rate=1.0, rng=None):
        self.n_steps = n_steps
        self.max_rate = max_rate
        self.rng = rng or np.random.default_rng()

    def encode(self, x):
        """x : (B, F) array in [0, 1]. Returns (T, B, F) spikes in {0,1}."""
        x = np.clip(x, 0.0, 1.0) * self.max_rate
        T, B, F = self.n_steps, x.shape[0], x.shape[1]
        draws = self.rng.random((T, B, F))
        return (draws < x[None, :, :]).astype(np.float64)


class RateEncoder:
    """
    Deterministic rate coding: neuron i fires regularly, once every
    round(1/intensity_i) steps, giving a reproducible spike train whose
    average rate over T steps equals the input intensity. Useful when
    determinism / low variance is preferred over Poisson's noise.
    """

    def __init__(self, n_steps):
        self.n_steps = n_steps

    def encode(self, x):
        """Encode intensities in [0, 1] (batch, features) into a (T, batch, features) spike train with regular, deterministic firing."""
        x = np.clip(x, 1e-6, 1.0)
        T, B, F = self.n_steps, x.shape[0], x.shape[1]
        phase = np.cumsum(np.tile(x[None, :, :], (T, 1, 1)), axis=0)
        prev_count = np.floor(phase - x[None, :, :])
        count = np.floor(phase)
        return (count > prev_count).astype(np.float64)


class LatencyEncoder:
    """
    Temporal / latency coding: stronger inputs fire earlier (a single
    spike per neuron), weaker inputs fire later or not at all. Mirrors
    the "time-to-first-spike" code observed in biological sensory
    systems and is very sparse (at most one spike per neuron).
    """

    def __init__(self, n_steps, tau=1.0, threshold=1e-3):
        self.n_steps = n_steps
        self.tau = tau
        self.threshold = threshold

    def encode(self, x):
        """Encode each input as a single spike whose time is earlier for stronger inputs; returns a (T, batch, features) spike train."""
        x = np.clip(x, self.threshold, 1.0)
        T, B, F = self.n_steps, x.shape[0], x.shape[1]
        # Higher intensity -> smaller (earlier) spike time.
        spike_time = (self.tau * np.log(1.0 / x)).astype(np.float64)
        spike_time = np.clip(spike_time, 0, T - 1).astype(int)

        spikes = np.zeros((T, B, F))
        b_idx, f_idx = np.meshgrid(np.arange(B), np.arange(F), indexing="ij")
        spikes[spike_time, b_idx, f_idx] = 1.0
        return spikes
