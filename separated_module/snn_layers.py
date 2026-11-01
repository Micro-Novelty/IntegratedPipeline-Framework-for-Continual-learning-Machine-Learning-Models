"""Spiking-network building blocks: surrogate spike function, LIF layer, readout, dropout, timing memory, STDP, homeostasis."""

import numpy as np
from .hnsw import HNSW


class HeaviSide:
    """Spike non-linearity: hard Heaviside step in the forward pass with smooth surrogate gradients for back-propagation."""
    def __init__(self, eps=1e-5):
        self.eps = eps


    def heaviside(self, x):
        """Forward spike nonlinearity: 1 where x >= 0, else 0."""
        return (x >= 0.0).astype(np.float64)


    def fast_sigmoid_surrogate(self, x, k=25.0):
        """
        Derivative surrogate based on the "fast sigmoid" sigma(x) = x / (1+k|x|).
        d/dx sigma(x) = 1 / (1 + k|x|)^2

        x is (membrane potential - threshold). k controls steepness: larger k
        -> sharper surrogate, closer to the true (but non-differentiable) step.
        """
        denom = (1.0 + k * np.abs(x))
        return 1.0 / (denom * denom) + self.eps


    def atan_surrogate(self, x, alpha=2.0):
        """
        Derivative surrogate based on the arctangent, used e.g. in SpikingJelly.
        d/dx [ (1/pi) atan(pi/2 * alpha * x) + 0.5 ] = alpha / (2 * (1 + (pi/2 * alpha * x)^2))
        """
        v = (np.pi / 2.0) * alpha * x
        return alpha / (2.0 * (1.0 + v * v) + self.eps)


class LIFDenseLayer:
    """
    Fully-connected layer of Leaky Integrate-and-Fire spiking neurons.

    Parameters
    ----------
    in_features, out_features : int
    decay : float
        Membrane leak factor beta in (0, 1). beta = exp(-dt/tau_mem).
    threshold : float
        Firing threshold v_th.
    surrogate_fn : callable(x, **kwargs) -> array
        Surrogate derivative of the spike function, evaluated at
        (v[t] - v_th).
    surrogate_kwargs : dict
    weight_scale : float
        Multiplier on the default fan-in-scaled initialization.
    use_bias : bool
    """

    def __init__(self, in_features, out_features, decay=0.9, threshold=1.0
                 , surrogate_kwargs=None,
                 weight_scale=1.0, use_bias=True, rng=None):
        self.in_features = in_features
        self.out_features = out_features
        self.decay = float(decay)
        self.threshold = float(threshold)
        self.heaviside = HeaviSide()
        self.surrogate_fn = self.heaviside.fast_sigmoid_surrogate
        self.surrogate_kwargs = surrogate_kwargs or {}
        self.use_bias = use_bias

        rng = rng or np.random.default_rng()
        # Fan-in scaled uniform init, standard for SNN synapses
        # (keeps membrane potentials in a reasonable dynamic range at t=0).
        limit = weight_scale * np.sqrt(1.0 / in_features)
        self.W = rng.uniform(-limit, limit, size=(in_features, out_features))
        self.b = np.zeros(out_features) if use_bias else None

        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b) if use_bias else None

        self._cache = None
        self.heaviside = HeaviSide()

    def forward(self, input_spikes, record_rate=False):
        """
        input_spikes : ndarray (T, B, in_features), values in {0,1} (or
                       real-valued currents if used as the first layer).
        returns spikes : ndarray (T, B, out_features)
        """
        T, B, _ = input_spikes.shape
        F_out = self.out_features

        mem = np.zeros((B, F_out))
        prev_spike = np.zeros((B, F_out))

        mem_hist = np.empty((T, B, F_out))
        spike_hist = np.empty((T, B, F_out))

        for t in range(T):
            I_t = input_spikes[t] @ self.W
            if self.use_bias:
                I_t = I_t + self.b
            mem = self.decay * mem + I_t - prev_spike * self.threshold
            spike = self.heaviside.heaviside(mem - self.threshold)

            mem_hist[t] = mem
            spike_hist[t] = spike
            prev_spike = spike

        self._cache = {
            "input_spikes": input_spikes,
            "mem_hist": mem_hist,
            "spike_hist": spike_hist,
        }
        if record_rate:
            self.last_firing_rate = spike_hist.mean(axis=(0, 1))
        return spike_hist


    def backward(self, grad_output):
        """
        grad_output : ndarray (T, B, out_features) -- dL/d(spike train)
        returns grad_input : ndarray (T, B, in_features) -- dL/d(input spike train)
        Accumulates self.grad_W, self.grad_b.
        """
        cache = self._cache
        input_spikes = cache["input_spikes"]
        mem_hist = cache["mem_hist"]
        T, B, F_out = grad_output.shape

        self.grad_W = np.zeros_like(self.W)
        if self.use_bias:
            self.grad_b = np.zeros_like(self.b)
        grad_input = np.empty_like(input_spikes)

        grad_mem_future = np.zeros((B, F_out))  # dL/dmem[t+1], starts at 0 for t=T-1

        for t in range(T - 1, -1, -1):
            # Gradient reaching spike[t] from: (a) external/next-layer grad,
            # (b) the reset term in mem[t+1] = ... - spike[t]*v_th
            grad_spike_t = grad_output[t] - grad_mem_future * self.threshold

            sg = self.surrogate_fn(mem_hist[t] - self.threshold, **self.surrogate_kwargs)
            grad_mem_t = grad_spike_t * sg + grad_mem_future * self.decay

            # mem[t] = decay*mem[t-1] + I[t] - spike[t-1]*v_th  =>  dmem[t]/dI[t] = 1
            grad_I_t = grad_mem_t

            grad_input[t] = grad_I_t @ self.W.T
            self.grad_W += input_spikes[t].T @ grad_I_t
            if self.use_bias:
                self.grad_b += grad_I_t.sum(axis=0)

            grad_mem_future = grad_mem_t

        # average weight grads over batch handled by caller/optimizer via batch size scaling
        return grad_input


    def parameters(self):
        """Return (params, grads) dicts for the layer: weights 'W' and, if enabled, bias 'b'."""
        params = {"W": self.W}
        grads = {"W": self.grad_W}
        if self.use_bias:
            params["b"] = self.b
            grads["b"] = self.grad_b
        return params, grads


class ReadoutLayer:
    """
    Non-spiking leaky-integrator output layer. Used as the final layer of
    a classifier: it accumulates synaptic current into a membrane trace
    without ever thresholding/spiking, avoiding the extra discretization
    noise a spiking readout would add to the loss.

        v[t] = decay * v[t-1] + s_in[t] @ W + b

    The full membrane trace v (T, B, out_features) is returned; how it is
    reduced to logits (mean / last / max over time) is decided by the
    loss / network layer, since that determines how gradients are routed
    back in through `backward`.
    """

    def __init__(self, in_features, out_features, decay=0.9,
                 weight_scale=1.0, use_bias=True, rng=None,
                 memory_min_neighbors=3.0):
        self.in_features = in_features
        self.out_features = out_features
        self.decay = float(decay)
        self.use_bias = use_bias

        rng = rng or np.random.default_rng()
        limit = weight_scale * np.sqrt(1.0 / in_features)
        self.W = rng.uniform(-limit, limit, size=(in_features, out_features))
        self.b = np.zeros(out_features) if use_bias else None

        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b) if use_bias else None
        self._cache = None
        
    
    
    def forward(self, input_spikes):
        """Integrate input spikes (T, B, F) through a leaky membrane with no spiking; returns the membrane trace (T, B, out_features)."""
        T, B, _ = input_spikes.shape
        F_out = self.out_features
        mem = np.zeros((B, F_out))
        mem_hist = np.empty((T, B, F_out))
        for t in range(T):
            I_t = input_spikes[t] @ self.W

            if self.use_bias:
                I_t = I_t + self.b

            mem = self.decay * mem + I_t
            mem_hist[t] = mem
        self._cache = {"input_spikes": input_spikes}

        return mem_hist

    def backward(self, grad_mem_hist):
        """
        grad_mem_hist : (T, B, out_features) dL/dv[t] for every t (caller
        decides how loss gradient is distributed across time, e.g. only
        at t=T-1, or evenly if using a mean readout).
        """
        input_spikes = self._cache["input_spikes"]
        T, B, F_out = grad_mem_hist.shape

        self.grad_W = np.zeros_like(self.W)
        if self.use_bias:
            self.grad_b = np.zeros_like(self.b)
        grad_input = np.empty_like(input_spikes)

        grad_mem_future = np.zeros((B, F_out))
        for t in range(T - 1, -1, -1):
            grad_mem_t = grad_mem_hist[t] + grad_mem_future * self.decay
            grad_I_t = grad_mem_t
            grad_input[t] = grad_I_t @ self.W.T
            self.grad_W += input_spikes[t].T @ grad_I_t
            if self.use_bias:
                self.grad_b += grad_I_t.sum(axis=0)
            grad_mem_future = grad_mem_t

        return grad_input

    def parameters(self):
        """Return (params, grads) dicts for the readout: weights 'W' and, if enabled, bias 'b'."""
        params = {"W": self.W}
        grads = {"W": self.grad_W}
        if self.use_bias:
            params["b"] = self.b
            grads["b"] = self.grad_b
        return params, grads


class SpikeDropout:
    """
    "Locked"/variational dropout for spike trains: a single Bernoulli mask
    per neuron is sampled once per forward pass and applied identically at
    every timestep (standard practice for RNN-like models -- sampling a
    fresh mask per timestep destroys temporal correlations the SNN relies
    on). Surviving units are scaled by 1/(1-p) (inverted dropout).

    No learnable parameters; `backward` just re-applies the same mask.
    """

    def __init__(self, p=0.2, rng=None):
        assert 0.0 <= p < 1.0
        self.p = p
        self.rng = rng or np.random.default_rng()
        self._mask = None
        self.training = True

    def forward(self, input_spikes):
        """Apply the shared mask during training (scaled by 1/(1-p)); identity in eval mode or when p == 0."""
        if not self.training or self.p == 0.0:
            self._mask = None
            return input_spikes
        _, B, F = input_spikes.shape
        keep = (self.rng.random((1, B, F)) >= self.p).astype(np.float64)
        self._mask = keep / (1.0 - self.p)
        return input_spikes * self._mask

    def backward(self, grad_output):
        """Route gradients through the same mask used in the forward pass."""
        if self._mask is None:
            return grad_output
        return grad_output * self._mask

    def parameters(self):
        """Dropout has no trainable parameters; returns empty dicts."""
        return {}, {}


class PerSpikeTimingMemory:
    """STDP-style synaptic traces backed by an HNSW memory.

    Stores post-synaptic trace vectors per neuron and retrieves similar past traces so STDP weight
    updates can be informed by earlier spike-timing patterns.
    """
    def __init__(self, a_plus=0.01, a_minus=0.012, tau_pre=20.0, tau_post=20.0,
                 w_min=0.0, w_max=1.0, dt=1.0, 
                 K=8, capacity=8000, k=8, metric='euclidean', evict_fraction=0.1):
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.decay_pre = np.exp(-dt / tau_pre)
        self.decay_post = np.exp(-dt / tau_post)
        self.w_min = w_min
        self.w_max = w_max

        self.memory_k = k
        self.mem_capacity = capacity
        self.evict_batch = max(1, int(capacity * evict_fraction))
        self.metric = metric

        self.indices = None
        self._values = None
        self._count = None
        self.dim = None

    def write(self, trace_pre, trace_post):
        """Insert the current pre/post traces for each batch element into the HNSW indices (until capacity)."""
        B, F_in = trace_pre.shape
        for b in range(B): 
            if self._count[b] >= self.mem_capacity:
                continue
            
            if b < len(self.indices):
                self.indices[b].insert(trace_post[b])
                self._count[b] += 1
            if b < len(self.indices):
                self._values[b].append(np.array(trace_pre[b], copy=True))
            
            if self._count[b] >= self.mem_capacity:
                self._evict_and_rebuild(b)

    def _evict_and_rebuild(self, b):
        keep_from = self.evict_batch
        surviving_vectors = self.indices[b].vectors[keep_from:]
        surviving_values = self._values[b][keep_from:]

        fresh_index = HNSW(dim=self.dim, M=self.indices[b].M,
                            ef_construction=self.indices[b].ef_construction,
                            metric=self.metric)
        for vec in surviving_vectors:
            fresh_index.insert(vec)

        self.indices[b] = fresh_index
        self._values[b] = surviving_values
        self._count[b] = len(surviving_values)
            

    def retrieve(self, W, pre_spikes, post_spikes):
        """Look up the nearest stored traces for the current spike activity and return the memory traces plus a validity mask."""
        T, B, F_in = pre_spikes.shape
        F_out = post_spikes.shape[2]
        k = self.memory_k
        
        dim = (B * F_out) 
        if not self.indices:
            self.indices = [HNSW(dim=dim, M=24, ef_construction=264, metric=self.metric) for _ in range(dim)]
            self._values = [[] for _ in range(dim)]
            self._count = [0] * dim
            self.dim = dim

        trace_pre = np.zeros((B, k, F_in))
        trace_post = np.zeros((B, k, F_out))
        mem_mask = np.zeros((B, k, F_out))

        for b in range(B):
            if self._count[b] == 0:
                continue
            results = self.indices[b].search(W[b], k=k, ef=max(k, 50))
            for j, (node_id, _dist) in enumerate(results):
                trace_post[b, j] = self.indices[b].vectors[node_id]
                if node_id < len(self._values[b]):
                    trace_pre[b, j] = self._values[b][node_id]
                mem_mask[b, j] = 1.0

        return trace_pre, trace_post, mem_mask


class STDPLearner:
    """Unsupervised Spike-Timing-Dependent Plasticity learner.

    Potentiates a synapse when the pre-synaptic spike precedes the post-synaptic one (a_plus) and
    depresses it otherwise (a_minus), keeping weights within [w_min, w_max].
    """
    def __init__(self, in_features, out_features, a_plus=0.01, a_minus=0.012,
                 tau_pre=20.0, tau_post=20.0, w_min=0.0, w_max=1.0, dt=1.0):
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.decay_pre = np.exp(-dt / tau_pre)
        self.decay_post = np.exp(-dt / tau_post)
        self.w_min = w_min
        self.w_max = w_max


    def memory_init(self, ind, pre_spikes, post_spikes, tr_pre, tr_post,
                    trace_post_mem, trace_pre_mem, mem_mask):
        """Blend retrieved memory traces (scored by similarity to the current traces) into the STDP trace terms."""
        d_B = pre_spikes.shape[-1]
        scale = 1.0 / np.sqrt(d_B)

        pre_logit_growth = 1.0 / (1.0 + np.exp(-tr_pre))
        post_logit_growth = 1.0 / (1.0 + np.exp(-tr_post))
        trans_pre_val = trace_pre_mem * pre_logit_growth[:, np.newaxis, :]
        trans_post_val  = trace_post_mem * post_logit_growth[:, np.newaxis, :]
        
        pre_scores = np.where(mem_mask > 0, trans_pre_val, -1e9)
        pre_scores = pre_scores - pre_scores.max(axis=-1, keepdims=True)
        post_scores = np.where(mem_mask > 0, trans_post_val, -1e9)
        post_scores = post_scores - post_scores.max(axis=-1, keepdims=True)
        
        pre_exp_scores = np.exp(pre_scores) * mem_mask
        pre_weights = (pre_exp_scores / (pre_exp_scores.sum(axis=-1, keepdims=True) + 1e-8)).sum(axis=1)
        post_exp_scores = np.exp(post_scores) * mem_mask
        post_weights = (post_exp_scores / (post_exp_scores.sum(axis=-1, keepdims=True) + 1e-8)).sum(axis=1)

        trace_pre = self.decay_pre * pre_weights + pre_spikes[ind]
        trace_post = self.decay_post * post_weights + post_spikes[ind]

        return trace_pre, trace_post



    def apply(self, W, pre_spikes, post_spikes):
        """
        W : (in_features, out_features) ndarray, updated in place.
        pre_spikes  : (T, B, in_features)
        post_spikes : (T, B, out_features)
        Returns the total weight delta applied (for logging).
        """
        T, B, F_in = pre_spikes.shape
        F_out = post_spikes.shape[2]

        trace_pre = np.zeros((B, F_in))
        trace_post = np.zeros((B, F_out))
        dW = np.zeros_like(W)

        for t in range(T):
            trace_pre = self.decay_pre * trace_pre + pre_spikes[t]
            trace_post = self.decay_post * trace_post + post_spikes[t]

            trace_pre_mem, trace_post_mem, mem_mask = self.memory_spike.retrieve(W, pre_spikes, post_spikes)
            trace_pre_out, trace_post_out = self.memory_init(t, pre_spikes, post_spikes,
                    trace_pre, trace_post,
                    trace_post_mem, trace_pre_mem, mem_mask)

            # potentiation: postsynaptic spike now, weighted by recent presynaptic trace
            dW += self.a_plus * (trace_pre_out.T @ post_spikes[t])
            # depression: presynaptic spike now, weighted by recent postsynaptic trace
            dW -= self.a_minus * (pre_spikes[t].T @ trace_post_out)

        W += dW / B
        np.clip(W, self.w_min, self.w_max, out=W)
        if self.memory_write_enabled:
            self.memory_spike.write(trace_pre, trace_post)
        return dW / B


class ThresholdHomeostasis:
    """
    Homeostatic threshold controller for LIFDenseLayer instances.

    After each epoch (or N steps), call `.step(layers)` with the same
    layer list used by the network. Every LIFDenseLayer's scalar
    `threshold` is nudged multiplicatively toward keeping its recorded
    mean firing rate close to `target_rate`:

        if rate too high -> raise threshold (neuron fires less easily)
        if rate too low  -> lower threshold (neuron fires more easily)

    This is a coarse, layer-wide analogue of intrinsic plasticity /
    synaptic scaling seen in biological homeostasis, and is useful as a
    cheap robustness measure against dead or saturated layers, especially
    early in training or under distribution shift.
    """

    def __init__(self, target_rate=0.1, adjust_rate=0.05, min_threshold=0.1,
                 max_threshold=10.0):
        self.target_rate = target_rate
        self.adjust_rate = adjust_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def step(self, layers):
        """Nudge each LIF layer's firing threshold up or down so its mean firing rate moves toward `target_rate`."""
        for layer in layers:
            if not isinstance(layer, LIFDenseLayer):
                continue
            if layer._cache is None:
                continue
            rate = layer._cache["spike_hist"].mean()
            error = rate - self.target_rate  # >0: firing too much
            layer.threshold *= (1.0 + self.adjust_rate * np.sign(error) * min(abs(error) / max(self.target_rate, 1e-6), 1.0))
            layer.threshold = float(np.clip(layer.threshold, self.min_threshold, self.max_threshold))
