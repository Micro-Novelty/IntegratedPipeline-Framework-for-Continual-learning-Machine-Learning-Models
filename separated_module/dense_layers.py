"""Dense, QATDense and SoftmaxOutput layers."""

import numpy as np
from .quantization import ActObserver, QuantConfig, fake_quant_weight
from .geometric_weight_shaping import GeometricWeightShaping
from .activations import Activation
from .optimizers import AdamOptimizer


class Dense:
    """
    Fully connected layer with an explicit shape contract.
 
      W : (input_size, output_size)      b : (1, output_size)
      forward : (batch, input_size)  ->  (batch, output_size)
 
    - input_size may be None; it is then inferred (from the previous layer by
      MLP.add / MLP.build, or from the first batch of data).
    - Shapes are checked, never "repaired": a mismatch raises a ValueError that
      names the layer, instead of silently slicing W or x.
    - backward() only computes gradients. Updating weights is done by
      apply_gradients() so every layer gets its own optimizer step.
    - The gradient `da` passed to backward must already be averaged over the
      batch (Loss.softmax_crossentropy_derivative and loss_mse_regression both
      do that), so no further division by batch size happens here.
    """
 
    def __init__(self, input_size=None, output_size=None, seed=0,
                 activation=None, init='auto', lr=0.001):
        if output_size is None or int(output_size) < 1:
            raise ValueError('[>] Dense needs a positive output_size (number of units)')
        self.input_size = None if input_size is None else int(input_size)
        self.output_size = int(output_size)
        self.seed = seed
        self.init = init            # 'auto' | 'he' | 'geometric'
        self.lr = lr
        self.fit_input = False      # MLP sets True on the first layer only
        self._warned_fit = False
 
        self.W = None
        self.b = np.zeros((1, self.output_size))
        self.special_weight = (GeometricWeightShaping(self.input_size, self.output_size)
                               if self.input_size is not None else None)
        self.params_shape = None
        self.params = None
        self.opt = None
        self.grads = None
        self.x = self.z = self.a = None
 
        self.activation_name = activation
        if activation:
            self.activation = getattr(Activation, activation)
            self.activation_derivative = getattr(Activation, activation + '_derivative')
        else:
            self.activation = None
            self.activation_derivative = None
 
    # old pickles (saved before these attributes existed) still load
    def __setstate__(self, state):
        self.__dict__.update(state)
        for k, v in (('init', 'auto'), ('lr', 0.001), ('fit_input', False),
                     ('_warned_fit', False), ('grads', None),
                     ('x', None), ('z', None), ('a', None)):
            self.__dict__.setdefault(k, v)
 
    # ------------------------------------------------------------------ build-----
    def build(self, input_size):
        """Fix the layer's input size and create its GeometricWeightShaping helper; raises if weights already exist with another size."""
        input_size = int(input_size)
        if self.W is not None and self.W.shape[0] != input_size:
            raise ValueError(f'[=] Dense already has weights {self.W.shape}; '
                             f'cannot rebuild with input_size={input_size}')
        self.input_size = input_size
        if self.special_weight is None or self.special_weight.input_size != input_size:
            self.special_weight = GeometricWeightShaping(input_size, self.output_size)
        return self
 
    def _init_weights(self, x):
        rng = np.random.default_rng(self.seed)
        W = rng.normal(0.0, np.sqrt(2.0 / self.input_size),
                        (self.input_size, self.output_size))
        self.W = np.ascontiguousarray(W, dtype=np.float64)
        self._init_optimizer()
 
    def _init_optimizer(self):
        self.params = {'W1': self.W, 'b1': self.b}        # same arrays, updated in place
        self.params_shape = {'W1': self.W.shape, 'b1': self.b.shape}
        self.opt = AdamOptimizer(self.params_shape, lr=self.lr, weight_decay=1e-4)
 
    # ---------------------------------------------------------------- helpers-----------
    def _sanitize_string_chars(self, x):
        if isinstance(x, (str, np.str_)):
            clean_str = str(x).replace('[', '').replace(']', '').replace('...', '').strip()
            x = np.fromstring(clean_str, sep=' ')
        if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.character):
            clean_str = ' '.join(x.astype(str).flatten()).replace('[', '').replace(']', '')
            skip_values = {'...', 'NaN', 'null'}
            x = np.fromiter((v for v in clean_str.split() if v not in skip_values), dtype=float)
        return x
 
    def _prepare_input(self, x):
        x = np.asarray(self._sanitize_string_chars(x), dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1) if self.input_size == 1 else x.reshape(1, -1)
        elif x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        return x
 
    def _fit_features(self, x):
        n = x.shape[1]
        if n == self.input_size:
            return x
        if not self.fit_input:
            raise ValueError(
                f'[>] Dense layer expects {self.input_size} input features but received {n}. '
                f'Its input_size must equal the previous layer\'s output_size.')
        if not self._warned_fit:
            what = 'truncating' if n > self.input_size else 'zero-padding'
            print(f'[⚠️] Dense: input has {n} features but the first layer expects '
                  f'{self.input_size} - {what} This is a harmless warning -> (weights are left untouched).')
            self._warned_fit = True
        if n > self.input_size:
            return x[:, :self.input_size]
        out = np.zeros((x.shape[0], self.input_size))
        out[:, :n] = x
        return out
 
    # ---------------------------------------------------------------- forward--------
    def forward(self, x, perf_score=1.0):
        """Compute activation(x @ W + b), building / initialising the weights lazily on the first call."""
        x = self._prepare_input(x)
        if self.input_size is None:
            self.build(x.shape[1])
        x = self._fit_features(x)
 
        if self.W is None:
            self._init_weights(x)
        else:
            if self.W.shape != (self.input_size, self.output_size):
                raise ValueError(f'[>] Dense weights {self.W.shape} do not match '
                                 f'({self.input_size}, {self.output_size})')
            if self.opt is None:
                self._init_optimizer()
 
        self.x = x
        self.z = x @ self.W + self.b
        self.a = self.activation(self.z) if self.activation else self.z
        return self.a
 
    # --------------------------------------------------------------- backward
    def backward(self, da, lr=None, clip_value=1.0, momentum=0.9):
        """Gradient computation only. Returns (dx, {'W1': dW, 'b1': db})."""
        if self.x is None:
            raise RuntimeError('[=] Dense.backward called before forward')
        eps = 1e-5
        da = np.asarray(da, dtype=np.float64)
        if da.ndim == 1:
            da = da.reshape(1, -1)
        if da.shape != self.z.shape:
            raise ValueError(f'[=] Dense.backward: incoming gradient {da.shape} does not match '
                             f'this layer\'s output {self.z.shape}')
 
        dz = da * self.activation_derivative(self.z) if self.activation_derivative else da
 
        dW = self.x.T @ dz                       # (in, out)
        db = dz.sum(axis=0, keepdims=True)       # (1, out)
        dx = dz @ self.W.T                       # (batch, in) -> previous layer
 
        # global-norm clipping: keeps direction, caps magnitude
        total_norm = np.sqrt(np.sum(dW ** 2) + np.sum(db ** 2))
        if total_norm > clip_value:
            scale = clip_value / (total_norm + eps)
            dW = dW * scale
            db = db * scale
 
        self.grads = {'W1': dW, 'b1': db}
        return dx, self.grads
 
    def parameter_count(self):
        """Number of trainable parameters (weights + bias)."""
        return 0 if self.W is None else int(self.W.size + self.b.size)


class QATDense(Dense):
    """Dense with fake-quantized weights + input activations (STE backward).

    Subclasses your Dense, so MLP.backward / MLP.apply_gradients / AdamOptimizer
    work unchanged. Master weights `self.W` stay float; the optimizer updates them.
    """

    def __init__(self, *args, qcfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.qcfg = qcfg or QuantConfig(enabled=False)
        self.in_obs = ActObserver(self.qcfg.a_bits, self.qcfg.act_momentum)
        self.update_obs = True
        self._Wq = None
        self._xmask = None

    def set_qcfg(self, qcfg):
        """Apply a QuantConfig: stores it and resets the input activation observer to the new bit width / momentum."""
        self.qcfg = qcfg
        self.in_obs.reset(bits=qcfg.a_bits)
        self.in_obs.momentum = qcfg.act_momentum

    def forward(self, x, perf_score=1.0):
        """Forward pass with fake-quantised weights and input activations."""
        x = self._fit_features(self._prepare_input(x))
        if self.W is None:
            self._init_weights(x)
        else:
            if self.W.shape != (self.input_size, self.output_size):
                raise ValueError(f'[>] Dense weights {self.W.shape} do not match '
                                 f'({self.input_size}, {self.output_size})')
            if self.opt is None:
                self._init_optimizer()

        if self.qcfg.enabled:
            x, self._xmask = self.in_obs.fake_quant(x, update=self.update_obs)
            Wq = fake_quant_weight(self.W, self.qcfg.w_bits, self.qcfg.per_channel)
        else:
            self._xmask = None
            Wq = self.W

        self._Wq = Wq
        self.x = x
        self.z = x @ Wq + self.b
        self.a = self.activation(self.z) if self.activation else self.z
        return self.a

    def backward(self, da, lr=None, clip_value=1.0, momentum=0.9):
        """Backward pass using the straight-through estimator: gradients computed through the quantised weights are applied to the float weights."""
        eps = 1e-5
        da = np.asarray(da, dtype=np.float64)
        dz = da * self.activation_derivative(self.z) if self.activation_derivative else da
        dW = self.x.T @ dz      
                         # STE: grad w.r.t. quantized W -> float W
        db = dz.sum(axis=0, keepdims=True)
        dx = dz @ self._Wq.T                     # backprop through the *quantized* weights
        if self._xmask is not None:
            dx = dx * self._xmask                # STE clip mask of the input fake-quant
        total_norm = np.sqrt(np.sum(dW ** 2) + np.sum(db ** 2))
        if total_norm > clip_value:              
            s = clip_value / (total_norm + eps)
            dW, db = dW * s, db * s
        self.grads = {'W1': dW, 'b1': db}
        return dx, self.grads


class SoftmaxOutput:
    """Final softmax layer, combined with cross-entropy so the backward pass is simply the supplied gradient."""
    def forward(self, x):
        """Apply softmax to the logits and cache the output."""
        self.out = Activation.softmax(x)

        return self.out

    def backward(self, dL_dZ):
        """Pass through dL/dZ (softmax + cross-entropy gradient is already w.r.t. the logits)."""
        # gradient already computed as y_pred - y_true
        return dL_dZ
