"""Optimisers: SGD, Adam and AdamOptimizer."""

import numpy as np


class SGD:
    """Stochastic gradient descent with optional momentum and weight decay for layers exposing `parameters()`."""
    def __init__(self, lr=1e-2, momentum=0.0, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity = {}

    def step(self, layers):
        """Apply one SGD update to every parameter of the given layers."""
        for layer in layers:
            params, grads = layer.parameters()
            for name, p in params.items():
                g = grads[name]
                if g is None:
                    continue
                if self.weight_decay:
                    g = g + self.weight_decay * p
                key = (id(layer), name)
                if self.momentum:
                    v = self._velocity.get(key, np.zeros_like(p))
                    v = self.momentum * v - self.lr * g
                    self._velocity[key] = v
                    p += v
                else:
                    p -= self.lr * g

    def zero_grad(self, layers):
        """Reset the gradient buffers of all given layers to zero."""
        for layer in layers:
            _, grads = layer.parameters()
            for name in grads:
                grads[name][...] = 0.0


class Adam:
    """Standard Adam (Kingma & Ba, 2015), implemented directly in NumPy."""

    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = {}
        self._v = {}
        self._t = {}

    def step(self, layers):
        """Apply one Adam update (bias-corrected moments) to every parameter of the given layers."""
        for layer in layers:
            params, grads = layer.parameters()
            for name, p in params.items():
                g = grads[name]
                if g is None:
                    continue
                if self.weight_decay:
                    g = g + self.weight_decay * p
                key = (id(layer), name)
                m = self._m.get(key, np.zeros_like(p))
                v = self._v.get(key, np.zeros_like(p))
                t = self._t.get(key, 0) + 1

                m = self.beta1 * m + (1 - self.beta1) * g
                v = self.beta2 * v + (1 - self.beta2) * (g * g)
                self._m[key], self._v[key], self._t[key] = m, v, t

                m_hat = m / (1 - self.beta1 ** t)
                v_hat = v / (1 - self.beta2 ** t)
                p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self, layers):
        """Reset the gradient buffers of all given layers to zero."""
        for layer in layers:
            _, grads = layer.parameters()
            for name in grads:
                grads[name][...] = 0.0


class AdamOptimizer:
    """Dictionary-based Adam / AdamW / AMSGrad optimiser.

    Keeps first/second-moment state per named parameter (from `params_shapes`) and updates a dict of
    parameters from a dict of gradients.
    """
    def __init__(self, params_shapes, lr=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0, amsgrad=False):
        """
        params_shapes: dict of {param_name: shape} for every trainable param
                        e.g. {'W1': (20,64), 'b1': (64,), 'W2': (64,3), 'b2': (3,)}
        weight_decay:  decoupled weight decay coefficient (AdamW-style, 0 = off)
        amsgrad:       if True, use the AMSGrad variant (max of v_hat history)
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.t = 0

        self.m = {k: np.zeros(shape) for k, shape in params_shapes.items()}
        self.v = {k: np.zeros(shape) for k, shape in params_shapes.items()}
        if self.amsgrad:
            self.v_max = {k: np.zeros(shape) for k, shape in params_shapes.items()}


    def step(self, params, grads, lr=None, clip_norm=None):
        """Update `params` in place from `grads` (optionally with a custom lr and global gradient-norm clipping `clip_norm`)."""
        self.t += 1
        lr = self.lr if lr is None else lr
        bc1 = 1 - self.beta1 ** self.t
        bc2 = 1 - self.beta2 ** self.t

        for key in grads:
            g = grads[key]
            g_shape = np.asarray(g).shape

            # reinitialize any moment buffer whose shape has
            # drifted from the current gradient, 
            for buf_name, buf_dict in (('m', self.m), ('v', self.v)):
                if key not in buf_dict or np.asarray(buf_dict[key]).shape != g_shape:
                    if key in buf_dict:
                        print(f'[⚠️] Adam: "{buf_name}" buffer for "{key}" stale '
                            f'shape {np.asarray(buf_dict[key]).shape} != '
                            f'expected {g_shape} — reinitializing to zeros')
                    buf_dict[key] = np.zeros(g_shape)

            if self.amsgrad and (key not in self.v_max or
                                np.asarray(self.v_max[key]).shape != g_shape):
                self.v_max[key] = np.zeros(g_shape)

            if clip_norm is not None:
                norm = np.linalg.norm(g)
                if norm > clip_norm:
                    g = g * (clip_norm / norm)

            if self.weight_decay > 0:
                params[key] -= lr * self.weight_decay * params[key]

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[key] / bc1

            if self.amsgrad:
                self.v_max[key] = np.maximum(self.v_max[key], self.v[key])
                v_hat = self.v_max[key] / bc2
            else:
                v_hat = self.v[key] / bc2

            min_batch = min(params[key].shape[0], m_hat.shape[0])
            min_seq = min(params[key].shape[1], m_hat.shape[1])

            params[key][:min_batch, :min_seq] -= lr * m_hat[:min_batch, :min_seq] / (np.sqrt(v_hat[:min_batch, :min_seq]) + self.eps)

        return params
