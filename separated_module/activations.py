"""Activation functions and their derivatives."""

import numpy as np
from .common import _OPT_AVAILABLE, optimized_softmax_2d


# ________ UTILITY functions for activations and losses, can be used across different models and architectures _________

def sigmoid(x):
    """Numerically safe logistic sigmoid (input clipped to [-500, 500])."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_deriv(s):          # s = sigmoid(x) already computed
    """Derivative of the sigmoid given its already-computed output s."""
    return s * (1.0 - s)


def tanh_deriv(t):             # t = tanh(x) already computed
    """Derivative of tanh given its already-computed output t."""
    return 1.0 - t ** 2


class Activation:
    """Collection of static activation functions (ReLU, sigmoid, softmax) and derivatives."""
    @staticmethod
    def relu(x):
        """ReLU: max(0, x)."""
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        """Derivative of ReLU (1 where x > 0, else 0)."""
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x):
        """Logistic sigmoid."""
        eps = 1e-5
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of the sigmoid evaluated at x."""
        eps = 1e-5
        s = Activation.sigmoid(x)
        return s * (1.0 - s)

    @staticmethod
    def softmax(x):
        """Row-wise softmax (uses the Cython kernel when available, numpy otherwise)."""
        if _OPT_AVAILABLE:
            output = optimized_softmax_2d(np.asarray(x, dtype=np.float64))    
            return output

        # numerical stability
        if x.ndim > 1:
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            normalized = exp_x / np.sum(exp_x, axis=1, keepdims=True)
            return normalized

        else:
            exp_x = np.exp(x - np.max(x, keepdims=True))
            return exp_x / np.sum(exp_x, keepdims=True)
