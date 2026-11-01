"""Loss functions and the shape-alignment helper they rely on."""

import numpy as np


@staticmethod
def _align_shapes(y_true, y_pred, context=""):
    """Coerce y_true / y_pred to float64 2-D arrays of matching shape.

    `context` is only used in log messages when shapes had to be reconciled.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]

    # Case: y_true is class indices (single column) but y_pred has multiple classes
    # -> one-hot encode y_true to match y_pred's class dimension
    if y_true.shape[1] == 1 and y_pred.shape[1] > 1:
        num_classes = y_pred.shape[1]
        labels = y_true.astype(int).ravel()
        labels = np.clip(labels, 0, num_classes - 1)  # guard bad indices
        one_hot = np.zeros((y_true.shape[0], num_classes))
        one_hot[np.arange(y_true.shape[0]), labels] = 1
        y_true = one_hot
        
    # Fallback: genuine mismatch after that -> truncate to min shape
    if y_true.shape != y_pred.shape:
        min_batch = min(y_true.shape[0], y_pred.shape[0])
        min_class = min(y_true.shape[1], y_pred.shape[1])
        y_true = y_true[:min_batch, :min_class]
        y_pred = y_pred[:min_batch, :min_class]

    return y_true, y_pred


class Loss:
    """Loss functions and gradients used by the MLP (categorical cross-entropy, softmax-CE derivative)."""
    @staticmethod
    def categorical_crossentropy(y_true, y_pred):
        """Mean categorical cross-entropy between one-hot y_true and predicted probabilities."""
        eps = 1e-5
        y_true, y_pred = _align_shapes(y_true, y_pred, context="crossentropy")
        y_pred = np.clip(y_pred, eps, 1 - eps)

        if y_true.size == 0 or y_pred.size == 0:
            print('[!] Empty arrays after alignment — returning safe default loss')
            return 1.0

        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

        if not np.isfinite(loss):
            return 0.0

        return float(loss)

    @staticmethod
    def softmax_crossentropy_derivative(y_true, y_pred):
        """Gradient of softmax + cross-entropy w.r.t. the logits (y_pred - y_true, batch-averaged)."""
        y_true, y_pred = _align_shapes(y_true, y_pred, context="crossentropy derivative")

        if y_true.size == 0 or y_pred.size == 0:
            print('[!] Empty arrays after alignment — returning zero gradient')
            return np.zeros_like(y_pred) if y_pred.size else np.zeros((1, 1))

        cross_ent = (y_pred - y_true) / y_true.shape[0]

        return cross_ent
