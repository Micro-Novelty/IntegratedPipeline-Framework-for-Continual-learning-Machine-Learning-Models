"""SNNUtility: seeding / RNG helpers for the SNN code."""

import numpy as np


class SNNUtility:
    """Small helper toolbox for the SNN code: seeding, one-hot encoding and accuracy."""
    def __init__(self):
        pass

    def seed_everything(self, seed):
        """Return a seeded numpy Generator; also seeds the legacy global state
        for any code that still calls np.random.* directly."""
        np.random.seed(seed)
        return np.random.default_rng(seed)


    def one_hot(self, labels, n_classes):
        """Convert integer class labels to a one-hot matrix of shape (n, n_classes)."""
        labels = np.asarray(labels)
        out = np.zeros((labels.shape[0], n_classes))
        out[np.arange(labels.shape[0]), labels] = 1.0
        return out

    def accuracy(self, preds, labels):
        """Fraction of predictions equal to the labels."""
        return float(np.mean(np.asarray(preds) == np.asarray(labels)))

    def clip_grad_norm(self, layers, max_norm, eps=1e-6):
        """Global-norm gradient clipping across every parameter of every layer,
        analogous to torch.nn.utils.clip_grad_norm_. Scales all gradients by
        the same factor so relative gradient direction is preserved."""
        total_sq = 0.0
        grad_refs = []
        for layer in layers:
            _, grads = layer.parameters()
            for g in grads.values():
                if g is None:
                    continue
                grad_refs.append(g)
                total_sq += float(np.sum(g * g))
        total_norm = np.sqrt(total_sq)
        if total_norm > max_norm:
            scale = max_norm / (total_norm + eps)
            for g in grad_refs:
                g *= scale
        return total_norm
