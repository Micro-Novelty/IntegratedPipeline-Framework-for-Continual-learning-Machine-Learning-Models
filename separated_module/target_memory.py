"""TargetMemory and TargetMemoryMixin: memory-blended predictions for the transformer."""

import hashlib
import io
import json
import numpy as np


class TargetMemory:
    """Nearest-neighbour memory of training targets.

    Stores standardised features with their targets and predicts a distance-weighted average
    (regression) or class distribution (classification) from the k closest entries.
    """
    def __init__(self, task="regression", n_classes=None, k=8, temperature=0.5):
        if task not in ("regression", "classification"):
            raise ValueError("task must be 'regression' or 'classification'")
        if task == "classification" and not n_classes:
            raise ValueError("n_classes is required for classification")
        self.task, self.n_classes, self.k, self.temperature = task, n_classes, k, temperature
        self.keys = self.targets = self.mu = self.sd = None
 
    def fit(self, feats, targets):
        """Store the (standardised) training features and targets; statistics come from the training data only."""
        feats = np.asarray(feats, dtype=np.float64)
        targets = np.asarray(targets)
        if len(feats) != len(targets):
            raise ValueError(f"{len(feats)} feature rows but {len(targets)} targets")
        self.mu, self.sd = feats.mean(0), feats.std(0) + 1e-8            # train statistics only
        self.keys = (feats - self.mu) / self.sd
        self.targets = targets
        return self
 
    def _dist2(self, q):
        q = (np.asarray(q, dtype=np.float64) - self.mu) / self.sd
        return (q ** 2).sum(1)[:, None] + (self.keys ** 2).sum(1)[None, :] - 2.0 * q @ self.keys.T
 
    def predict(self, feats, k=None, temperature=None, exclude_self=False):
        """Predict from the k nearest stored entries; `exclude_self` ignores each sample's own entry (leave-one-out)."""
        if self.keys is None:
            raise RuntimeError("call fit() first")
        k, T = k or self.k, temperature or self.temperature
        d2 = np.maximum(self._dist2(feats), 0.0)
        if exclude_self:
            np.fill_diagonal(d2, np.inf)
        k = min(k, d2.shape[1])
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        d = np.take_along_axis(d2, idx, 1)
        w = np.exp(-(d - d.min(1, keepdims=True)) / (T * (np.median(d, 1, keepdims=True) + 1e-8)))
        w /= w.sum(1, keepdims=True)
        if self.task == "regression":
            return (w * self.targets[idx]).sum(1)
        onehot = np.eye(self.n_classes)[self.targets[idx].astype(int)]
        return (w[:, :, None] * onehot).sum(1)


def blend(model_pred, mem_pred, lam):
    """Linear blend of model and memory predictions: (1 - lam) * model_pred + lam * mem_pred."""
    return (1.0 - lam) * model_pred + lam * mem_pred


class TargetMemoryMixin:
    """Mixin that gives a model an optional TargetMemory blended with its own predictions.

    Handles fitting the memory on the model's features, fingerprinting the weights it was built from
    (so stale memories can be detected) and (de)serialisation.
    """
    target_memory = None          # TargetMemory or None
    target_memory_cfg = None      # {'k':..,'temperature':..,'lam':..}
    target_memory_fp = None       # fingerprint of the weights the memory was built from
 
    # ---- helpers ------------------------------------------------------------------------------
    def _param_fingerprint(self):
        h = hashlib.sha1()
        for name in ("input_proj", "W_q", "W_k", "W_v", "W_o", "ffn1", "ffn2", "output"):
            a = getattr(self, name, None)
            if a is not None:
                h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        return h.hexdigest()
 
    def _pooled(self, X, input_mode="continuous", batch_size=256):
        """Model output + pooled representation. input_mode='continuous' -> forward_continuous (regression,
        raw (B,T,F) windows); 'embedded' -> forward(embedded=True) (your classification path)."""
        X = np.asarray(X, dtype=np.float64)
        outs, feats = [], []
        for s in range(0, len(X), batch_size):
            xb = X[s:s + batch_size]
            if input_mode == "embedded":
                p, _ = self.forward(input_ids=xb, embedded=True, training=False,
                                    attn_dropout=0.0, ffn_dropout=0.0)
            else:
                p, _ = self.forward_continuous(xb, training=False, attn_dropout=0.0, ffn_dropout=0.0)
            outs.append(np.asarray(p))
            feats.append(self.cache["x_pooled"].copy())
        return np.concatenate(outs), np.concatenate(feats)
 
    # ---- fit + tune ---------------------------------------------------------------------------
    def fit_target_memory(self, X_train, y_train, X_val, y_val, input_mode="continuous",
                          n_classes=None, min_gain=0.02, verbose=True):
        """Call ONCE after training. Memory is built from X_train only; k / temperature / lam are tuned on
        X_val. If the best blend is not at least `min_gain` (relative) better than the model alone on
        validation, lam is set to 0 (memory switched off)."""
        reg = self.task == "regression"
        if len(X_train) < 20 or len(X_val) < 5:
            if verbose: print("[=] target memory skipped: too few samples")
            return None
        _, F_tr = self._pooled(X_train, input_mode)
        p_va, F_va = self._pooled(X_val, input_mode)
        y_tr, y_va = np.asarray(y_train), np.asarray(y_val)
        if reg:
            y_tr, y_va, base_va = y_tr.reshape(-1).astype(float), y_va.reshape(-1).astype(float), p_va[:, 0]
            metric = lambda p: float(np.mean((p - y_va) ** 2))
        else:
            y_tr = y_tr.argmax(1) if y_tr.ndim == 2 else y_tr.astype(int)
            y_va = y_va.argmax(1) if y_va.ndim == 2 else y_va.astype(int)
            n_classes = n_classes or self.output.shape[1]
            base_va = p_va
            metric = lambda p: float(-np.mean(np.log(p[np.arange(len(y_va)), y_va] + 1e-9)))   # NLL
        mem = TargetMemory("regression" if reg else "classification", n_classes=n_classes).fit(F_tr, y_tr)
 
        base = metric(base_va)
        best = (base, 8, 0.5, 0.0)
        for k in (4, 8, 16, 32):
            for T in (0.25, 0.5, 1.0):
                mp = mem.predict(F_va, k=k, temperature=T)
                for lam in np.linspace(0.1, 1.0, 10):
                    m = metric(blend(base_va, mp, lam))
                    if m < best[0]:
                        best = (m, k, T, float(lam))
        if best[3] > 0 and best[0] > base * (1.0 - min_gain):
            best = (base, 8, 0.5, 0.0)                       # not clearly better -> off
        self.target_memory = mem
        self.target_memory_cfg = {"k": best[1], "temperature": best[2], "lam": best[3], "input_mode": input_mode}
        self.target_memory_fp = self._param_fingerprint()
        if verbose:
            state = "ON" if best[3] > 0 else "OFF (did not beat the model on validation)"
            print(f"[=] target memory {state} | val {'MSE' if reg else 'NLL'}: model={base:.4f} "
                  f"blend={best[0]:.4f} | k={best[1]} T={best[2]} lam={best[3]:.1f} | {len(F_tr)} entries")
        return self.target_memory_cfg
 
    # ---- inference ----------------------------------------------------------------------------
    def predict_with_memory(self, X, input_mode=None):
        """Regression: (B,) predictions. Classification: (B, C) probabilities (argmax them yourself)."""
        cfg = self.target_memory_cfg or {}
        mode = input_mode or cfg.get("input_mode", "continuous")
        p, F = self._pooled(X, mode)
        base = p[:, 0] if self.task == "regression" else p
        if self.target_memory is None or cfg.get("lam", 0.0) == 0.0:
            return base
        if self._param_fingerprint() != self.target_memory_fp:
            print("[!] target memory is stale (model weights changed since it was built) -> using model only. "
                  "Call fit_target_memory() again.")
            return base
        mp = self.target_memory.predict(F, k=cfg["k"], temperature=cfg["temperature"])
        return blend(base, mp, cfg["lam"])
 
    # ---- (de)serialisation --------------------------------------------------------------------
    def target_memory_to_bytes(self):
        """Serialise the memory (keys, targets, statistics, config, weight fingerprint) to compressed .npz bytes; None if no memory."""
        if self.target_memory is None:
            return None
        tm, cfg = self.target_memory, self.target_memory_cfg
        meta = {"task": tm.task, "n_classes": tm.n_classes, "cfg": cfg, "fp": self.target_memory_fp}
        buf = io.BytesIO()
        np.savez_compressed(buf, keys=tm.keys, targets=tm.targets, mu=tm.mu, sd=tm.sd,
                            meta=np.array(json.dumps(meta)))
        return buf.getvalue()
 
    def target_memory_from_bytes(self, blob):
        """Restore a memory from bytes produced by `target_memory_to_bytes`."""
        d = np.load(io.BytesIO(blob), allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        tm = TargetMemory(meta["task"], n_classes=meta["n_classes"])
        tm.keys, tm.targets, tm.mu, tm.sd = d["keys"], d["targets"], d["mu"], d["sd"]
        self.target_memory, self.target_memory_cfg, self.target_memory_fp = tm, meta["cfg"], meta["fp"]
        if self._param_fingerprint() != meta["fp"]:
            print("[!] stored target memory was built from different weights -> disabled; refit it.")
            self.target_memory_cfg = dict(meta["cfg"], lam=0.0)
            return False
        return True
