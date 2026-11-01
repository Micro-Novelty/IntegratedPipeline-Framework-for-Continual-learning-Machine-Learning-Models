"""MLP classifier/regressor."""

import numpy as np
import traceback
from .common import _OPT_AVAILABLE, optimized_ame_encoder, optimized_anisotropy
from .metrics import expected_calibration_error
from .losses import Loss
from .dense_layers import Dense, QATDense, SoftmaxOutput


# enhanced MLP with focused forward and backward for better handling of data with varying geometric complexity,
# allowing it to complement the transformer module in the ensemble method.
# providing robust performance across a wider range of data complexities by dynamically adjusting its learning focus based on the data's geometric properties.
# focused forward and backward allows the MLP to adaptively concentrate on abundant layers during training, enhancing its ability to learn from data with varying geometric complexity for flexible applications.
# and providing a complementary learning dynamic when combined with the transformer in the ensemble.
# source of geometric weight research: https://github.com/Micro-Novelty/Specialized-MLP-for-noise-robustness


class MLP:
    """Multi-layer perceptron built from Dense / QATDense layers.

    Supports classification (softmax + cross-entropy) and regression (MSE), Adam updates, k-fold
    splitting, calibration metrics, and AME/anisotropy-based confidence and error-correction hooks.
    """
    def __init__(self):
        self.layers = []
        self.lr = 0.1

        self.error_counts = None
        self.pred_counts = None
        self.error_decay = None
        self.use_qat = True

        self.temp_AME_sample = 0
        self.temp_anisotropy_sample = 0

        self.softmax = SoftmaxOutput()


    def add(self, layer):
        """Append a layer, checking (or inferring) its input size immediately."""
        if self.layers:
            prev = self.layers[-1]
            if layer.input_size is None:
                layer.build(prev.output_size)
            elif layer.input_size != prev.output_size:
                raise ValueError(
                    f'Cannot add Dense({layer.input_size} -> {layer.output_size}) after '
                    f'a layer that outputs {prev.output_size} features. '
                    f'Use input_size={prev.output_size}, or MLP.add_dense(units).')
        self.layers.append(layer)
        self.layers[0].fit_input = True
        return self
 

    def add_dense(self, units, activation=None, **kw):
        """MLP().add_dense(64,'relu').add_dense(32,'relu').add_dense(3) - no sizes to match."""
        prev_out = self.layers[-1].output_size if self.layers else None
        return self.add(Dense(prev_out, units, activation=activation, **kw))


    @classmethod
    def from_qat_sizes(cls, input_dim, hidden, output_dim, activation='relu',
                   output_activation=None, seed=0, init='auto', lr=0.001):
        """
        MLP.from_qat_sizes(20, [64, 32], 3) builds 20->64->32->3. Sizes are chained by
        construction. Keep output_activation=None for classification (softmax is
        applied by MLP.forward) (ONLY FOR CLASSIFICATION)
        """
        m = cls()
        prev = int(input_dim)

        for i, units in enumerate(hidden):
            m.add(QATDense(prev, int(units), seed=seed + i, activation=activation,
                        init=init, lr=lr))
            prev = int(units)
        m.add(QATDense(prev, int(output_dim), seed=seed + len(hidden),
                    activation=output_activation, init=init, lr=lr))
        
        return m

    @classmethod
    def from_sizes(cls, input_dim, hidden, output_dim, activation='relu',
                   output_activation=None, seed=0, init='auto', lr=0.001):
        """
        MLP.from_sizes(20, [64, 32], 3) builds 20->64->32->3. Sizes are chained by
        construction. Keep output_activation=None for classification (softmax is
        applied by MLP.forward) and for regression.
        """
        m = cls()
        prev = int(input_dim)
        for i, units in enumerate(hidden):
            m.add(Dense(prev, int(units), seed=seed + i, activation=activation,
                        init=init, lr=lr))
            prev = int(units)
        m.add(Dense(prev, int(output_dim), seed=seed + len(hidden),
                    activation=output_activation, init=init, lr=lr))
        
        return m
 
    def build(self, input_dim):
        """Resolve any deferred input sizes and validate the chain."""
        if not self.layers:
            raise ValueError('MLP has no layers')
        if self.layers[0].input_size is None:
            self.layers[0].build(input_dim)
        for a, b in zip(self.layers, self.layers[1:]):
            if b.input_size is None:
                b.build(a.output_size)
        self.validate()
        return self
 
    def add_dense(self, layer):
        """Append a Dense (or QATDense) layer to the network."""
        self.layers.append(layer)

    def k_fold_split(self, X, y, k=5, seed=42, min_fold_size=2):
        """Split (X, y) into k shuffled folds (k is reduced for small datasets so no fold is smaller than `min_fold_size`)."""
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(X)

        # adapt k downward for small datasets, never produce
        # folds smaller than min_fold_size
        effective_k = min(k, max(2, n // min_fold_size))
        if effective_k < k:
            print(f'[⚠️] k_fold_split: reduced k from {k} to {effective_k} '
                f'for n={n} samples, to avoid folds smaller than '
                f'{min_fold_size} samples')

        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        folds = np.array_split(idx, effective_k)

        for i in range(effective_k):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(effective_k) if j != i])
            yield X[train_idx], y[train_idx], X[val_idx], y[val_idx]


    def confusion_matrix(self, y_true, y_pred, num_classes):
        """Return the (num_classes x num_classes) confusion matrix for true vs predicted labels."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        cm = np.zeros((num_classes, num_classes), dtype=int)

        # Convert to label arrays: use argmax only if input is one-hot/prob (2D+),
        # otherwise need to treat as already being class labels.
        true_labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
        pred_labels = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred

        for t, p in zip(true_labels, pred_labels):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t, p] += 1
            else:
                print(f'[⚠️] confusion_matrix: label out of range '
                    f'(true={t}, pred={p}, num_classes={num_classes}) — skipped')
        return cm


    def performance_calculation(self, x, AME=None, anisotropy=None):
        """Estimate prediction quality for input x from its AME / anisotropy scores and the model's running error statistics."""
        eps = 1e-5
        standard_low_error_mean = eps
        standard_pred_quality = eps

        if AME is None and anisotropy is None:
            AME = self.AME_Encoder(x)     
            anisotropy = self.anisotropy_measurement(x)

        gate_uncertain = (AME * anisotropy) + eps / 4
        gate_certainty = (1.0 - AME) + eps / 4
        if self.error_counts is not None and self.pred_counts is not None:
            standard_low_error_mean = (1.0 - np.mean(self.error_counts) + eps / 4)
            standard_pred_quality = np.mean(self.pred_counts) + eps / 4

        performance_score = (gate_uncertain + gate_certainty + 
                               standard_low_error_mean + standard_pred_quality)

        if np.isnan(performance_score) or np.isinf(performance_score):
            performance_score = 0.15

        return performance_score


    def validate(self):
        """Check that consecutive layers have matching sizes and that the final layer is compatible; raises ValueError otherwise."""
        for i, (a, b) in enumerate(zip(self.layers, self.layers[1:])):
            if b.input_size != a.output_size:
                raise ValueError(f'Layer {i + 1} expects {b.input_size} inputs but '
                                 f'layer {i} outputs {a.output_size}')
        last = self.layers[-1]
        if last.activation_name:
            print(f'[⚠️] MLP: output layer uses activation "{last.activation_name}". '
                  f'Logits should be linear (softmax/MSE is applied afterwards).')
 

    def parameter_count(self):
        """Total number of trainable parameters across all layers."""
        return sum(l.parameter_count() for l in self.layers)

    def set_qcfg(self, qcfg):
        """Apply a QuantConfig to every QATDense layer."""
        if qcfg is not None:
            self.use_qat = True
            for layers in self.layers:
                layers.set_qcfg(qcfg)
        else:
            print('[>] QuantConfig cannot be None!')


    # -------------------------------------------------------------- forward
    def forward(self, x, y=None, AME=None, anisotropy=None, condition=None, mode='classification'):
        """Forward pass through all layers (classification probabilities or regression outputs); in training mode it caches data for back-propagation."""
        if not self.layers:
            raise ValueError('[>] MLP has no layers')
        x = self._sanitize_string_chars(x)
        x = np.asarray(x)
        if x.ndim > 2:
            x = self._safe_to_2d_float(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y is not None:
            y = np.asarray(y)
            if y.ndim > 2:
                y = self._safe_to_2d_float(y)
 
        if any(l.input_size is None for l in self.layers):
            self.build(x.shape[1])
 
        performance_score = 1.0
        if mode != 'regression':
            performance_score = self.performance_calculation(x, AME=AME, anisotropy=anisotropy)
 
        out = x
        for layer in self.layers:                  
            out = layer.forward(out, performance_score)
 
        output = self.softmax.forward(out) if mode != 'regression' else out
 
        if mode != 'regression' and condition != 'training' and y is not None:
            if y.shape == output.shape:
                acc = np.mean(np.argmax(output, axis=1) == np.argmax(y, axis=1))
                loss = Loss.categorical_crossentropy(y, output)
                print(f"[=] MLP Forward validation score: | loss: {loss:.4f} | Acc: {acc:.2%}")
        return output



    def _sanitize_string_chars(self, x):
        if isinstance(x, (str, np.str_)):
            clean_str = str(x).replace('[', '').replace(']', '').replace('...', '').strip()
            x = np.fromstring(clean_str, sep=' ')

        if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.character):
            # catches arrays filled with string text
            clean_str = ' '.join(x.astype(str).flatten()).replace('[', '').replace(']', '')
            skip_values = {"...", "NaN", "null"}
            x = np.fromiter((v for v in clean_str.split() if v not in skip_values), dtype=float)

        return x

    

    def loss_mse_regression(self, epoch, preds, targets, AMR=None):
        """
        MSE loss for regression targets 

        preds   : (batch_size, output_dim) — MLP's raw forward output
        targets : (batch_size,) or (batch_size, output_dim) — continuous
                regression targets, NOT class indices or one-hot vectors
        AMR     : optional geometric complexity weight (from AME_Encoder),
                consistent with the same weighting convention used in
                your LSTM's loss_mse — pass None to skip it entirely

        Returns (loss: float, da: array) where da matches what
        backward(da, lr, clip_value) expects as its gradient input.
        """
        preds   = np.asarray(preds, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)

        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)

        if preds.shape[0] != targets.shape[0]:
            if epoch < 2:
                print(f'[⚠️] loss_mse_regression: batch size mismatch — '
                    f'preds={preds.shape[0]}, targets={targets.shape[0]} — '
                    f'aligning to shorter length')
            min_batch = min(preds.shape[0], targets.shape[0])
            preds   = preds[:min_batch]
            targets = targets[:min_batch]

        if preds.shape[1] != targets.shape[1]:
            if epoch < 2:
                print(f'[⚠️] loss_mse_regression: output dim mismatch — '
                    f'preds={preds.shape[1]}, targets={targets.shape[1]} — '
                    f'this usually means the wrong target array was passed '
                    f'(e.g. one-hot labels instead of regression values). '
                    f'Aligning to shorter width.')
            
            min_dim = min(preds.shape[1], targets.shape[1])
            preds   = preds[:, :min_dim]
            targets = targets[:, :min_dim]

        batch_size = preds.shape[0]
        diff = preds - targets

        weight = (1.0 - AMR)
        loss   = weight * np.mean(diff ** 2)

        da = weight * 2.0 * diff / batch_size

        return float(loss), da


    def r_squared(self, preds, targets):
        """Coefficient of determination (R^2) between regression predictions and targets."""
        preds   = np.asarray(preds, dtype=np.float64).reshape(-1)
        targets = np.asarray(targets, dtype=np.float64).reshape(-1)

        if len(preds) != len(targets):
            min_len = min(len(preds), len(targets))
            preds, targets = preds[:min_len], targets[:min_len]

        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - targets.mean()) ** 2)

        if ss_tot < 1e-8:
            # targets have essentially zero variance — R² is undefined,
            # not zero. 
            print('[⚠️] r_squared: target variance near zero — R² undefined')
            return float('nan')

        return float(1.0 - ss_res / ss_tot)   # 1.0 = perfect fit, 0.0 = no better than mean, negative = worse than mean



    def continuous_predictive_correction(self, manager, prob, predicted_index, AME=None, anisotropy=None):
        """Adjust a predicted probability using the manager's running per-class error/prediction counters and AME/anisotropy, to correct systematically over- or under-confident classes."""
        eps = 1e-5
        label_map = manager.label_map
        n_classes = len(label_map)
        manager._repair_class_counters(n_classes)   # guarantees valid 1-D counters
    
        error_counts = manager.error_counts
        pred_counts = manager.pred_counts
        decay = manager.decay
    
        self.error_counts = error_counts                  
        self.pred_counts = pred_counts
        self.error_decay = decay
    
        if prob is None:
            print('[!] Probabilities is None! returning the probabilities...')
            return prob
    
        try:
            idx = np.atleast_1d(np.asarray(predicted_index)).astype(int).ravel()
            idx = idx[(idx >= 0) & (idx < n_classes)]
            np.add.at(self.pred_counts, idx, 1.0)
    
            performance_score = self.performance_calculation(prob, AME=AME, anisotropy=anisotropy)
    
            for c in range(n_classes):
                if len(self.pred_counts) < c:             # (unchanged - see note 1)
                    if isinstance(self.pred_counts[c], (int, float)) and self.pred_counts[c] > 0:
                        error_rate = self.error_counts[c] / (self.pred_counts[c] + 1e-8)
                        small_reputation = 1.0 / (1.0 + error_rate)
                        weighted_reputation = small_reputation * performance_score
                        reputation_growth = 1.0 / (1.0 + np.exp(-performance_score))
                        compounding_factor = weighted_reputation * reputation_growth
    
                        if isinstance(error_rate, (list, np.ndarray)):
                            if len(error_rate) != len(prob):
                                weighted_reputation = 1.0 / (1.0 + np.mean(error_rate))
                        if c < len(prob):
                            prob[c] *= small_reputation
                        if c < len(self.pred_counts):
                            self.pred_counts[c] *= compounding_factor
    
            prob_sum = prob.sum()                          
            if prob_sum > 1e-8:
                prob /= prob_sum
    
        except Exception as e:
            print(f'[!] Cant check and calibrate probs based on penalty due to: {e}')
            traceback.print_exc()
    
        manager.pred_counts = self.pred_counts
        manager.error_counts = self.error_counts
        return prob
        
    def backward(self, grad, lr=None, momentum=0.9):
        """Backprop through every layer. Returns (grad_wrt_input, [per-layer grads])."""
        grad = self.softmax.backward(grad)
        all_grads = [None] * len(self.layers)
        for i in reversed(range(len(self.layers))):
            grad, all_grads[i] = self.layers[i].backward(grad, lr, momentum=momentum)
        return grad, all_grads
 

    def apply_gradients(self, lr=None, clip_norm=5.0):
        """Apply the accumulated gradients with each layer's Adam optimiser (optional norm clipping) and clear them."""
        for layer in self.layers:
            if layer.grads is None:
                continue
            layer.params = layer.opt.step(layer.params, layer.grads, lr=lr, clip_norm=clip_norm)
            layer.grads = None
 
    def regressive_training(self, epoch, X, y, lr=1e-3, momentum=0.9, clip_value=1.0, AME=None):
        """One regression training step: forward, AME-weighted MSE loss and backward. Returns (loss, predictions, gradients)."""
        AMR = (1.0 - AME) if AME is not None else 0.0
        preds = self.forward(X, mode='regression', condition='training')
        loss, da = self.loss_mse_regression(epoch, preds, y, AMR=AMR)
 
        if da.shape != preds.shape:        # loss helper may have trimmed to align; pad back
            full = np.zeros_like(preds)
            full[:da.shape[0], :da.shape[1]] = da
            da = full
 
        grad = da
        all_grads = [None] * len(self.layers)
        for i in reversed(range(len(self.layers))):   # <- grad is chained, not `da` every time
            grad, all_grads[i] = self.layers[i].backward(grad, lr, momentum=momentum)
        return loss, preds, all_grads 
            
    def predict(self, X, y, epochs=1000, verbose=True):
        """Simple full-batch training loop of `epochs` iterations with cross-entropy, printing progress when verbose."""
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = Loss.categorical_crossentropy(y, y_pred)    		
            if verbose and epoch % 100 == 0:
                acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
                self.acc2.append(acc)
                print(f"[=] Epoch {epoch} | loss:{loss:.4f} | Acc: {acc:.2f}")

	   
    def prediction(self, X):
        """Return the network output (probabilities) for X."""
        y_pred = self.forward(X)   
        return y_pred      
     
         
    def AME_Encoder(self, x):
        """Compute the AME score of x (Cython kernel when available, numpy fallback)."""
        X = np.asarray(x)
        if len(X) == 0:
            print('[!] X size is 0, AME Will be replaced by minimum confidence threshold')
            return 0.0

        if _OPT_AVAILABLE and np.asarray(X).ndim == 2:
            return optimized_ame_encoder(np.asarray(X, dtype=np.float64))     

        if x.shape[1] == 1:
            x = x.T
            x= x.flatten()

        try:
            gradient = np.gradient(x, axis=-1)
        except:
            x = np.reshape(x, (x.shape[0], -1))
            gradient = np.gradient(x, axis=-1, edge_order=1)

        grad_energy = np.mean(np.linalg.norm(gradient, axis=-1))       
        X_mag = np.mean(np.linalg.norm(X, axis=-1))

        AME = np.log1p(X_mag) * np.log1p(grad_energy) 
        return AME


    def anisotropy_measurement(self, x):
        """Measure the anisotropy of x (Cython kernel when available, numpy fallback)."""
        eps = 1e-5

        if _OPT_AVAILABLE:
            x = np.asarray(x)            
            x = x.reshape(x.shape[0], -1)
            return optimized_anisotropy(np.asarray(x, dtype=np.float64))

        try:
            gradient = np.gradient(x)
        except:
            subnet = x[:min(10, x.shape[0]), :min(10, x.shape[1])]
            gradient = np.gradient(subnet.flatten())

        val = [np.linalg.norm(v) for v in gradient]
        anisotropy = np.std(val) / np.mean(val) + eps

        return anisotropy

    def adapt_predict_shape(self, y_pred, y_true):
        """Reshape / trim predictions and targets so they have compatible 2-D shapes for loss computation."""
        try:
            y_pred_arr = np.asarray(y_pred)
            y_arr      = np.asarray(y_true)

            # normalize to 2D
            if y_pred_arr.ndim == 1:
                y_pred_arr = y_pred_arr[np.newaxis, :]
            if y_arr.ndim == 1:
                y_arr = y_arr[np.newaxis, :]

            # align batch and class dims — same approach as Loss
            min_batch = min(y_pred_arr.shape[0], y_arr.shape[0])
            min_class = min(y_pred_arr.shape[1], y_arr.shape[1])

            y_pred_aligned = y_pred_arr[:min_batch, :min_class]
            y_aligned      = y_arr[:min_batch, :min_class]

            if y_pred_aligned.size == 0 or y_aligned.size == 0:
                print(f'[!] Cannot compute accuracy — empty after alignment')
                acc = 0.0
            else:
                preds = np.argmax(y_pred_aligned, axis=1)
                true  = np.argmax(y_aligned, axis=1)
                acc   = float(np.mean(preds == true))

            return y_pred_aligned, y_aligned

        except Exception as e:
            print(f'[!] Cant adapt shape of Y sample arrays due to: {e}')
            return y_pred, y_true

    def _pad_ragged_to_array(self, x) -> np.ndarray:
        """
        Convert a ragged list-of-lists to a 2D array by padding shorter
        rows with zeros to match the longest row.
        Preserves per-row geometric structure for gradient computation.
        """
        try:
            rows = [np.asarray(item, dtype=np.float64).ravel()
                    for item in x]
        except (TypeError, ValueError):
            return np.asarray(x, dtype=object)

        if not rows:
            return None

        max_len = max(len(r) for r in rows)
        if max_len == 0:
            return None

        padded = np.zeros((len(rows), max_len), dtype=np.float64)
        for i, row in enumerate(rows):
            padded[i, :len(row)] = row

        return padded


    def _coerce_to_2d_float(self, x) -> np.ndarray:
        """
        NumPy 1.24+ raises ValueError on ragged asarray() without dtype=object.
        This method handles that cross-platform consistently.
        """
        if x is None:
            return None

        if isinstance(x, np.ndarray):
            return self._normalize_to_2d(x.astype(np.float64, copy=False))

        # list/tuple — check for raggedness 
        if isinstance(x, (list, tuple)) and len(x) > 0:
            first = x[0]

            # check if items are sequences of potentially different lengths here
            if isinstance(first, (list, tuple, np.ndarray)):
                lengths = set()
                for item in x:
                    if isinstance(item, np.ndarray):
                        lengths.add(item.size)
                    elif isinstance(item, (list, tuple)):
                        lengths.add(len(item))
                    else:
                        lengths.add(1)

                if len(lengths) > 1:
                    # RAGGED — pad to uniform length.
                    max_len = max(lengths)
                    print(f'[=] AME_Encoder: ragged input detected '
                        f'(lengths: {lengths}) — padding to {max_len}')
                    padded = np.zeros((len(x), max_len), dtype=np.float64)
                    for i, item in enumerate(x):
                        arr = np.asarray(item, dtype=np.float64).ravel()
                        padded[i, :len(arr)] = arr
                    return padded

        # homogeneous — safe to call asarray directly
        try:
            X = np.asarray(x, dtype=np.float64)
            return self._normalize_to_2d(X)
        except ValueError:
            # last resort — try with dtype=object then convert
            try:
                X = np.asarray(x, dtype=object)
                flat = np.array([
                    float(v) for v in X.ravel()
                    if v is not None
                ], dtype=np.float64)
                return flat.reshape(1, -1)
            except Exception:
                print('[!] Cant Process and Convert X samples!')
                return np.asarray(x, dtype=object)


    def _safe_to_2d_float(self, x) -> np.ndarray:
        """
        Attempt direct numpy conversion first (fast path for normal inputs).
        Only activates ragged/string handling when direct conversion fails.
        This avoids over-inspection that returns None for valid inputs.
        """
        try:
            if x is None:
                raise Warning('[!] X samples is None!')
                return np.asarray(x, dtype=object)

            # FAST PATH — already a numeric numpy array, most common case
            if isinstance(x, np.ndarray):
                if np.issubdtype(x.dtype, np.floating) or \
                np.issubdtype(x.dtype, np.integer):
                    return self._normalize_to_2d(x.astype(np.float64, copy=False))
                # non-numeric numpy array — string/object dtype
                return self._safe_to_2d(x)

            # ATTEMPT direct conversion — works for all homogeneous inputs
            # including: flat lists, lists of same-length arrays, 2D lists, etc.
            try:
                X = np.asarray(x, dtype=np.float64)
                return self._normalize_to_2d(X)

            except ValueError:
                # NumPy 1.24+ ragged array — activate padding path
                print(f'[=] _to_2d_float: ragged input detected — padding to uniform shape')
                return self._coerce_to_2d_float(x)

            except TypeError:
                # string content or incompatible types — activate safe path
                print(f'[=] _to_2d_float: type conversion failed — trying safe path')
                return self._safe_to_2d(x)

        except Exception as e:
            print(f'[!] _to_2d_float: unexpected error: {e} — using Robust method')
            try:
                result = self._convert_to_2d_float(x)
            except Exception as e:
                print('[!] cant Convert and calculate samples! - Using dtype object to compensate for the failure')
                return np.asarray(x, dtype=object)


    def _normalize_to_2d(self, X: np.ndarray) -> np.ndarray:
        """Reshape to 2D"""
        X = np.asarray(X)
        if X.ndim == 0:
            return X.reshape(1, 1)
        if X.ndim == 1:
            return X.reshape(1, -1)
        if X.ndim == 2:
            return X
        # >2D: squeeze singleton axes except the batch axis, then flatten the rest
        axes = tuple(i for i in range(1, X.ndim) if X.shape[i] == 1)
        if axes:
            X = X.squeeze(axis=axes)
        return X if X.ndim == 2 else X.reshape(X.shape[0], -1)

    def _convert_to_2d_float(self, x) -> np.ndarray:
        """
        Adaptive gate — detects input characteristics and routes to
        the appropriate normalization path:
        
        _coerce_to_2d_float  → when input is ragged (variable-length sequences)
                            or when NumPy 1.24+ would raise ValueError
        _safe_to_2d          → when input has string/object dtype, nested
                            structures, or needs deep type conversion
        direct numpy         → when input is already a clean numeric array
        """
        if x is None:
            raise Warning('[!] X samples is None!')
            return None

        # FAST PATH — already a clean numpy float array, skip both functions
        if isinstance(x, np.ndarray):
            if np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.integer):
                return self._normalize_to_2d(x.astype(np.float64, copy=False))
            # non-numeric numpy array → needs _safe_to_2d for string/object handling
            return self._safe_to_2d(x)

        # string input - always _safe_to_2d (handles parsing logic)
        if isinstance(x, (str, np.str_)):
            return self._safe_to_2d(x)

        # list/tuple — detect ragged vs homogeneous vs string content
        if isinstance(x, (list, tuple)) and len(x) > 0:
            first = x[0]

            # strings inside list → _safe_to_2d
            if isinstance(first, (str, np.str_)):
                return self._safe_to_2d(x)

            # nested sequences — check for raggedness
            if isinstance(first, (list, tuple, np.ndarray)):
                lengths = set()
                has_strings = False

                for item in x:
                    if isinstance(item, (str, np.str_)):
                        has_strings = True
                        break
                    elif isinstance(item, np.ndarray):
                        lengths.add(item.size)
                        if np.issubdtype(item.dtype, np.character):
                            has_strings = True
                            break
                    elif isinstance(item, (list, tuple)):
                        lengths.add(len(item))

                # string content anywhere → _safe_to_2d
                if has_strings:
                    return self._safe_to_2d(x)

                # ragged numeric - _coerce_to_2d_float (pads to uniform length)
                if len(lengths) > 1:
                    return self._coerce_to_2d_float(x)

                # homogeneous numeric - direct numpy, fastest path
                try:
                    X = np.asarray(x, dtype=np.float64)
                    return self._normalize_to_2d(X)
                except ValueError:
                    # for edge case — fall back to coerce
                    return self._coerce_to_2d_float(x)

            # flat list of scalars 
            try:
                X = np.asarray(x, dtype=np.float64)
                return self._normalize_to_2d(X)
            except (ValueError, TypeError):
                return self._safe_to_2d(x)

        # scalar input
        try:
            return np.array([[float(x)]])
        except (TypeError, ValueError):
            print('[!] Cant normalize and Convert X Samples! - Using dtype object to compensate failure!')
            return np.asarray(x, dtype=object)

    
    def train(self, X, y, X_val=None, y_val=None, epochs=1000, lr=0.01, momentum=0.9, verbose=True, mode='classification'):
        """Main training loop (classification or regression) with optional validation data, momentum and per-epoch metrics. Returns (loss, accuracy), or (None, None) if training could not run."""
        y_val_pred = None
        y_val_true = None
        loss, acc = 0.0, 0.0
        all_probs = []
        all_val = []

        X = self._sanitize_string_chars(X)
        y = self._sanitize_string_chars(y)
        self.lr = lr
 
        if len(X.shape) > 2 or len(X.shape) == 1:
            X = self._safe_to_2d_float(X)
        if len(y.shape) > 2 or len(y.shape) == 1:
            y = self._safe_to_2d_float(y)
            if y.shape[0] == 1:
                y = y.copy().T

        AME = self.AME_Encoder(X)
        AMR = 1.0 / (1.0 + np.exp(-float(AME)))

        anisotropy = self.anisotropy_measurement(X)
        self.temp_AMR_sample = AMR
        self.temp_anisotropy_sample = anisotropy
 
        if mode == 'classification':
            training_not_allowed = (np.isnan(anisotropy) or np.isinf(anisotropy) or
                                    np.isnan(AME) or np.isinf(AME) or AME < 0.1)
            if training_not_allowed:
                print(f'[!] MLP Training not allowed due to unsuitable data characteristics. '
                      f'Anisotropy: {anisotropy:.4f}, AME: {AME:.4f}')
                return None, None
 
            print(f'[TRAIN] MLP Training started with samples that has a total length of {len(X)}...')
            for epoch in range(epochs):
                if X_val is not None and y_val is not None:
                    
                    y_val_pred = self.forward(X_val, y=y_val, AME=AME, anisotropy=anisotropy,
                                              condition='training')
                    y_val_pred, y_val_true = self.adapt_predict_shape(y_val_pred, y_val)

                    all_probs.append(y_val_pred)
                    all_val.append(y_val_true)
 
                y_pred = self.forward(X, y=y, AME=AME, anisotropy=anisotropy, condition='training')
                if y_pred.shape[0] != y.shape[0]:
                    y_pred, y_true = self.adapt_predict_shape(y_pred, y)
                else:
                    y_pred, y_true = y_pred.copy(), y.copy()
 
                loss = Loss.categorical_crossentropy(y_true, y_pred)
                grad = Loss.softmax_crossentropy_derivative(y_true, y_pred)
                if grad.shape != y_pred.shape:              
                    full = np.zeros_like(y_pred)
                    full[:grad.shape[0], :grad.shape[1]] = grad
                    grad = full
                self.backward(grad, self.lr, momentum=momentum)
                self.apply_gradients()                       # <- all layers, own optimizer each
 
                acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
                if verbose and epoch % 100 == 0:
                    if y_val_pred is not None and y_val_true is not None:
                        val_acc = np.mean(np.argmax(y_val_pred, axis=1) == np.argmax(y_val_true, axis=1))
                        print(f"[=] Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc:.2f} | Validation Acc: {val_acc:.2f}")
                    else:
                        print(f"[=] Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc:.2f}")
                
 
        elif mode == 'regression':
            print(f'[TRAIN] MLP Regression based Training started with samples that has a total length of {len(X)}...')
            for epoch in range(epochs):
                loss, preds, _ = self.regressive_training(epoch, X, y, lr=lr, momentum=momentum,
                                                          clip_value=1.0, AME=AME)
                self.apply_gradients()
                acc = self.r_squared(preds, y)
                if verbose and epoch % 100 == 0:
                    print(f'[=] Epoch {epoch} | Loss: {loss:.4f} | R²: {acc:.4f}')
        else:
            print('[>] Unknown Training mode, cant continue Training.')
 
        if self.layers and self.layers[0].W is not None:
            if len(all_probs) > 0 and len(all_val) > 0:
                all_probs = np.vstack(all_probs)
                all_val = np.concatenate(all_val)

                ece = expected_calibration_error(all_probs, all_val)
                print(f'[TRAIN] Expected calibration error: {ece}')

            print(f'[TRAIN] Done Training MLP with: {self.parameter_count()} parameters!')
        return loss, acc
