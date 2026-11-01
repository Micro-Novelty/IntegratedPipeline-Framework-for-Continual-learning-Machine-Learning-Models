"""SNN construction, mini-batching, STDP pre-training, training and prediction entry points."""

import numpy as np
import traceback
from .data_utils import _prepare_train_val_split
from .snn_encoders import PoissonEncoder
from .snn_layers import LIFDenseLayer, ReadoutLayer, STDPLearner, SpikeDropout, ThresholdHomeostasis
from .snn_utility import SNNUtility
from .optimizers import Adam
from .snn_network import SNNNetwork, mse_loss, softmax_cross_entropy
from .snn_diagnostics import DiagnosticsTracker, check_health, network_report, print_report
from .snn_calibration import mc_confidence, print_confidence_report


# ----------------------------------------
# SNN Network Build
#-----------------
def build_snn_network(n_in, n_hidden, n_classes, rng, decay=0.95, 
                      threshold=0.1, input_noise_std=0.0, reduction='mean', p=0.2):
    """Assemble the default SNN: LIF hidden layer -> SpikeDropout -> ReadoutLayer wrapped in an SNNNetwork."""
    layer1 = LIFDenseLayer(n_in, n_hidden, decay=decay, threshold=threshold, rng=rng)
    drop1 = SpikeDropout(p=p, rng=rng)
    readout = ReadoutLayer(n_hidden, n_classes, decay=decay, rng=rng)
    net = SNNNetwork([layer1, drop1, readout], readout_reduction=reduction,
                      input_noise_std=input_noise_std, rng=rng)
    return net, layer1


def iterate_minibatches(X, y, batch_size, rng, shuffle=True):
    """Yield (X_batch, y_batch) mini-batches, optionally shuffled with the given rng."""
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        sel = idx[start:start + batch_size]
        yield X[sel], y[sel]


def stdp_pretrain(layer, encoder, X_unlabeled, rng, n_steps=200, batch_size=32,
                   a_plus=0.01, a_minus=0.011, tau_pre=15.0, tau_post=15.0):
    """
    Unsupervised pretraining loop for a single LIFDenseLayer using pair-based
    STDP. Only spikes are used -- no labels, no loss, no optimizer step.
    """
    # STDP is defined for weights that stay in a bounded, non-negative range;
    # re-init the layer's weights accordingly before pretraining with it.
    layer.W[:] = rng.uniform(0.0, 0.4, size=layer.W.shape)
    stdp = STDPLearner(layer.in_features, layer.out_features,
                        a_plus=a_plus, a_minus=a_minus,
                        tau_pre=tau_pre, tau_post=tau_post,
                        w_min=0.0, w_max=1.0)
 
    n = X_unlabeled.shape[0]
    for step in range(n_steps):
        xb = X_unlabeled[rng.integers(0, n, size=batch_size)]
        pre_spikes = encoder.encode(xb)
        post_spikes = layer.forward(pre_spikes)   # plain LIF forward pass; no backward needed
        stdp.apply(layer.W, pre_spikes, post_spikes)
 
    return layer


def initialize_snn_network(X=None, y=None, label_map=None, n_hidden=64, n_in=None, n_classes=None, seed=0, random_state=42):
    """Create an untrained SNN, inferring input/class sizes from X / y / label_map when not given. Returns (net, hidden_layer, utility), or (None, None, None) on failure."""
    try:
        utility = SNNUtility()
        rng = utility.seed_everything(seed)
        if n_in is not None and n_classes is not None:
            net, hidden_layer = build_snn_network(n_in, n_hidden, n_classes, rng)
        else:
            X_train, X_test, y_train, y_test, X_val, y_val = _prepare_train_val_split(X=X, y=y, random_state=random_state)
            n_in = X_train.shape[1]
            n_classes = max(len(label_map), int(y_train.max() + 1))
            net, hidden_layer = build_snn_network(n_in, n_hidden, n_classes, rng)

        print(f'[>] Initialized SNN Network with hidden dim: {n_hidden}')
        return net, hidden_layer, utility
    except Exception as e:
        print(f'[>] Error initializing SNN Network and hidden layer: {e}')
        return None, None, None


def train_snn_network(X=None, y=None, net=None, hidden_layer=None, utility=None, task='classification', label_map=None,
         epochs=50, batch_size=32, n_steps=20, 
         n_hidden=64, lr=2e-3, seed=0, pretrain_steps=200, unsupervised_stdp=False,
         random_state=42, ):
    """Train a spiking network for classification or regression.

    Splits train/validation data, optionally STDP-pretrains the hidden layer, trains with Adam,
    applies threshold homeostasis and tracks diagnostics each epoch. Returns (encoder, net, report, warnings).
    """
    tracker = DiagnosticsTracker()
    cache = {}
    reg = (task == 'regression')
    if utility is None:
        utility = SNNUtility()

    rng = utility.seed_everything(seed)
    if reg:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim > 2:
            X = X.reshape(len(X), -1)                         # (N, W, F) -> (N, W*F)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if len(X) != len(y):
            raise ValueError(f'X has {len(X)} samples but y has {len(y)}')
        n = len(X)
        a, b = int(0.70 * n), int(0.85 * n)                   # chronological: never shuffle a time series
        X_train, X_val, X_test = X[:a], X[a:b], X[b:]
        y_train, y_val, y_test = y[:a], y[a:b], y[b:]
        lo, hi = X_train.min(0), X_train.max(0)               # [0,1] scaling for spike encoding (train stats)
        span = np.where(hi - lo < 1e-9, 1.0, hi - lo)
        scale_x = lambda A: np.clip((A - lo) / span, 0.0, 1.0)
        X_train, X_val, X_test = scale_x(X_train), scale_x(X_val), scale_x(X_test)
        y_mu, y_sd = y_train.mean(0), y_train.std(0) + 1e-9   # per-output standardisation
        y_train_net = (y_train - y_mu) / y_sd
        n_classes = y.shape[1]                                # readout width = number of regression outputs
        loss_fn = mse_loss
    else:

        X_train, X_test, y_train, y_test, X_val, y_val = _prepare_train_val_split(X=X, y=y, random_state=random_state)
        y_train_net = y_train
        n_classes = max(len(label_map), int(y_train.max() + 1))

    n_in = X_train.shape[1]
    if net is None or hidden_layer is None:
        net, hidden_layer = build_snn_network(n_in, n_hidden, n_classes, rng)

    if reg:
        net.regression_scalers = {'lo': lo, 'span': span, 'y_mu': y_mu, 'y_sd': y_sd}

    encoder = PoissonEncoder(n_steps=n_steps, max_rate=1.0, rng=rng)
    optimizer = Adam(lr=lr, weight_decay=1e-5)
    homeostasis = ThresholdHomeostasis(target_rate=0.15, adjust_rate=0.05)

    # ---- Phase 1: unsupervised STDP pretraining (labels NOT used) ----
    if unsupervised_stdp:
        if n_in == n_hidden:
            utility.stdp_pretrain(hidden_layer, encoder, X_train, rng, n_steps=pretrain_steps,
                        batch_size=batch_size)
            print(f"[=] After STDP pretraining: hidden layer mean firing rate = "
                f"{hidden_layer._cache['spike_hist'].mean():.3f}")
            per_neuron_rate = hidden_layer._cache["spike_hist"].mean(axis=(0, 1))  # shape (n_hidden,)
            print("[=] Dead neurons:", (per_neuron_rate < 1e-4).sum(), "/", len(per_neuron_rate))
            per_neuron_rate = hidden_layer._cache["spike_hist"].mean(axis=(0, 1))
            print("[=] Neuron firing rates - min:", per_neuron_rate.min(), "max:", per_neuron_rate.max(), "std:", per_neuron_rate.std())
        else:
            print('[>] Unsupervised STDP pretraining skipped due to mismatched Input_shapes[1] size with the amount of network hidden layers')

    # ---- Phase 2: supervised BPTT fine-tuning of the whole network ----
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_losses = []
        for xb, yb in iterate_minibatches(X_train, y_train, batch_size, rng):
            spikes = encoder.encode(xb)
            loss, logits = net.train_step(
                spikes, yb, optimizer,
                loss_fn=softmax_cross_entropy,
                max_grad_norm=5.0,
            )
            epoch_losses.append(loss)

        homeostasis.step(net.layers)

        # Evaluation
        net.eval()
        preds, _ = net.predict(encoder.encode(X_test), task=task)
        train_preds, _ = net.predict(encoder.encode(X_train), task=task)
        if reg:
            p_te = np.asarray(preds).reshape(len(X_test), -1) * y_sd + y_mu          # back to original units
            p_tr = np.asarray(train_preds).reshape(len(X_train), -1) * y_sd + y_mu
            test_rmse = float(np.sqrt(np.mean((p_te - y_test) ** 2)))
            test_mae = float(np.mean(np.abs(p_te - y_test)))
            train_rmse = float(np.sqrt(np.mean((p_tr - y_train) ** 2)))
            r2 = float(1.0 - np.sum((p_te - y_test) ** 2) / (np.sum((y_test - y_train.mean(0)) ** 2) + 1e-12))
            print(f"[=] Epoch {epoch:2d} || train_loss={np.mean(epoch_losses):.4f}  train_rmse={train_rmse:,.1f} || "
                  f"test_rmse={test_rmse:,.1f} test_mae={test_mae:,.1f} R2={r2:.3f} || "
                  f"hidden_threshold={hidden_layer.threshold:.3f}")
            test_score = r2
        else:
            test_acc = utility.accuracy(preds, y_test)
            train_acc = utility.accuracy(train_preds, y_train)
            print(f"[=] Epoch {epoch:2d} || train_loss={np.mean(epoch_losses):.4f}  "
                  f"train_acc={train_acc*100:5.1f} || test_acc={test_acc*100:5.1f}% || "
                  f"hidden_threshold={hidden_layer.threshold:.3f}")
            test_score = test_acc
 
        tracker.record(net, epoch, loss=loss, test_acc=test_score)

    report = network_report(net)
    print_report(report)
    warnings = check_health(report, target_rate=0.15)
    if warnings:
        print("\n".join(warnings))

    return encoder, net, report, warnings


def _snn_predict_regression(X, net, encoder, n_mc=20):
    """Regression inference: Monte-Carlo average over `n_mc` stochastic passes, then un-scale with the net's stored regression scalers."""
    sc = getattr(net, 'regression_scalers', None)
    if sc is None:
        raise ValueError('[>] net has no regression_scalers: train it with train_snn_network(task="regression") first')
    X = np.asarray(X, dtype=np.float64)
    X = X.reshape(len(X), -1)                                                    # (N, W*F)
    Xs = np.clip((X - sc['lo']) / sc['span'], 0.0, 1.0)

    # the Poisson encoder is stochastic, its repeated to get a prediction and a spread
    outs = np.stack([net.predict(encoder.encode(Xs), task='regression')[1] for _ in range(n_mc)])   # (n_mc, N, n_out)
    mean = outs.mean(0) * sc['y_sd'] + sc['y_mu']
    std = outs.std(0) * sc['y_sd']

    if mean.shape[1] == 1:
        mean, std = mean[:, 0], std[:, 0]
    return {'prediction': mean, 'std': std}


def snn_predict(X=None, y=None, task='classification', label_map=None, encoder=None, net=None, random_state=42, max_print_threshold=50):
    """Predict with a trained SNN: prepares X (and y/label_map), encodes spikes and returns the prediction result (with Monte-Carlo confidence for classification, values for regression); (None, 0.0) if no usable input."""
    utility = SNNUtility()
    reverse_map = None

    if X is not None and y is not None:
        print(f"[🔍] Processing X and y samples for Spiking Neural Network...")
        if isinstance(X, (str, np.str_)):
            clean_str = str(X).replace('[', '').replace(']', '')
            X = np.fromstring(clean_str, sep=' ')
        if isinstance(X, np.ndarray) and np.issubdtype(X.dtype, np.character):
            # catches arrays filled with string text
            clean_str = ' '.join(X.astype(str).flatten()).replace('[', '').replace(']', '')
            X = np.fromiter(
                    (x for x in clean_str.split() if x != "..."), dtype=float
                )   

        if isinstance(y, (str, np.str_)):
            clean_str = str(y).replace('[', '').replace(']', '')
            y = np.fromstring(clean_str, sep=' ')
        if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.character):
            # catches arrays filled with string text
            clean_str = ' '.join(y.astype(str).flatten()).replace('[', '').replace(']', '')
            y = np.fromiter(
                    (x for x in clean_str.split() if x != "..."), dtype=float
                )
    else:
        raise AttributeError('[>] Missing Samples during SNN prediction! All X and Y samples must be required for SNN prediction!')

    if len(X) < 10:
        print(f"[⚠️] Warning: Only {len(X)} samples provided. Consider providing more samples for better prediction accuracy!")

    if label_map and isinstance(label_map, dict):
        reverse_map = {v: k for k, v in label_map.items()}

    if task == 'regression':
        result = _snn_predict_regression(X, net, encoder)
        return result

    try:
        test_spikes = encoder.encode(X)
        preds, logits = net.predict(test_spikes, task=task)
        if reverse_map is not None:
            # SNN-specific: uncertainty from the encoder's own stochasticity
            mc_results = mc_confidence(net, encoder, X, n_samples=20, class_names=reverse_map)
            print(f'[>] Displaying {max_print_threshold} predictions over {len(X)} samples...')
            _ = print_confidence_report(mc_results, max_print_threshold=max_print_threshold) 
        else:
            print('[>] Label map is Missing during SNN prediction, returning SNN Logits as is without any label mapping...')
    except Exception as e:
        print(f'[⚠️] Error occured during SNN prediction: {e}')
        traceback.print_exc()
        return None, 0.0

    return mc_results
