"""Data-aware ("calibrated") weight initialisation and loading of calibrated weights into QAT layers."""

import numpy as np


def _ame_and_anisotropy(X):
    """AME and anisotropy same copy as in GeometricWeightShaping (np.gradient over sample and feature axes)."""
    try:
        g = np.gradient(X)
        mean_vec_mag = np.mean(np.linalg.norm(g, axis=-1))
        norms = [np.linalg.norm(v) for v in g]
        aniso = np.std(norms) / (np.mean(norms) + 1e-5)
    except ValueError:                       # axis too small for a gradient (e.g. a 1-wide layer or 1 sample)
        mean_vec_mag, aniso = 0.0, 0.0
    x_mag = np.mean(np.linalg.norm(X, axis=-1))
    ame = np.log1p(x_mag) * np.log1p(mean_vec_mag)
    return float(ame), float(aniso)


def _principal_directions(X):
    """Eigenvectors of the input covariance (descending) and k = #components covering 90% of the variance."""
    d = X.shape[1]
    if X.shape[0] < 2:
        return np.eye(d), d
    cov = np.atleast_2d(np.cov(X - X.mean(0, keepdims=True), rowvar=False))
    ev, V = np.linalg.eigh(cov)
    ev, V = ev[::-1].clip(min=0), V[:, ::-1]
    p = ev / (ev.sum() + 1e-12)
    k = int(np.searchsorted(np.cumsum(p), 0.90) + 1)
    return V, max(1, min(k, d))


def _uniform_layer(X, fan_out, rng, *, last, last_layer_var):
    """Calibrated uniform control: U(-1,1) base + the same variance calibration"""
    W = rng.uniform(-1.0, 1.0, (X.shape[1], fan_out))
    z = X @ W
    W *= np.sqrt((last_layer_var if last else 2.0) / (np.mean(z ** 2) + 1e-12))
    return W, dict(base='uniform')


# ----------------------------------------- one layer -------
def _layer(X, fan_out, rng, *, last, gain_slope, last_layer_var, align):
    """Build one calibrated layer: weights initialised from the data's principal directions
    and scaled by the data's anisotropy so activations stay well-conditioned (optionally `align`-ed).
    """
    eps = 1e-12
    n, d = X.shape
    ame, aniso = _ame_and_anisotropy(X)
    aniso = float(np.clip(aniso, 0.0, 0.95))
    if not align:
        aniso = 0.0
    V, k = _principal_directions(X)
 
    # (1) direction: variance-preserving mix of random and PC-aligned components
    W_rand = rng.standard_normal((d, fan_out))
    W_align = V[:, :k] @ rng.standard_normal((k, fan_out))
    W_rand /= W_rand.std() + eps
    W_align /= W_align.std() + eps
    W = np.sqrt(1.0 - aniso) * W_rand + np.sqrt(aniso) * W_align
 
    # (2) gain: centred on structureless data of the same shape
    ame_ref, _ = _ame_and_anisotropy(np.random.default_rng(0).standard_normal(X.shape))
    gain = 1.0 + gain_slope * np.tanh(ame - ame_ref)
 
    # (3) calibrate pre-activation second moment
    z = X @ W
    target = (last_layer_var if last else 2.0) * gain ** 2
    W *= np.sqrt(target / (np.mean(z ** 2) + eps))
    return W, dict(ame=ame, aniso=aniso, k=k, gain=float(gain))


def _sanitize_string_chars(x):
    """Turn a stringified array (e.g. '[0.1 0.2 ...]') or character array back into a numeric numpy array."""
    if isinstance(x, (str, np.str_)):
        clean_str = str(x).replace('[', '').replace(']', '').replace('...', '').strip()
        x = np.fromstring(clean_str, sep=' ')
    if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.character):
        clean_str = ' '.join(x.astype(str).flatten()).replace('[', '').replace(']', '')
        skip_values = {'...', 'NaN', 'null'}
        x = np.fromiter((v for v in clean_str.split() if v not in skip_values), dtype=float)
    return x


# ----------------------------------------------------- init--------------
def calibrated_init(sizes, X, seed=0, *, gain_slope=0.2, last_layer_var=1.0, align=True,
             max_samples=4096, return_info=False, bits=None, base='auto', flat_max_bits=2):
    """Weights for an MLP with ReLU hidden layers and a linear output layer.
 
    sizes   : [in_dim, h1, ..., n_out]
    X       : (n_samples, in_dim) training inputs (roughly standardized)
    seed    : makes the draw reproducible
    align   : False switches the PCA-alignment off (ablation: calibrated random weights with the tanh gain)
    bits    : weight bit-width you will quantize to. With base='auto', bits <= flat_max_bits (default 2) selects the
              calibrated-uniform base; None or larger keeps the geometric sketch-tanh.
    base    : 'auto' | 'geometric' | 'uniform'  (force one or the other regardless of bits)
    Returns : list of float64 arrays, W[i].shape == (sizes[i], sizes[i+1]); biases should start at zero.
    """
    
    X = _sanitize_string_chars(np.asarray(X, dtype=np.float64))
    if bits is None or not isinstance(bits, int):
        print(f'[INFO] Bit is unsupported format -> bit data type: {type(bits)}')
        if not isinstance(bits, int) and isinstance(bits, float):
            bits = int(bits)
        if bits is None:
            bits = 7

    if X.ndim != 2 or X.shape[1] != sizes[0]:
        raise ValueError(f'[>] X must be (n, {sizes[0]}), got {X.shape}')
    if X.shape[0] > max_samples:                  # keep row order (AME uses np.gradient over rows)
        idx = np.sort(np.random.default_rng(seed + 12345).choice(X.shape[0], max_samples, replace=False))
        X = X[idx]
    if base not in ('auto', 'geometric', 'uniform'):
        raise ValueError("base must be 'auto', 'geometric' or 'uniform'")
    use_uniform = base == 'uniform' or (base == 'auto' and bits is not None and bits <= flat_max_bits)
    rng = np.random.default_rng(seed)
    Ws, infos, h = [], [], X
    for i, fan_out in enumerate(sizes[1:]):
        last = i == len(sizes) - 2
        if use_uniform:
            W, info = _uniform_layer(h, fan_out, rng, last=last, last_layer_var=last_layer_var)
        else:
            W, info = _layer(h, fan_out, rng, last=last, gain_slope=gain_slope,
                             last_layer_var=last_layer_var, align=align)
            info['base'] = 'geometric'
        if not np.all(np.isfinite(W)):
            raise RuntimeError(f'[>] non-finite weights in layer {i}; check X for NaN/inf or constant columns')
        Ws.append(W)
        infos.append(info)
        if not last:
            h = np.maximum(0.0, h @ W)
    return (Ws, infos) if return_info else Ws


def describe(Ws, bits=(8, 4, 2)):
    """Per-layer print: weight stats and the fraction of weights that survive quantization
    (per-tensor symmetric max-abs scaling, levels +-(2^(b-1)-1)). Very low survival at 2 bits => a near-dead start."""
    for i, W in enumerate(Ws):
        row = f'[LAYERS] layer {i + 1}: shape={W.shape} || std={W.std():.3f} || min={W.min():+.2f} || max={W.max():+.2f} || frac>0={np.mean(W > 0):.2f}'
        for b in bits:
            qmax = 2 ** (b - 1) - 1
            scale = np.abs(W).max() / qmax
            row += f'[>] nonzero@w{b}={np.mean(np.round(np.abs(W) / scale) >= 1):.2f}'
        print(row)


def load_into_qat_layers(model, X, verbose=True, seed=0, **kw):
    """Assign the init to MLP layers
    """
    layers = model.layers
    sizes = [layers[0].input_size] + [l.output_size for l in layers]
    Ws = calibrated_init(sizes, X, seed, **kw)
    if verbose:
        describe(Ws)

    for l, W in zip(layers, Ws):
        l.W = np.ascontiguousarray(W, dtype=np.float64)
        if hasattr(l, 'b'):
            l.b = np.zeros((1, W.shape[1]))
        if hasattr(l, '_init_optimizer'):
            l._init_optimizer()                   # rebuild Adam state for the new weights
    return model
