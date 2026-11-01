"""Shared setup for the AbstractIntegratedModule package.

Holds the optional accelerators (Cython kernels and the Rust weight-storage
module), their availability flags, the pipeline-wide lock and APP_BASE_DIR.
When an accelerator is missing, its names are bound to ``None``; all callers
must check ``_OPT_AVAILABLE`` / ``_RUST_MODULE_AVAILABLE`` before using them.
"""
import os
import threading

# Directory that contains the package (where the original single-file module
# lived). Used to locate databases, certificates and other side files so that
# they resolve exactly as they did before the split.
APP_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Optimized Modules In Cython implementation.
try:
    from AbstractOptimizedModules import (
        optimized_sigmoid,
        optimized_sigmoid_deriv,
        optimized_tanh_deriv,
        optimized_lstm_cell_forward,
        optimized_project_heads,
        optimized_ame_encoder,
        optimized_anisotropy,
        optimized_cosine_similarity,
        optimized_softmax_2d,
        optimized_dynamic_weighted_ensemble,
        optimized_qkv_weight_grad,
        optimized_qkv_input_grad,
        optimized_lstm_cell_backward,
    )
    _OPT_AVAILABLE = True
    print('[=] Cython acceleration loaded ✅')
except ImportError as e:
    _OPT_AVAILABLE = False
    optimized_sigmoid = None
    optimized_sigmoid_deriv = None
    optimized_tanh_deriv = None
    optimized_lstm_cell_forward = None
    optimized_project_heads = None
    optimized_ame_encoder = None
    optimized_anisotropy = None
    optimized_cosine_similarity = None
    optimized_softmax_2d = None
    optimized_dynamic_weighted_ensemble = None
    optimized_qkv_weight_grad = None
    optimized_qkv_input_grad = None
    optimized_lstm_cell_backward = None
    print(f'[=] Cython not available: {e}, using numpy fallback')

try:
    import abstract_weights_core as wc
    _RUST_MODULE_AVAILABLE = True
    print('[=] Rust weight storage loaded ✅')
except ImportError as e:
    wc = None
    _RUST_MODULE_AVAILABLE = False
    print(f'[=] Rust weight storage unavailable due to: {e}, using Python sqlite3 fallback')

# Lock guarding IntegratedPipeline construction / shared state.
_integrated_pipeline_lock = threading.Lock()
