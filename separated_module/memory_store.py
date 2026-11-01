"""SQLite-backed "model memory" primitives: schema, connection helpers, CKA similarity, set/save memory."""

import io
import json
import numpy as np
import os
import random
import secrets
import sqlite3
import sys
import traceback
from dataclasses import asdict
from contextlib import closing
from .common import APP_BASE_DIR
from .data_utils import NormalizationConfig, fix_serialized_string


# ________________________
# MEMORY HANDLER PRIMITIVES
#______________________________

_SCHEMA = """CREATE TABLE IF NOT EXISTS model_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_name TEXT,
    model_sample BLOB,
    x_shape TEXT,
    y_shape TEXT,
    label_csv TEXT,
    task TEXT,
    label_map TEXT,
    feature_cols TEXT,
    config TEXT,
    is_active INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""


def get_database_path(db_path):
    """Resolve a database filename to an absolute path.

    Absolute paths are returned unchanged. Otherwise the file is placed next to the executable (frozen
    builds), next to the project directory, or in the current working directory as a last resort.
    """
    if os.path.isabs(db_path):
        return db_path
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)      # persistent, unlike _MEIPASS
    elif '__file__' in globals():
        base = APP_BASE_DIR
    else:
        base = os.getcwd()
    return os.path.join(base, db_path)


def _connect(db_path):
    """Open the SQLite database (creating the model_memory table if needed) and return the connection."""
    conn = sqlite3.connect(get_database_path(db_path), timeout=10)
    conn.execute(_SCHEMA)          # table always exists
    return conn


def _to_array(a):
    """Convert a tensor / list / array into a plain numpy array (detaching torch tensors first)."""
    if hasattr(a, 'detach'):                       # torch tensor
        a = a.detach().cpu().numpy()
    a = np.asarray(a)
    if a.dtype == object:
        raise ValueError("object-dtype arrays are not supported")
    return a


def _shape_str(a):
    """Return an array's shape as a tuple string, e.g. '(32, 10)', used as a lookup key."""
    return str(tuple(int(s) for s in a.shape))


def setup_memory_table(db_path):
    """Ensure the model_memory table exists. Returns True on success, False on any DB error."""
    try:
        with closing(_connect(db_path)) as conn:
            conn.commit()
        return True
    except sqlite3.Error as e:
        print(f'[>] Cannot set up database: {e}')
        return False


def linear_cka(X, Y, eps=1e-12):
    """Linear Centered Kernel Alignment between two representations X and Y.

    Returns a similarity in [0, 1] (1 = identical up to rotation/scale). Used to decide whether a
    stored memory resembles the current data.
    """
    X = _to_array(X).astype(np.float64)
    Y = _to_array(Y).astype(np.float64)
    if X.ndim == 0 or Y.ndim == 0:
        raise ValueError("inputs must have at least 1 dimension")
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")
    n = X.shape[0]
    if n < 2:
        raise ValueError("CKA requires at least 2 samples.")
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        raise ValueError("X and Y must not contain NaN or inf.")

    X -= X.mean(axis=0, keepdims=True)
    Y -= Y.mean(axis=0, keepdims=True)

    # normalise first so the zero-variance check is scale-independent
    nx, ny = np.linalg.norm(X), np.linalg.norm(Y)
    if nx < eps or ny < eps:
        return 0.0
    X /= nx
    Y /= ny

    if max(X.shape[1], Y.shape[1]) > n:            # Gram form: n x n
        K, L = X @ X.T, Y @ Y.T
        hsic = np.sum(K * L)
        denom = np.linalg.norm(K, 'fro') * np.linalg.norm(L, 'fro')
    else:                                           # feature form: d x d
        hsic = np.linalg.norm(X.T @ Y, 'fro') ** 2
        denom = np.linalg.norm(X.T @ X, 'fro') * np.linalg.norm(Y.T @ Y, 'fro')

    if denom < eps:
        return 0.0
    return float(min(max(hsic / denom, 0.0), 1.0))


def set_memory(db_path, x, y, threshold=0.95, new_memory=None):
    """Load the stored memory most similar to the given (x, y) samples.

    Looks up active memories with the same x/y shapes, scores each by the average linear CKA of x and y
    and, if the best score exceeds `threshold`, returns
    (memory_name, label_csv, task, label_map, feature_cols, NormalizationConfig). Otherwise returns a
    default tuple (new_memory, None, ...) so the caller creates a fresh memory.
    """
    if new_memory is None:
        new_memory = f'memory-{secrets.token_hex(3)}-{random.randint(1000, 1000000000)}"'

    table = setup_memory_table(db_path)
    if table:
        print('[DATABASE] Database condition is OK')
    else:
        raise Warning('[DATABASE] Database experiences error, aborting memory load...')

    if x is None:
        raise ValueError('[>] X sample cant be None! ')
    if y is None:
        raise ValueError('[>] y sample cant be None! ')

    default = (new_memory, None, None, None, None, None)
    try:
        x, y = _to_array(x), _to_array(y)
        with closing(_connect(db_path)) as conn:
            rows = conn.execute(
                """SELECT model_sample, memory_name, label_csv, task, label_map, feature_cols, config FROM model_memory
                   WHERE x_shape = ? AND y_shape = ? AND is_active = 1
                   ORDER BY id DESC""",
                (_shape_str(x), _shape_str(y))).fetchall()

        if not rows:
            print(f'[>] No saved memory for {new_memory}')
            return default

        best_score, best = -1.0, None
        for blob, name, label_csv, task, label_map, feature_cols, config in rows:
            try:
                with np.load(io.BytesIO(bytes(blob)), allow_pickle=False) as data:
                    xd, yd = data['x'], data['y']
                score = (linear_cka(x, xd) + linear_cka(y, yd)) / 2
            except Exception as e:
                print(f'[!] Skipping unreadable memory {name!r}: {e}')
                continue
            if score > best_score:
                best_score, best = score, (name, label_csv, task, json.loads(label_map), json.loads(feature_cols) if feature_cols is not None else None, 
                NormalizationConfig.from_json(fix_serialized_string(config)) if fix_serialized_string(config) else None)

        if best and best_score > threshold:
            print(f'[MEMORY] Loaded memory {best[0]!r} (similarity {best_score:.1%})')
            print(f'[>] Loaded task: {best[2]!r}')
            print(f'[>] Loaded label csv: {best[1]!r}')
            print(f'[>] Loaded label map {best[3]!r} (type: {type(best[3])})')
            return best

        print(f'[MEMORY] No memory above threshold (best {best_score:.1%})')
        return default

    except Exception as e:
        print(f'[!] Memory load failed: {e}')
        traceback.print_exc()
        return default


def save_memory(db_path, memory_name, label_csv, X, Y, task, label_map, feature_cols=None, config=None):
    """Persist a memory (sample x/y arrays plus metadata) into the model_memory table.

    Stores the arrays as an .npz blob together with name, label csv, task, label map, feature columns
    and the optional NormalizationConfig. Skips saving if the required arguments are missing.
    """
    try:
        config_dict = None
        all_none = memory_name is None or label_csv is None or X is None or Y is None or task is None or label_map is None

        if all_none:
            raise ValueError('[>] Some sample/samples are None, make sure to pass all Samples correctly!')

        X, Y = _to_array(X), _to_array(Y)
        buf = io.BytesIO()
        np.savez_compressed(buf, x=X, y=Y)

        if config is not None:
            config_dict = asdict(config)

            # Convert numpy arrays to standard lists so json.dumps works natively
            config_dict["mean"] = config_dict["mean"].tolist()
            config_dict["std"] = config_dict["std"].tolist()

        with closing(_connect(db_path)) as conn, conn:   # `with conn` = transaction
            conn.execute("UPDATE model_memory SET is_active = 0 WHERE memory_name = ?",
                         (memory_name,))
            conn.execute(
                """INSERT INTO model_memory
                   (memory_name, model_sample, x_shape, y_shape, label_csv, task, label_map, feature_cols, config, is_active)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                (memory_name, sqlite3.Binary(buf.getvalue()),
                 _shape_str(X), _shape_str(Y), label_csv, task, json.dumps(label_map, default=str),
                json.dumps(feature_cols, default=str), 
                json.dumps(config_dict)))

        print('[MEMORY] Memory saved!')
    except Exception as e:
        print(f'[MEMORY] Memory save failed: {e}')
        traceback.print_exc()
