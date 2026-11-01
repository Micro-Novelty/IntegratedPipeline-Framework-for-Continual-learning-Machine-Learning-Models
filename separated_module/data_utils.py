"""Dataset helpers: DB-string repair, normalisation config, sliding-window sequences, train/val splitting, scalar conversion."""

import json
import numpy as np
import re
from dataclasses import dataclass
from typing import Optional, Union


def fix_serialized_string(raw_db_value):
    """Repair a NormalizationConfig value read back from the database.

    Older rows stored the dataclass repr (e.g. "NormalizationConfig(feature_cols=[...], mean=array([...]))")
    instead of JSON. This unpacks tuples/lists, parses such a repr into a clean JSON string and
    returns any other value unchanged (None for empty input).
    """
    if not raw_db_value:
        return None
        
    # If the database gave us a tuple, unpack it
    if isinstance(raw_db_value, (tuple, list)):
        raw_db_value = raw_db_value[0]
        
    # If it's a string representation of the dataclass, we parse out the values manually
    if isinstance(raw_db_value, str) and raw_db_value.startswith("NormalizationConfig"):
        # Use regex to extract the lists and strings from the text representation
        try:
            # Extract feature_cols
            feature_cols = eval(re.search(r"feature_cols=(\[.*?\])", raw_db_value).group(1))
            # Extract target_col_name
            target_col_name = re.search(r"target_col_name=['\"](.*?)['\"]", raw_db_value).group(1)
            # Extract mean numbers
            mean_str = re.search(r"mean=array\(\[(.*?)\]\)", raw_db_value).group(1)
            mean = [float(x) for x in mean_str.replace(',', ' ').split()]
            # Extract std numbers
            std_str = re.search(r"std=array\(\[(.*?)\]\)", raw_db_value).group(1)
            std = [float(x) for x in std_str.replace(',', ' ').split()]
            
            # Reconstruct the perfect dictionary the compiled class needs
            clean_dict = {
                "feature_cols": feature_cols,
                "target_col_name": target_col_name,
                "mean": mean,
                "std": std
            }
            # Return it as a clean JSON string
            return json.dumps(clean_dict)
        except Exception:
            return raw_db_value

    return raw_db_value


@dataclass
class NormalizationConfig:
    """Z-score normalisation parameters for a regression dataset.

    Stores feature names, the target column, and per-feature mean/std, and can serialise to/from JSON.
    """
    feature_cols: list
    target_col_name: str
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_json(cls, data: Optional[Union[str, dict]] = None ):
        """Build a NormalizationConfig from a JSON string or dict; returns None for empty input."""
        if data is None or (isinstance(data, str) and not data.strip()):
            return None
            
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                # If it fails to parse as JSON text, check if it's already a clean string representation
                return None

        if data is None:
            return None

        data = dict(data)
        data["mean"] = np.array(data["mean"])
        data["std"] = np.array(data["std"])
        
        # **data unpacks ALL 4 keys: feature_cols, target_col_name, mean, std
        return cls(**data)

    @property
    def target_col_idx(self):
        """Index of the target column inside `feature_cols`."""
        return self.feature_cols.index(self.target_col_name)
    
    @property
    def target_mean(self):
        """Mean of the target column."""
        return self.mean[self.target_col_idx]
    
    @property
    def target_std(self):
        """Standard deviation of the target column."""
        return self.std[self.target_col_idx]
    
    def normalize(self, data):
        """Return `(data - mean) / std` using the stored statistics."""
        return (data - self.mean) / self.std

    
    def unnormalize_target(self, values_norm):
        """Convert normalised target values back to the original scale."""
        return values_norm * self.target_std + self.target_mean


def create_sequences(data, window_size, horizon=1, target_col=0):
    """Slice a 2-D array into sliding windows for time-series regression.

    Returns X with shape (n, window_size, n_features) holding all features, and y holding the next
    `horizon` values of the target column only.
    """
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i : i + window_size])                                       # semua fitur
        y.append(data[i + window_size : i + window_size + horizon, target_col])   # cuma target
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def get_dataset(df, feature_cols, target_col_name, window_size, horizon=1):
    """Build a normalised sliding-window regression dataset from a DataFrame.

    Computes mean/std over `feature_cols`, z-scores the data, windows it with `create_sequences` and
    returns (X, y, NormalizationConfig). Raises ValueError if df, feature_cols or window_size is missing.
    """
    if df is None or feature_cols is None or window_size is None:
        raise ValueError(f'[>] Some samples is missing! cannot create Dataset for regression Tasks!')

    data = df[feature_cols].values.astype(np.float32)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    
    config = NormalizationConfig(
        feature_cols=feature_cols,
        target_col_name=target_col_name,
        mean=mean,
        std=std
    )
    
    data_norm = config.normalize(data)
    X, y = create_sequences(data_norm, window_size, horizon, target_col=config.target_col_idx)
    
    return X, y, config


def _scalar(v):
    """numpy -> plain python, results stay JSON-friendly."""
    if isinstance(v, np.ndarray):
        return v.item() if v.size == 1 else v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def dynamic_test_size(y, min_val_per_class=5, min_frac=0.1, max_frac=0.3):
    """
    Pick test_size so the smallest class still gets a usable validation sample,
    without departing too far from typical train/val ratios.
    """
    classes, counts = np.unique(y, return_counts=True)
    n_total = len(y)
    smallest_class_count = counts.min()

    # fraction needed to guarantee min_val_per_class from the rarest class
    needed_frac = min_val_per_class / smallest_class_count

    # clamp to a sane range so it doesn't swing to extremes
    test_size = np.clip(needed_frac, min_frac, max_frac)
    return float(test_size)


def _prepare_train_val_split(X, y, min_val_per_class=5,
                                min_frac=0.1, max_frac=0.3, random_state=42):
    """
    Safely aligns X/y sample counts and splits into train/val,
    never guessing via transpose, always checking stratify viability.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.ravel()   # flatten (N,1) labels to (N,) for stratify compatibility

    if X.shape[0] != y.shape[0]:
        print(f'[⚠️] X/y sample count mismatch: X={X.shape[0]}, '
            f'y={y.shape[0]} — aligning to shorter length. '
            f'This likely indicates an upstream bug where X and y '
            f'became desynchronized; investigate the caller.')
        min_samples = min(X.shape[0], y.shape[0])
        X = X[:min_samples]
        y = y[:min_samples]

    if X.shape[0] < 4:
        print(f'[⚠️] Only {X.shape[0]} samples — too few for a train/val '
            f'split, returning all data as training set, no validation')
        return X, np.empty((0,) + X.shape[1:]), y, np.empty((0,) + y.shape[1:])

    dynamic_test_bed = dynamic_test_size(
        y, min_val_per_class=min_val_per_class,
        min_frac=min_frac, max_frac=max_frac
    )

    unique_classes, class_counts = np.unique(y, return_counts=True)
    can_stratify = np.all(class_counts >= 2) and len(unique_classes) > 1

    if not can_stratify:
        sparse_classes = unique_classes[class_counts < 2]
        print(f'[⚠️] Cannot stratify: class(es) with length: {len(sparse_classes.tolist())} '
            f'have fewer than 2 samples — falling back to non-stratified split')

    try:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=dynamic_test_bed,
            stratify=y if can_stratify else None,
            random_state=random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=dynamic_test_bed + min_frac, random_state=42
        )
    except ValueError as e:
        raise Warning(f'[⚠️] train_test_split failed even without stratify: {e} — '
                'The model cant learn with very Few Y Signals inside the sample,'
            f'Cannot proceed with Training and prediction and make sure to have Y samples with proper shapes!')

    return X_train, X_test, y_train, y_test, X_val, y_val
