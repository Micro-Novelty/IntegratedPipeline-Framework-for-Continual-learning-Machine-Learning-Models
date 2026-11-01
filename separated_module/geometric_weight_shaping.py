"""GeometricWeightShaping: geometry-aware weight shaping / regularisation used by Dense, Transformer and LSTM layers."""

import numpy as np
from .common import _OPT_AVAILABLE, optimized_ame_encoder, optimized_anisotropy


# geometric weight shaping provides the model with a robust geometric complexity alignment>
#  allowing it to better process data with varying geometric complexity, and providing a more stable training process in scarce data environment. 
# It can be used as a general weight initialization and shaping method for various models, especially in scenarios where data geometry is complex and data is scarce.

class GeometricWeightShaping:
    """Geometry-aware weight shaping used by Dense, Transformer and LSTM layers.

    Measures properties of the input data (AME - an adaptive magnitude/entropy score - anisotropy and
    spectral signatures) and uses them to shape weight initialisations and updates. Uses the Cython
    kernels for AME/anisotropy when available.
    """
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
    

    def eigenvalue_encoder(self, x):
        """Encode the eigen-structure of x; returns (k, floating_point, structured_noise), the values used to build spectral signatures and condition weights."""
        eps = 1e-5
        raw_X = np.asarray(x)
        AME = self.AME_Encoder(raw_X)  
        AMR = 1.0 / (1.0 + np.exp(-AME)) + eps
        mag = np.mean(np.linalg.norm(raw_X, axis=-1))

        if raw_X.ndim > 2:
            raw_X = raw_X.reshape(raw_X.shape[0], -1)

        anisotropy = self.anisotropy_measurement(raw_X)

        structured_noise = np.random.uniform(0, mag, size=raw_X.shape)
        X = np.vstack((raw_X, structured_noise))
        if X.ndim == 2 and X.shape[1] == 1:
            X = np.hstack((raw_X, structured_noise))

        cov = np.cov(X, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]

        energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        energy_sigmoid_growth = 1.0 / (1.0 + np.exp(-energy))
        energy_consistency = np.std(energy_sigmoid_growth)
        k = np.searchsorted(energy, 0.90) + 1     # +1 converts 0-based index to count

        trA = k / (1.0 - anisotropy) + eps  
        trB = (1/2 + energy_consistency) / (1.0 + trA**2)
        trC = (1/6 + AMR) / (1.0 - trB**2) + eps

        if np.isnan(trC) or np.isinf(trC):
            trC = anisotropy * (trB**2 - 1.0) + eps
            if np.isnan(trC) or np.isinf(trC):
                trC = (1.0 - AMR)

        min_val = min(trC, 0) 
        max_val = max(trC, 0) 
        floating_point = np.random.uniform(min_val, max_val, size=X.shape) 
        return k, floating_point, structured_noise


    def spectral_signature(self, x, structured_noise, k=5):
        """Return the top-k spectral signature (leading singular/eigen values, noise-regularised) of x flattened to 2-D."""
        raw_X = np.asarray(x, dtype=np.float64)
        if raw_X.ndim > 2:
            X = raw_X.reshape(raw_X.shape[0], -1)
        else:
            X = raw_X.reshape(raw_X.shape[0], -1)

        X = np.atleast_2d(X)

        if X.ndim == 2 and X.shape[1] == 1:
            # normalize structured_noise to 2D matching X's row count
            noise = np.asarray(structured_noise, dtype=np.float64)

            if noise.ndim == 1:
                # reshape to (n_samples, n_noise_features)
                # if noise length matches X's row count, treat as column vector
                if noise.shape[0] == X.shape[0]:
                    noise = noise.reshape(-1, 1)
                else:
                    # noise is a flat feature vector not aligned to X's rows —
                    # broadcast it across all rows instead of stacking blindly
                    noise = np.tile(noise.reshape(1, -1), (X.shape[0], 1))
            elif noise.ndim > 2:
                noise = noise.reshape(noise.shape[0], -1)

            # align row counts before hstack
            if noise.shape[0] != X.shape[0]:
                min_rows = min(noise.shape[0], X.shape[0])
                X     = X[:min_rows]
                noise = noise[:min_rows]
                print(f'[⚠️] spectral_signature: row count mismatch, '
                    f'truncated to {min_rows} rows')

            X = np.hstack((X, noise))

        # guard against degenerate covariance — need at least 2 samples
        if X.shape[0] < 2:
            print(f'[⚠️] spectral_signature: only {X.shape[0]} sample(s), '
                f'cannot compute covariance — returning zeros')
            return np.zeros(k)

        try:
            cov     = np.cov(X, rowvar=False, ddof=1)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]
            eig_sum = eigvals.sum()
            if eig_sum <= 1e-8:
                return np.zeros(min(k, len(eigvals)))
            return eigvals[:k] / (eig_sum + 1e-8)
        except np.linalg.LinAlgError as e:
            print(f'[⚠️] spectral_signature: eigendecomposition failed: {e}')
            return np.zeros(k)


    def spectral_similarity(self, a, b, structured_noise):
        """Similarity in (0, 1] between two inputs: exp(-||signature(a) - signature(b)||)."""
        sa = self.spectral_signature(a, structured_noise)
        sb = self.spectral_signature(b, structured_noise)
        if sa.shape != sb.shape:
            min_rows = min(sa.shape[0], sb.shape[0])

            sa = sa[:min_rows]
            sb = sb[:min_rows]

        return np.exp(-np.linalg.norm(sa - sb))

    # abstract modelling error provides the model how to better process weights when the data complexity has little geometric complexity
    def AME_Encoder(self, x):
        """Compute the AME score of x (0.0 for empty input). Uses the Cython kernel for 2-D float input when available."""
        X = np.asarray(x)

        if len(X) == 0:
            print('[!] X size is 0, AME Will be replaced by minimum confidence threshold')
            return 0.0

        if _OPT_AVAILABLE and np.asarray(X).ndim == 2:
            return optimized_ame_encoder(np.asarray(X, dtype=np.float64))     

        try:
            gradient = np.gradient(x)
        except:
            subnet = x[:min(10, x.shape[0]), :min(10, x.shape[1])]
            gradient = np.gradient(subnet.flatten())

        mean_vector_mag  = np.mean(np.linalg.norm(gradient, axis=-1))       
        X_mag = np.mean(np.linalg.norm(X, axis=-1))
        # Regular AME Equations, higher AME provides capabilities for the model to experience errors during abstraction
        # Lower AME means lower chance for un optimal abstraction.

        AME =  np.log1p(X_mag) * np.log1p(mean_vector_mag) 
        return AME

    # anisotropy provides the model the standard complexity of the data geometry, allowing it to know how complex the data needs to be processed.
    def anisotropy_measurement(self, x):
        """Measure how directionally uneven (anisotropic) the data x is; used to scale weight initialisation."""
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

    # weight shaping provides directional context in which how the data should be processed in order to align with the data geometry
    def abstract_weight_shaping(self, x):
        """Shape weights for the layer from the data's anisotropy and magnitude; returns the resulting weight array (abstract_context)."""
        input_size = self.input_size
        output_size = self.output_size


        eps = 1e-5
        x = np.asarray(x)

        rng = np.random.default_rng()

        anisotropy = self.anisotropy_measurement(x)
        mag = np.mean(np.linalg.norm(x))

        k, floating_point, structured_noise = self.eigenvalue_encoder(x)
        AME = self.AME_Encoder(x)
        AMR = 1.0 / (1.0 + np.exp(-AME)) # abstract modelling rate        

        spectral_similarity = self.spectral_similarity(x, floating_point, structured_noise)

        AEL = (0.3 + spectral_similarity + eps) * anisotropy 
        scaled_anisotropy = anisotropy / (anisotropy + 1.0)
        
        abstraction_efficiency = (1.0 + AEL) * (1.0 - AMR) + eps
        abstraction_efficiency = (k + AEL) * (1.0 - AMR) + eps

        if np.isnan(abstraction_efficiency) or np.isinf(abstraction_efficiency):
            abstraction_efficiency = (1 - AMR) + eps

        abstract_context = rng.uniform(0, abstraction_efficiency, size=(input_size, output_size)) 

        return abstract_context



    def weight_shaping(self, x, type=None):
        """Sanitise x (NaN/inf, >2-D shapes, zero variance) and return shaped weights (abstract_context) for a layer; `type` selects the shaping variant."""
        if np.isnan(x).any() or np.isinf(x).any():
            x = np.nan_to_num(x, nan=0.0, posinf=1e99, neginf=-1e99)     

        if isinstance(x, list):
            x = np.asarray(x)

        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        if np.std(x) == 0:
            x = np.random.uniform(0, 1, size=x.shape)

        abstract_context = self.abstract_weight_shaping(x)
        abstract_context /= np.max(np.abs(abstract_context)) + 1e-5  # Normalized to [-1, 1]

        return abstract_context
