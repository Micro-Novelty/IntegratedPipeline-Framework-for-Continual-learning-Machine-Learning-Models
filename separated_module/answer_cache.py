"""AccurateAnswerCache: cache of verified answers."""

import hashlib
import numpy as np
from datetime import datetime


class AccurateAnswerCache:
    """Cache of user-verified predictions.

    Entries (features, token ids, prediction, confidence) are matched by exact hash or by similarity
    above `similarity_threshold`, so already-confirmed answers can be returned without re-running models.
    Persisted through ModelStorage.
    """
    def __init__(self, pipeline, similarity_threshold=0.85, max_size=500):
        self.pipeline = pipeline
        self.memory_name = self.pipeline.memory_name
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.max_threshold = 0.7

        self.memory_exist = self.pipeline.storage.memory_exists(self.memory_name, type='Accurate-cache')
        if self.memory_exist:
            self.cache = self.pipeline.storage.load_accurate_cache(self.memory_name)
        else:
            self.cache = {}

        self.exact_hash_index = {}   # O(1) exact match lookup

    def _flatten_indices(self, input_ids):
        print('[=] Handling Input indices for inhomogenous shape checks.')
        try:
            try:
                ids = np.asarray(input_ids).ravel()
                return ids
            except:
                def flatten(x):
                    for item in x:
                        if isinstance(item, (list, tuple)):
                            yield from flatten(item)
                        else:
                            yield item

                if isinstance(input_ids, (list, tuple)):
                    flat_ids = list(flatten(input_ids))
                else:
                    flat_ids = input_ids  # already a flat array/tensor

                ids = np.asarray(flat_ids).ravel()
        except:
            flat_ids = self.pipeline._safe_to_2d_float(input_ids)
            ids = np.asarray(flat_ids).ravel()

        return ids

    def _adapt_ids_shape(self, ids_a, ids_b):
        try:
            flat_ids_a = self.pipeline._safe_to_2d_float(ids_a)
            flat_ids_b = self.pipeline._safe_to_2d_float(ids_b)

            if flat_ids_a.shape != flat_ids_b.shape:
                min_rows = min(flat_ids_a.shape[0], flat_ids_b.shape[0])
                min_cols = min(flat_ids_a.shape[1], flat_ids_b.shape[1])
                flat_ids_a = flat_ids_a[:min_rows, :min_cols]
                flat_ids_b = flat_ids_b[:min_rows, :min_cols]                          

            return flat_ids_a, flat_ids_b
        except Exception as e:
            print(f'[!] cannot adapt indices shape due to: {e}')
            return ids_a, ids_b
            

        
    def add_verified(self, x_mlp, input_ids, prediction, confidence, index,
                     source='user_confirmed'):

        """Store a verified prediction (with its features, ids, confidence and source), evicting old entries beyond `max_size`."""
        try:
            key = self._make_key(x_mlp)

            # hash input_ids for fast exact-match 
            ids_hash = self._hash_ids(input_ids) if input_ids is not None else None

            flat_ids = self._flatten_indices(input_ids)
            entry = {
                'x_mlp'      : np.asarray(x_mlp, dtype=np.float64).ravel(),
                'input_ids'  : flat_ids if flat_ids is not None else None,
                'ids_hash'   : ids_hash,
                'prediction' : prediction,
                'confidence' : float(confidence),
                'index'      : int(index) if index is not None else None,
                'source'     : source,
                'hit_count'  : 0,
                'added_at'   : datetime.now().isoformat(),
                'last_hit'   : None
            }

            self.cache[key] = entry

            # maintain O(1) exact match index
            if ids_hash is not None:
                self.exact_hash_index[ids_hash] = key

            if len(self.cache) > self.max_size:
                self._evict_lru()

            print(f'[💎] Verified answer cached: {prediction} (source={source})')

            if self.cache[key]['source'] != 'automatic_verified' and not source.startswith('automatic'):
                self.pipeline.storage.save_accurate_cache_dict(self.memory_name, self.cache)
            
        except Exception as e:
            print(f'[!] Failed to add samples and answer to Answer cache due to: {e}')


    def lookup(self, x_mlp, input_ids=None):
        """Find a cached verified answer for (x_mlp, input_ids) by exact hash or feature similarity; returns the cached entry as a dict, or None."""
        try:
            confidence_threshold = self.pipeline.confidence_threshold
            if confidence_threshold <= 0.5:
                confidence_treshold = self.max_threshold

            if not self.cache:
                return None
            
            AME = self.pipeline.AME_Encoder(x_mlp)
            anisotropy = self.pipeline.anisotropy_measurement(x_mlp)
            if np.isinf(AME) or np.isnan(AME):
                AME = (1.0 - self.pipeline.confidence_threshold)
            if np.isinf(anisotropy) or np.isnan(anisotropy):
                anisotropy = 0.15

            # FAST PATH — O(1) hash lookup 
            if input_ids is not None:
                ids_hash = self._hash_ids(input_ids)
                if ids_hash in self.exact_hash_index:
                    key   = self.exact_hash_index[ids_hash]
                    entry = self.cache.get(key)
                    if isinstance(entry, dict) and entry['source'] != 'automatic_verified' and not entry['source'].startswith('automatic'):
                          
                        entry['hit_count'] += 1
                        entry['last_hit']   = datetime.now().isoformat()
                        return {
                            'prediction' : entry['prediction'],
                            'confidence' : entry['confidence'],
                            'index'      : entry['index'],
                            'similarity' : 1.0,
                            'source'     : entry['source'],
                            'hit_count'  : entry['hit_count'],
                            'match_type' : 'exact_ids'
                        }  

                    if entry is not None and isinstance(entry, dict):
                        entry['hit_count'] += 1
                        entry['last_hit']   = datetime.now().isoformat()
                        return {
                            'prediction' : entry['prediction'],
                            'confidence' : entry['confidence'],
                            'index'      : entry['index'],
                            'similarity' : 1.0,
                            'source'     : entry['source'],
                            'hit_count'  : entry['hit_count'],
                            'match_type' : 'exact_ids'
                        }

            # SIMILARITY PATH — same as before, x_mlp + input_ids combined
            x_mlp = np.asarray(x_mlp, dtype=np.float64).ravel()
            best_match = None
            best_combined_sim = 0.0

            for entry in self.cache.values():
                mlp_sim = self.pipeline.cosine_robust_similarity(x_mlp, entry['x_mlp'])

                seq_sim = 1.0
                if input_ids is not None and entry['input_ids'] is not None:
                    ids_a = self._flatten_indices(input_ids)
                    ids_b = entry['input_ids'].ravel()

                    if ids_a.shape != ids_b.shape:
                        ids_a, ids_b = self._adapt_ids_shape(ids_a, ids_b)

                    min_len = min(len(ids_a), len(ids_b))
                    if min_len > 0:
                        seq_sim = float(np.mean(ids_a[:min_len] == ids_b[:min_len]))

                combined_env_sim = ((mlp_sim + seq_sim) / 2 * anisotropy) 
                deterministic_modelling_sim = ((mlp_sim + seq_sim) / 2 * AME)  

                combinatorial_absolute_factor = (
                    deterministic_modelling_sim > best_combined_sim and 
                    AME < 0.5 and 
                    entry['confidence'] > 0.5
                )
                dynamic_environmental_factor = (
                    combined_env_sim > best_combined_sim and 
                    AME > 0.5 and
                    entry['confidence'] > confidence_threshold
                )

                if combinatorial_absolute_factor:
                    best_combined_sim = deterministic_modelling_sim
                else:
                    best_combined_sim = combined_env_sim

                best_match = entry

            if best_match and best_combined_sim >= self.similarity_threshold:
                best_match['hit_count'] += 1
                best_match['last_hit']   = datetime.now().isoformat()
                return {
                    'prediction' : best_match['prediction'],
                    'confidence' : best_match['confidence'],
                    'index'      : best_match['index'],
                    'similarity' : float(best_combined_sim),
                    'source'     : best_match['source'],
                    'hit_count'  : best_match['hit_count'],
                    'match_type' : 'feature_similarity'
                }
                
            return None

        except Exception as e:
            print(f'[!] Cant search for Correct answer in cache due to: {e}')
            return None

    def _hash_ids(self, input_ids):
        try:
            print('[=] Creating a hash ID ')
            ids = self._flatten_indices(input_ids)
            return hashlib.md5(ids.tobytes()).hexdigest()
        except Exception as e:
            print(f'[!] cannot create a Hash ID for samples due to: {e}')
            return None

    def _make_key(self, x_mlp):
        try:
            def flatten(x):
                for item in x:
                    if isinstance(item, (list, tuple)):
                        yield from flatten(item)
                    else:
                        yield item

            if isinstance(x_mlp, (list, tuple)):
                flat_x = list(flatten(x_mlp))
            else:
                flat_x = x_mlp  # already a flat array/tensor

            x = np.asarray(flat_x, dtype='<i4').ravel()       
        except:
            flat_x = self.pipeline._safe_to_2d_float(x)
            x = np.asarray(flat_x, dtype='<i4').ravel()

        return hashlib.md5(x.tobytes()).hexdigest()

    def _evict_lru(self):
        try:
            if not self.cache:
                return
            lru_key = min(
                self.cache.keys(),
                key=lambda k: (self.cache[k]['hit_count'], self.cache[k]['added_at'])
            )
            entry = self.cache[lru_key]
            if entry.get('ids_hash'):
                self.exact_hash_index.pop(entry['ids_hash'], None)
            del self.cache[lru_key]
        except Exception as e:
            print(f'[!] cannot delete certain cache due to: {e}')
