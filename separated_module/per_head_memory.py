"""Per-attention-head k-NN memory used by the KNN-augmented transformer."""

import numpy as np
from .hnsw import HNSW


class PerHeadMemory:
    """External key/value memory with one HNSW index per attention head.

    Stores keys/values written during training and retrieves the k closest memories for each query,
    evicting the oldest fraction when `capacity` is reached.
    """
    def __init__(self, n_heads, head_dim, capacity=4000, memory_k=8,
                  metric='euclidean', M=16, evict_fraction=0.1, ef_construction=100,
                  seed=42):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.capacity = capacity
        self.memory_k = memory_k
        self.ef_construction = 100
        self.metric = metric
        self.M = M

        self.seed = seed
        self.evict_batch = max(1, int(capacity * evict_fraction))
        self._head_seeds = [(seed + h if seed is not None else None) for h in range(n_heads)]
        self.proper_seed = [self._head_seeds[h] for h in range(n_heads)]
        self.indices = [HNSW(dim=self.head_dim, M=self.M, ef_construction=self.ef_construction,
                     metric=self.metric, seed=int(self.proper_seed[h])) for h in range(n_heads)]
        self.values = [[] for _ in range(n_heads)]
        self._count = [0] * n_heads


    def write(self, K, V):
        """Store keys K and values V (shape batch, heads, seq, dim) into each head's index until capacity is reached."""
        batch, heads, seq, _ = K.shape
        for h in range(heads):
            if self._count[h] >= self.capacity:
                continue
            for b in range(batch):
                for s in range(seq):
                    if self._count[h] >= self.capacity:
                        break

                    self.indices[h].insert(K[b, h, s])
                    self.values[h].append(np.array(V[b, h, s], copy=True))
                    self._count[h] += 1
            if self._count[h] > self.capacity:
                self._evict_and_rebuild(h)


    def _evict_and_rebuild(self, h):
        keep_from = self.evict_batch
        surviving_vectors = self.indices[h].vectors[keep_from:]
        surviving_values = self.values[h][keep_from:]

        fresh_index = HNSW(dim=self.head_dim, M=self.indices[h].M,
                            ef_construction=self.indices[h].ef_construction,
                            metric=self.metric)
        for vec in surviving_vectors:
            fresh_index.insert(vec)
 
        self.indices[h] = fresh_index
        self.values[h] = surviving_values
        self._count[h] = len(surviving_values)
  

    def retrieve(self, Q):
        """For every query in Q return (K_mem, V_mem, mem_mask): the k nearest stored keys/values per position and a mask marking which slots are filled."""
        batch, heads, seq, d = Q.shape
        k = self.memory_k
        K_mem = np.zeros((batch, heads, seq, k, d), dtype=np.float32)
        V_mem = np.zeros((batch, heads, seq, k, d), dtype=np.float32)
        mem_mask = np.zeros((batch, heads, seq, k), dtype=np.float32)

        for h in range(heads):
            if self._count[h] == 0:
                continue

            for b in range(batch):
                for s in range(seq):
                    results = self.indices[h].search(Q[b, h, s], k=k, ef=max(k, 50))
                    for j, (node_id, _dist) in enumerate(results):
                        K_mem[b, h, s, j] = self.indices[h].vectors[node_id]
                        V_mem[b, h, s, j] = self.values[h][node_id]
                        mem_mask[b, h, s, j] = 1.0

        return K_mem, V_mem, mem_mask

    def to_dict(self) -> dict:
        """Capture everything needed to exactly reconstruct this memory --
        one HNSW dict per head"""
        return {
            "n_heads": self.n_heads,
            "head_dim": self.head_dim,
            "capacity": self.capacity,
            "k": self.memory_k,
            "metric": self.metric,
            "evict_batch": self.evict_batch,
            "seed": self.seed,
            "head_seeds": self._head_seeds,
            "indices": [idx.to_dict() for idx in self.indices],
            "values": [[v.tolist() for v in head_values] for head_values in self.values],
            "count": self._count,
        }
