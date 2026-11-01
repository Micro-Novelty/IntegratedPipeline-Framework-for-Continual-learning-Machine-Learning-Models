"""HNSW approximate nearest-neighbour index."""

import heapq
import math
import numpy as np
import random
from typing import List


class HNSW:
    """Hierarchical Navigable Small World index for approximate nearest-neighbour search.

    Supports 'euclidean' and 'cosine' metrics; vectors are inserted one at a time and queried with `search`.
    """
    def __init__(self, dim, M=16, ef_construction=100, metric='euclidean', seed=None):
        self.dim = dim
        self.M= M
        self.M0 = 2 * M
        self.ef_construction = ef_construction
        self.metric = metric
        self.mL = 1.0 / np.log(M)

        self.vectors: List[np.ndarray] = []
        self.neighbors: List[List[set]] = []
        self.max_level_of: List[int] = []

        self.entry_point = -1
        self.max_level = -1
        self.seed = seed
        self._rng = random.Random(self.seed)


    def _distance(self, a, b):
        if self.metric == 'euclidean':
            return float (np.linalg.norm(a - b))
        if self.metric == 'cosine':
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
            return 1.0 - float(np.dot(a, b) / denom)
        raise ValueError(f'Unknown Metric: {self.metric}')

    def _random_level(self):
        return int(-math.log(self._rng.random()) * self.mL)

    def _search_layer(self, query, entry_points, ef, layer):
        visited = set(entry_points)
        candidates = [(self._distance(query, self.vectors[ep]), ep) for ep in entry_points]
        heapq.heapify(candidates)
        found = [(-d, n) for d, n in candidates]
        heapq.heapify(found)

        while candidates:
            dist_c, c = heapq.heappop(candidates)
            worst_found = -found[0][0]
            if dist_c > worst_found and len(found) >= ef:
                break
            for neighbor in self.neighbors[c][layer]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                d = self._distance(query, self.vectors[neighbor])
                worst_found = -found[0][0]
                if d < worst_found or len(found) < ef:
                    heapq.heappush(candidates, (d, neighbor))
                    heapq.heappush(found, (-d, neighbor))
                    if len(found) > ef:
                        heapq.heappop(found)
        return [(-d, n) for d, n in found]

    def _select_neighbors_heuristic(self, query, candidates, m):
        candidates = sorted(candidates)
        selected = []
        for dist_to_query, c in candidates:
            if len(selected) >= m:
                break
            vec_c = self.vectors[c]
            competitive = True
            for _, s in selected:
                if self._distance(vec_c, self.vectors[s]) < dist_to_query:
                    competitive = False
                    break
            if competitive:
                selected.append((dist_to_query, c))
        return [n for _, n in selected]

    def insert(self, vector):
        """Add a vector to the index (assigns a random level and links it to its nearest neighbours). Returns the new node id (ids follow insertion order)."""
        vector = np.asarray(vector, dtype=np.float32)
        node_id = len(self.vectors)
        node_level = self._random_level()

        self.vectors.append(vector)
        self.neighbors.append([set() for _ in range(node_level + 1)])
        self.max_level_of.append(node_level)

        if self.entry_point == -1:
            self.entry_point = node_id
            self.max_level = node_level
            return node_id

        ep = [self.entry_point]
        for layer in range(self.max_level, node_level, -1):
            result = self._search_layer(vector, ep, ef=1, layer=layer)
            ep = [min(result, key=lambda x: x[0])[1]]

        for layer in range(min(node_level, self.max_level), -1, -1):
            candidates = self._search_layer(vector, ep, ef=self.ef_construction, layer=layer)
            m = self.M0 if layer == 0 else self.M
            chosen = self._select_neighbors_heuristic(vector, candidates, m)
            self.neighbors[node_id][layer] = set(chosen)
            for neighbor in chosen:
                self.neighbors[neighbor][layer].add(node_id)
                self._prune(neighbor, layer)
            ep = [n for _, n in candidates]
        
        if node_level > self.max_level:
            self.max_level = node_level
            self.entry_point = node_id

        return node_id

    def _prune(self, node, layer):
        m = self.M0 if layer == 0 else self.M
        neighs = self.neighbors[node][layer]
        if len(neighs) <= m:
            return
        vec = self.vectors[node]
        scored = [(self._distance(vec, self.vectors[n]), n) for n in neighs]
        kept = self._select_neighbors_heuristic(vec, scored, m)
        self.neighbors[node][layer] = set(kept)

    def search(self, query, k=5, ef=None):
        """Return up to k nearest neighbours of `query` as a list of (node_id, distance), closest first. Empty list if the index is empty."""
        if self.entry_point == -1:
            return []
        query = np.asarray(query, dtype=np.float32)
        ef = ef or max(k, self.ef_construction)
        ep = [self.entry_point]
        for layer in range(self.max_level, 0, -1):
            result = self._search_layer(query, ep, ef=1, layer=layer)
            ep = [min(result, key=lambda x: x[0])[1]]
        candidates = self._search_layer(query, ep, ef=ef, layer=0)
        candidates.sort(key=lambda x: x[0])
        return [(n, d) for d, n in candidates[:k]]


    def _rng_state_to_json(self, state):
        """random.Random.getstate() returns (version, internal_state_tuple,
        gauss_next); JSON can't serialize tuples, so convert the middle
        element to a list."""
        version, internal_state, gauss_next = state
        return [version, list(internal_state), gauss_next]


    def to_dict(self) -> dict:
        """Capture full state (data + RNG) needed to exactly reconstruct
        this index, including behavior of any future inserts after load."""
        return {
            "dim": self.dim,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "metric": self.metric,
            "seed": self.seed,
            "vectors": [v.tolist() for v in self.vectors],
            "neighbors": [[list(layer_set) for layer_set in node_layers] for node_layers in self.neighbors],
            "max_level_of": self.max_level_of,
            "entry_point": self.entry_point,
            "max_level": self.max_level,
            "rng_state": self._rng_state_to_json(self._rng.getstate()),
        }
