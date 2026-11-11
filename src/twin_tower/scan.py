"""Detect duplicated and redundant layers by cross-layer similarity.

Point TWIN-TOWER at a stack of same-shape weight matrices. It
flattens every layer, measures absolute cosine similarity between
all pairs, clusters near-duplicates into towers, and reports how
much of the stack is redundant. Pure NumPy.
"""

import numpy as np


def _checked(weights):
    if not isinstance(weights, (list, tuple)) or len(weights) == 0:
        raise ValueError("weights must be a non-empty list")
    arrs = [np.asarray(w, dtype=float) for w in weights]
    for a in arrs:
        if a.ndim != 2:
            raise ValueError("each layer must be a 2-D matrix")
    shape = arrs[0].shape
    for a in arrs:
        if a.shape != shape:
            raise ValueError("all layers must share one shape")
    return arrs


def similarity_matrix(weights):
    """Absolute cosine similarity between flattened layers."""
    arrs = _checked(weights)
    flat = np.stack([a.reshape(-1) for a in arrs])
    norms = np.clip(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12, None)
    unit = flat / norms
    return np.abs(unit @ unit.T)


def duplication_report(weights, cos_tol=0.9):
    """Cluster layers into towers of near-duplicates."""
    sim = similarity_matrix(weights)
    n = sim.shape[0]
    clusters = []
    assigned = [False] * n
    for i in range(n):
        if assigned[i]:
            continue
        assigned[i] = True
        cluster = [i]
        for j in range(i + 1, n):
            if not assigned[j] and sim[i, j] >= cos_tol:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)
    duplicate_pairs = int(sum(len(c) - 1 for c in clusters))
    most_similar = None
    if n > 1:
        iu = np.triu_indices(n, k=1)
        pair_scores = sim[iu]
        k = int(np.argmax(pair_scores))
        most_similar = [int(iu[0][k]), int(iu[1][k]), float(pair_scores[k])]
    return {
        "num_layers": n,
        "clusters": clusters,
        "num_towers": len(clusters),
        "duplicate_layers": duplicate_pairs,
        "redundancy_rate": duplicate_pairs / n,
        "most_similar_pair": most_similar,
    }