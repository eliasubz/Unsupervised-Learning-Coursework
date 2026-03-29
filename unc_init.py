"""
UNC (Useful Nearest Centers) initialization — vectorized + profiled version.

Run directly to benchmark on S4 (or synthetic data if S4 not found):
    python unc_init.py
"""

import numpy as np
import time


# ---------------------------------------------------------------------------
# Vectorized core
# ---------------------------------------------------------------------------

def _find_useful_mask(P_C_dists, C_C_dists, chunk_size=2000):
    """
    Returns a boolean mask of shape (n_points, n_centers).

    Center j is useless for point i iff there exists some center x such that:
        max(P_C[i, x],  C_C[j, x])  <  P_C[i, j]
    i.e. x is closer to i than j AND x is closer to j than j is to i.

    Vectorised: build tensor (chunk, n_centers, n_centers):
        max_dist[i, j, x] = max(P_C[i, x], C_C[j, x])
    then useless[i, j] = any_x( max_dist[i, j, x] < P_C[i, j] )

    chunk_size controls peak RAM: (chunk, k, k) float64 array.
    For k=15, chunk=2000: ~3.6 MB.  For k=100, chunk=2000: ~160 MB.
    """
    n_points, n_centers = P_C_dists.shape
    useful_mask = np.ones((n_points, n_centers), dtype=bool)

    C_jx = C_C_dists[np.newaxis, :, :]   # (1, k, k) — broadcast over chunks

    for start in range(0, n_points, chunk_size):
        end   = min(start + chunk_size, n_points)
        P     = P_C_dists[start:end]           # (chunk, k)

        # max(P[i,x], C[j,x]) for all i,j,x — shape (chunk, k, k)
        max_dist = np.maximum(P[:, np.newaxis, :], C_jx)

        # useless[i,j] = any x where max_dist[i,j,x] < P[i,j]
        useless = np.any(max_dist < P[:, :, np.newaxis], axis=2)   # (chunk, k)
        useful_mask[start:end] = ~useless

    return useful_mask


def _compute_scores(P_C_dists, useful_mask, exclude_indices=None):
    """
    Vectorised UNC score:
        score[i] = (avg_useful_dist / max_useful_dist) * sum_log_useful_dist
    """
    eps = 1e-10
    safe_dists = np.maximum(P_C_dists, eps)

    # log of distances — zero for non-useful entries so sum is unaffected
    log_dists = np.where(useful_mask, np.log(safe_dists), 0.0)

    # mask non-useful with nan for avg/max aggregation
    masked = np.where(useful_mask, safe_dists, np.nan)

    avg_dis    = np.nanmean(masked, axis=1)
    max_dis    = np.nanmax(masked, axis=1)
    sum_ln_dis = np.sum(log_dists, axis=1)

    scores = (avg_dis / max_dis) * sum_ln_dis

    if exclude_indices:
        scores[list(exclude_indices)] = -np.inf

    return scores


# ---------------------------------------------------------------------------
# Public init function
# ---------------------------------------------------------------------------

def unc_init(points, k, return_indices=False, verbose=True, chunk_size=2000):
    """
    UNC initialization with per-step timing breakdown.

    Parameters
    ----------
    points         : (n, d) float array
    k              : number of centers to select
    return_indices : return point indices instead of coordinates
    verbose        : print timing table
    chunk_size     : rows processed at once in find_useful (tune for RAM)
    """
    n_points = points.shape[0]
    if k > n_points:
        raise ValueError(f"k ({k}) > n_points ({n_points})")

    t_dist = t_find = t_score = 0.0

    total_start = time.perf_counter()

    first_center_idx = int(np.argmin(points[:, 0]))
    selected_indices = [first_center_idx]
    centers = points[[first_center_idx]].copy()

    for _ in range(k - 1):
        t0 = time.perf_counter()
        P_C_dists = np.linalg.norm(
            points[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2
        )
        C_C_dists = np.linalg.norm(
            centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2
        )
        t_dist += time.perf_counter() - t0

        t0 = time.perf_counter()
        useful_mask = _find_useful_mask(P_C_dists, C_C_dists, chunk_size=chunk_size)
        t_find += time.perf_counter() - t0

        t0 = time.perf_counter()
        scores = _compute_scores(P_C_dists, useful_mask, exclude_indices=set(selected_indices))
        t_score += time.perf_counter() - t0

        next_idx = int(np.argmax(scores))
        selected_indices.append(next_idx)
        centers = points[selected_indices].copy()

    total = time.perf_counter() - total_start

    if verbose:
        w = 55
        other = total - t_dist - t_find - t_score
        print(f"\n{'='*w}")
        print(f"  unc_init  n={n_points}  k={k}  chunk={chunk_size}")
        print(f"{'='*w}")
        print(f"  Total time          : {total*1000:8.1f} ms")
        print(f"  Distance matrices   : {t_dist*1000:8.1f} ms")
        print(f"  find_useful (vec)   : {t_find*1000:8.1f} ms")
        print(f"  score compute (vec) : {t_score*1000:8.1f} ms")
        print(f"  Other (argmax etc.) : {other*1000:8.1f} ms")
        print(f"{'='*w}\n")

    if return_indices:
        return np.array(selected_indices)
    return centers


# ---------------------------------------------------------------------------
# Quick benchmark
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    s4_path = os.path.join(os.path.dirname(__file__), "data", "s4.txt")
    if os.path.exists(s4_path):
        X = np.loadtxt(s4_path, dtype=np.float64)
        k = 15
        label = "S4 (real)"
    else:
        print("[INFO] data/s4.txt not found — using synthetic 5000x2 data")
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5000, 2)) * 100
        k = 15
        label = "synthetic 5000x2"

    print(f"Dataset : {label}   shape={X.shape}   k={k}")
    centers = unc_init(X, k, verbose=True)
    print(f"Selected {len(centers)} centers.")
