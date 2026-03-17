import numpy as np
import time
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_array, check_random_state
from sklearn.datasets import make_blobs

class IKMeansPlusMinus(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=8, max_iters=20, local_refine_steps=5, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.local_refine_steps = local_refine_steps
        self.random_state = random_state

    def _full_assignment(self, X, centers):
        """Vectorized full assignment for initialization or final check."""
        dists = euclidean_distances(X, centers)
        sorted_indices = np.argsort(dists, axis=1)
        labels = sorted_indices[:, 0]
        second_nearest = sorted_indices[:, 1]
        
        dist_1st = dists[np.arange(len(X)), labels]
        dist_2nd = dists[np.arange(len(X)), second_nearest]
        return labels, second_nearest, dist_1st, dist_2nd

    def _update_centers(self, X, labels):
        centers = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            mask = (labels == k)
            if np.any(mask):
                centers[k] = np.mean(X[mask], axis=0)
        return centers

    def _topical_refinement(self, X, centers, labels, second_centers, si, sj):
        """
        Implementation of Section 3.2: Topical K-means.
        Only updates points whose 1st or 2nd nearest neighbor was si or sj.
        """
        curr_centers = centers.copy()
        curr_labels = labels.copy()
        
        # Affected Points (AP): Points whose neighborhood is compromised
        affected_mask = (labels == si) | (labels == sj) | (second_centers == si) | (second_centers == sj)
        affected_idx = np.where(affected_mask)[0]
        
        if len(affected_idx) == 0:
            return curr_centers, curr_labels

        for _ in range(self.local_refine_steps):
            # Only compute distances for the affected subset
            dists_subset = euclidean_distances(X[affected_idx], curr_centers)
            new_labels_subset = np.argmin(dists_subset, axis=1)
            
            # Check for convergence in the local area
            if np.array_equal(curr_labels[affected_idx], new_labels_subset):
                break
                
            curr_labels[affected_idx] = new_labels_subset
            
            # Update only centers that could have changed (AC + AC-Adjacent)
            # For simplicity, we update all centers, but only using current label assignments
            curr_centers = self._update_centers(X, curr_labels)
            
        return curr_centers, curr_labels

    def fit(self, X, y=None):
        X = check_array(X)
        rs = check_random_state(self.random_state)
        n_samples, n_features = X.shape

        # Initial Assignment (Standard K-means start)
        init_idx = rs.permutation(n_samples)[:self.n_clusters]
        self.cluster_centers_ = X[init_idx].copy()
        self.labels_, self.second_centers_, d1, d2 = self._full_assignment(X, self.cluster_centers_)
        
        current_sse = np.sum(np.square(d1))
        
        indivisible = np.zeros(self.n_clusters, dtype=bool)
        irremovable = np.zeros(self.n_clusters, dtype=bool)
        unmatchable_pairs = set()

        for it in range(self.max_iters):
            # Vectorized Gain calculation (Eq 7)
            cluster_sses = np.array([np.sum(np.square(X[self.labels_ == i] - self.cluster_centers_[i])) 
                                   for i in range(self.n_clusters)])
            gains = 0.75 * cluster_sses
            gains[indivisible] = -1
            
            si = np.argmax(gains)
            if gains[si] <= 0: break

            # Vectorized Cost calculation (Eq 1)
            # Cost = Sum(dist_to_2nd_nearest^2 - dist_to_1st_nearest^2) for points in Sj
            costs = np.zeros(self.n_clusters)
            for j in range(self.n_clusters):
                if j == si or irremovable[j]:
                    costs[j] = np.inf
                else:
                    mask_j = (self.labels_ == j)
                    costs[j] = np.sum(np.square(d2[mask_j]) - np.square(d1[mask_j]))

            sj = np.argmin(costs)

            # Heuristics & Constraints
            if costs[sj] >= gains[si] or (si, sj) in unmatchable_pairs:
                indivisible[si] = True
                continue

            # Plus-Minus: Teleport Cj into the densest part of Si
            new_centers = self.cluster_centers_.copy()
            si_points = X[self.labels_ == si]
            new_centers[sj] = si_points[rs.randint(len(si_points))]

            # Topical Refinement (Speed boost happens here)
            refined_centers, refined_labels = self._topical_refinement(
                X, new_centers, self.labels_, self.second_centers_, si, sj
            )
            
            # Verify improvement
            new_labels, new_2nd, new_d1, new_d2 = self._full_assignment(X, refined_centers)
            new_sse = np.sum(np.square(new_d1))

            if new_sse < current_sse:
                self.cluster_centers_, self.labels_, self.second_centers_ = refined_centers, new_labels, new_2nd
                d1, d2 = new_d1, new_d2
                current_sse = new_sse
                irremovable[si] = irremovable[sj] = True
            else:
                unmatchable_pairs.add((si, sj))
                indivisible[si] = True

        return self

    def predict(self, X):
        X = check_array(X)
        return np.argmin(euclidean_distances(X, self.cluster_centers_), axis=1)

# --- Comparison Script ---
if __name__ == "__main__":
    from sklearn.cluster import KMeans
    
    X, _ = make_blobs(n_samples=5000, centers=15, cluster_std=1.2, random_state=1)
    
    methods = [
        ("K-Means (Random)", KMeans(n_clusters=15, init='random', n_init=1)),
        ("K-Means++", KMeans(n_clusters=15, init='k-means++', n_init=1)),
        ("I-K-Means-+", IKMeansPlusMinus(n_clusters=15, random_state=1))
    ]
    
    for name, model in methods:
        start = time.time()
        model.fit(X)
        elapsed = time.time() - start
        
        # Consistent SSE calculation
        final_labels = model.predict(X)
        centers = model.cluster_centers_ if hasattr(model, 'cluster_centers_') else model.cluster_centers_
        sse = np.sum([np.sum(np.square(X[final_labels == i] - centers[i])) for i in range(15)])
        
        print(f"{name:15} | SSE: {sse:12.2f} | Time: {elapsed:.4f}s")