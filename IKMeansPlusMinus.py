import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_array, check_random_state
from sklearn.datasets import make_blobs


class IKMeansPlusMinus(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=8, max_iters=10, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        
    def _get_sse(self, X, labels, centers):
        sse = 0
        for i in range(self.n_clusters):
            points = X[labels == i]
            if len(points) > 0:
                sse += np.sum(np.square(points - centers[i]))
        return sse

    def _assign_clusters(self, X, centers):
        """Returns labels, 1st distances, and 2nd nearest center indices."""
        distances = euclidean_distances(X, centers)
        sorted_idx = np.argsort(distances, axis=1)
        labels = sorted_idx[:, 0]
        second_nearest = sorted_idx[:, 1]
        
        # Distances to 1st and 2nd nearest for Cost calculation
        dist_1st = distances[np.arange(len(X)), labels]
        dist_2nd = distances[np.arange(len(X)), second_nearest]
        
        return labels, dist_1st, dist_2nd, second_nearest
    
    def _get_affected_points(self, labels, second_centers, ci, cj):
        # Only points belonging to center i or j, 
        # OR points that had i or j as their backup (second nearest)
        mask = (labels == ci) | (labels == cj) | (second_centers == ci) | (second_centers == cj)
        return np.where(mask)[0]

    def _topical_kmeans(self, X, centers, affected_indices):
        """Step 3.2: Only update affected points and centers."""
        
        # Simple implementation of topical refinement
        # In a full 'topical' version, we only re-assign points in 'AP'
        # and update 'AC' centers until convergence.
        temp_centers = centers.copy()
        for _ in range(5): # Local refinement iterations
            distances = euclidean_distances(X, temp_centers)
            labels = np.argmin(distances, axis=1)
            
            for k in range(self.n_clusters):
                mask = (labels == k)
                if np.any(mask):
                    temp_centers[k] = X[mask].mean(axis=0)
        return temp_centers, labels

    def fit(self, X, y=None):
        X = check_array(X)
        rs = check_random_state(self.random_state)
        n_samples, n_features = X.shape

        # Instruction #1: Initial K-means (Standard implementation)
        initial_indices = rs.permutation(n_samples)[:self.n_clusters]
        self.cluster_centers_ = X[initial_indices].copy()
        
        # Standard assignment to start
        self.labels_, dist_1st, dist_2nd, second_centers = self._assign_clusters(X, self.cluster_centers_)
        current_sse = self._get_sse(X, self.labels_, self.cluster_centers_)

        # State tracking for heuristics
        indivisible = np.zeros(self.n_clusters, dtype=bool)
        irremovable = np.zeros(self.n_clusters, dtype=bool)
        unmatchable_pairs = set()

        # Main Iterative Process (Instruction #3 - #8)
        for iteration in range(self.max_iters):
            # Calculate Gain for each cluster (Eq 7)
            gains = []
            for i in range(self.n_clusters):
                cluster_mask = (self.labels_ == i)
                sse_i = np.sum(np.square(X[cluster_mask] - self.cluster_centers_[i]))
                # Using the heuristic Gain = 0.75 * SSE(Si) from paper logic
                gains.append(0.75 * sse_i if not indivisible[i] else -1)
            
            si = np.argmax(gains)
            if gains[si] <= 0: break

            # Calculate Cost for each cluster (Eq 1)
            costs = []
            for j in range(self.n_clusters):
                if j == si or irremovable[j]:
                    costs.append(np.inf)
                    continue
                
                mask_j = (self.labels_ == j)
                # Cost is increase in SSE when moving points to 2nd nearest center
                cost_j = np.sum(np.square(dist_2nd[mask_j]) - np.square(dist_1st[mask_j]))
                costs.append(cost_j)

            sj = np.argmin(costs)
            
            # Instruction #5.2 & #8 validation
            if costs[sj] >= gains[si] or (si, sj) in unmatchable_pairs:
                indivisible[si] = True
                continue

            # Instruction #7: Minus-Plus Phase
            new_centers = self.cluster_centers_.copy()
            # Split Si: Cj becomes a random point in Si
            si_points = X[self.labels_ == si]
            if len(si_points) > 0:
                new_centers[sj] = si_points[rs.randint(len(si_points))]
            
            # Step 3.2: Topical Re-clustering (Simplified for API)
            refined_centers, refined_labels = self._topical_kmeans(X, new_centers, None)
            new_sse = self._get_sse(X, refined_labels, refined_centers)

            # Instruction #8: Acceptance
            if new_sse < current_sse:
                self.cluster_centers_ = refined_centers
                self.labels_ = refined_labels
                current_sse = new_sse
                # Heuristic markings
                irremovable[si] = True
                irremovable[sj] = True
            else:
                unmatchable_pairs.add((si, sj))
                indivisible[si] = True

        return self

    def predict(self, X):
        X = check_array(X)
        distances = euclidean_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)
    
if __name__ == "__main__":
    X, _ = make_blobs(n_samples=1000, centers=5, random_state=42)
    model = IKMeansPlusMinus(n_clusters=5)
    model.fit(X)
    labels = model.predict(X)
    print(f"Final SSE: {model._get_sse(X, model.labels_, model.cluster_centers_)}")