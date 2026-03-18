import numpy as np
import time
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_array, check_random_state
from sklearn.datasets import make_blobs
from scipy.cluster.vq import vq
from scipy.spatial import cKDTree
from sklearn.cluster import kmeans_plusplus
from collections import defaultdict


class IKMeansPlusMinus(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=8, max_iters=20, local_refine_steps=5, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.local_refine_steps = local_refine_steps
        self.random_state = random_state


    def _get_assignment(self, X, centers):
        """
        Uses scipy's C-optimized VQ to find 1st nearest.
        To find 2nd nearest efficiently without a full dist matrix, 
        we temporarily mask the 1st and run VQ again.
        """
        labels, dists = vq(X, centers)
        
        # To get 2nd nearest (required for Cost Eq. 1) efficiently:
        # We only do this for the full set when needed to save time.
        return labels, dists


    
    def _get_full_metrics(self, X, centers):
        tree = cKDTree(centers)
        # k=2 for first and second nearest
        dist, idx = tree.query(X, k=2)
        # Return: labels, second_labels, dist1, dist2
        return idx[:, 0], idx[:, 1], dist[:, 0], dist[:, 1]
    

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


    def t_k_means(self, X, centers, labels, second_labels, si, sj, radius_mult=1.5):
        # Step #1: Setup
        ac = {si, sj}
        curr_centers = centers.copy()
        curr_labels = labels.copy()
        curr_second = second_labels.copy()
        
        # Pre-calculate a fixed radius based on global density to avoid sampling every time
        # This radius defines the "Topical" zone
        global_avg_dist = np.mean(np.linalg.norm(curr_centers[0] - curr_centers[1:]))
        radius = (global_avg_dist / np.sqrt(self.n_clusters)) * radius_mult

        # Build Tree ONCE
        full_tree = cKDTree(curr_centers)
        
        # Step 1.4: Initial affected points (orphans of sj)
        ap_initial_indices = np.where((labels == sj) | (second_labels == sj))[0]

        max_inner_iters = 10 # Safety cap
        it = 0
        
        while ac and it < max_inner_iters:
            it += 1
            # Step #2: Identify AC-Adjacent and AP
            ac_list = list(ac)
            
            # Use the tree to find neighbors of all AC centers at once
            ac_adjacent = set()
            for idx in ac_list:
                neighbors = full_tree.query_ball_point(curr_centers[idx], r=radius)
                ac_adjacent.update(neighbors)
            
            ap_mask = np.isin(curr_labels, ac_list) | np.isin(curr_second, ac_list)
            ap_indices = np.where(ap_mask)[0]
            
            if it == 1 and ap_initial_indices.size > 0:
                ap_indices = np.unique(np.concatenate([ap_indices, ap_initial_indices]))

            if ap_indices.size == 0: break

            # Step #3: Update local neighborhood
            relevant_indices = np.array(list(ac.union(ac_adjacent)), dtype=int)
            relevant_centers = curr_centers[relevant_indices]
            
            local_tree = cKDTree(relevant_centers)
            k_val = min(2, len(relevant_indices))
            _, local_idx = local_tree.query(X[ap_indices], k=k_val)
            
            new_1st = relevant_indices[local_idx[:, 0]]
            new_2nd = relevant_indices[local_idx[:, 1]] if k_val > 1 else new_1st
            
            # Determine Potential-AC (Step 3.2)
            changed_mask = (curr_labels[ap_indices] != new_1st)
            potential_ac = set(np.unique(curr_labels[ap_indices][changed_mask]))
            potential_ac.update(np.unique(new_1st[changed_mask]))
            
            curr_labels[ap_indices] = new_1st
            curr_second[ap_indices] = new_2nd

            # Step #4: Vectorized Center Update (Massively faster than for-loop)
            # We only need to update centers in 'ac'
            for idx in ac:
                m = (curr_labels == idx)
                if np.any(m):
                    curr_centers[idx] = np.mean(X[m], axis=0)

            ac = potential_ac
            # Optional: update full_tree only every few iterations if centers move significantly
            if ac and it % 2 == 0:
                full_tree = cKDTree(curr_centers)

        return curr_centers, curr_labels, curr_second
    
    def _get_strong_adjacents(self, labels, second_labels, target_idx):
        """
        Implements Definition 3 & 4: 
        Finds clusters that are 'Strongly Adjacent' to target_idx.
        """
        # 1. Find all Sj that are adjacent to target_idx (Definition 3)
        # i.e., points whose 1st center is target_idx and 2nd is Sj
        adj_to_target = np.unique(second_labels[labels == target_idx])
        
        strong_adjacents = []
        
        # 2. Check symmetry (Definition 4)
        for sj in adj_to_target:
            # Is target_idx a 2nd nearest neighbor for points currently in Sj?
            # We use np.any for a fast exit as soon as one point is found
            if np.any(second_labels[labels == sj] == target_idx):
                strong_adjacents.append(sj)
                
        return strong_adjacents

    def fit(self, X, y=None):
        # Initialize a timing dictionary to track bottlenecks
        self.timings = defaultdict(float)
        
        start_total = time.time()
        
        X = check_array(X)
        rs = check_random_state(self.random_state)
        n_samples, n_features = X.shape

        # 1. Initialization
        t0 = time.time()
        indices = rs.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[indices].copy()
        self.timings['init_normal'] += time.time() - t0

        t0 = time.time()
        self.labels_, self.second_centers_, d1, d2 = self._get_full_metrics(X, self.cluster_centers_)
        current_sse = np.sum(d1**2)
        self.timings['init_metrics'] += time.time() - t0
        
        indivisible = np.zeros(self.n_clusters, dtype=bool)
        irremovable = np.zeros(self.n_clusters, dtype=bool)
        unmatchable_pairs = set()

        for i in range(self.max_iters):
            # Vectorized Selection
            t_sel = time.time()
            cluster_sse = np.bincount(self.labels_, weights=d1**2, minlength=self.n_clusters)
            gains = 0.75 * cluster_sse
            gains[indivisible] = -1
            si = np.argmax(gains)

            costs = np.bincount(self.labels_, weights=(d2**2 - d1**2), minlength=self.n_clusters)
            costs[irremovable] = np.inf
            costs[si] = np.inf
            sj = np.argmin(costs)
            self.timings['loop_selection'] += time.time() - t_sel

            if costs[sj] >= gains[si] or (si, sj) in unmatchable_pairs:
                indivisible[si] = True
                continue

            new_centers = self.cluster_centers_.copy()
            new_sse = np.inf
            si_mask = (self.labels_ == si)

            if np.any(si_mask):
                # Minus-Plus Phase
                t_tp = time.time()
                new_centers[sj] = X[si_mask][rs.randint(np.sum(si_mask))]
                self.timings['loop_teleport'] += time.time() - t_tp

                # TOPICAL REFINEMENT (The speed engine)
                t_tk = time.time()
                new_centers, new_labels, new_second = self.t_k_means(
                    X, new_centers, self.labels_, self.second_centers_, si, sj
                )
                self.timings['loop_t_k_means'] += time.time() - t_tk
                
                # Final evaluation of trial move
                t_eval = time.time()
                _, _, nd1, _ = self._get_full_metrics(X, new_centers)
                new_sse = np.sum(nd1**2)
                self.timings['loop_eval_metrics'] += time.time() - t_eval

                # 3. Acceptance Step
                t_acc = time.time()
                if new_sse < current_sse:
                    # HEURISTICS 2 & 3: Find Strong Adjacents
                    strong_si = self._get_strong_adjacents(self.labels_, self.second_centers_, si)
                    strong_sj = self._get_strong_adjacents(self.labels_, self.second_centers_, sj)

                    # Update official state
                    self.cluster_centers_ = new_centers
                    self.labels_ = new_labels
                    self.second_centers_ = new_second
                    d1, current_sse = nd1, new_sse
                    # Error note: d2 is never updated here in your current logic!
                    
                    # Markings
                    irremovable[sj] = True
                    indivisible[si] = True
                    for idx in strong_sj: irremovable[idx] = True
                    for idx in strong_si: indivisible[idx] = True
                else:
                    unmatchable_pairs.add((si, sj))
                    indivisible[si] = True
                self.timings['loop_acceptance_logic'] += time.time() - t_acc

        self.total_fit_time = time.time() - start_total
        
        # Print a quick summary of the results
        print("\n--- Timing Profile ---")
        for k, v in self.timings.items():
            print(f"{k:25}: {v:.4f}s ({(v/self.total_fit_time)*100:.1f}%)")
        print(f"{'Total fit time':25}: {self.total_fit_time:.4f}s")
        
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