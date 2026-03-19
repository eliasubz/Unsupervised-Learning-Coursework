import numpy as np
import time
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_array, check_random_state
from sklearn.datasets import make_blobs
from scipy.cluster.vq import vq
from scipy.spatial import cKDTree
from collections import defaultdict


class IKMeansPlusMinus(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        n_clusters=8,
        max_iters=50,
        local_refine_steps=3,
        random_state=None,
        printing=True,
    ):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.local_refine_steps = local_refine_steps
        self.random_state = random_state
        self.printing = printing

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
            mask = labels == k
            if np.any(mask):
                centers[k] = np.mean(X[mask], axis=0)
        return centers

    def t_k_means(self, X, centers, labels, second_labels, si, sj):
        curr_centers = centers.copy()
        curr_labels = labels.copy()
        curr_second = second_labels.copy()

        ac = {si, sj}

        ap_initial = np.where((labels == sj) | (second_labels == sj))[0]
        
        full_tree = cKDTree(centers[list(ac)])
        first_iter = True
        max_inner_iters = 15

        for _ in range(max_inner_iters):
            if not ac:
                break

            ac_list = list(ac)
            ac_array = np.array(ac_list)

            # Step 2: AC-Adjacent via second-nearest relationships
            first_mask = np.isin(curr_labels, ac_array)
            second_mask = np.isin(curr_second, ac_array)
            ac_adjacent = set(curr_second[first_mask].tolist())
            ac_adjacent.update(curr_labels[second_mask].tolist())
            ac_adjacent -= ac

            # Step 2.4: Collect affected points
            ap_mask = first_mask | second_mask
            ap_indices = np.where(ap_mask)[0]

            if first_iter and ap_initial.size > 0:
                ap_indices = np.unique(np.concatenate([ap_indices, ap_initial]))
                first_iter = False

            if ap_indices.size == 0:
                break

            # Step 3: Query ALL centers for affected points, not just AC ∪ AC-Adjacent
            # This ensures boundary points always find their true nearest center
            # k_val = min(2, len(curr_centers))
            # _, all_idx = full_tree.query(X[ap_indices], k=k_val)

            # new_1st = all_idx[:, 0]
            # new_2nd = all_idx[:, 1] if k_val > 1 else new_1st.copy()
            relevant = ac.union(ac_adjacent)
            relevant_indices = np.array(list(relevant), dtype=int)
            relevant_centers = curr_centers[relevant_indices]

            local_tree = cKDTree(relevant_centers)
            k_val = min(2, len(relevant_indices))
            _, local_idx = local_tree.query(X[ap_indices], k=k_val)

            new_1st = relevant_indices[local_idx[:, 0]]
            new_2nd = relevant_indices[local_idx[:, 1]] if k_val > 1 else new_1st.copy()

            # Step 3.2: Potential-AC from changed first assignments only
            changed_mask = curr_labels[ap_indices] != new_1st
            potential_ac = set(curr_labels[ap_indices][changed_mask].tolist())
            potential_ac.update(new_1st[changed_mask].tolist())

            # Apply updates
            curr_labels[ap_indices] = new_1st
            curr_second[ap_indices] = new_2nd
            # Step 4: actually update centers in AC
            for idx in ac_list:
                mask = curr_labels == idx
                if np.any(mask):
                    curr_centers[idx] = np.mean(X[mask], axis=0)

            # # Step 4: Update centers in AC
            # for idx in ac_list:
            #     mask = (curr_labels == idx)
            #     if np.any(mask):
            #         curr_centers[idx] = np.mean(X[mask], axis=0)

            # Rebuild full tree since AC centers moved
            if potential_ac:
                full_tree = cKDTree(curr_centers)

            ac = potential_ac

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
        self.timings["init_normal"] += time.time() - t0

        t0 = time.time()
        self.labels_, self.second_centers_, d1, d2 = self._get_full_metrics(
            X, self.cluster_centers_
        )
        current_sse = np.sum(d1**2)
        self.timings["init_metrics"] += time.time() - t0

        indivisible = np.zeros(self.n_clusters, dtype=bool)
        irremovable = np.zeros(self.n_clusters, dtype=bool)
        unmatchable_pairs = set()

        for i in range(self.max_iters):
            # Vectorized Selection
            t_sel = time.time()
            cluster_sse = np.bincount(
                self.labels_, weights=d1**2, minlength=self.n_clusters
            )
            gains = 0.75 * cluster_sse
            gains[indivisible] = -1
            si = np.argmax(gains)

            # Instruction 4: terminate if k/2 clusters have a larger gain than si
            # if np.sum(gains > gains[si]) >= self.n_clusters / 2:
            #     break

            costs = np.bincount(
                self.labels_, weights=(d2**2 - d1**2), minlength=self.n_clusters
            )
            costs[irremovable] = np.inf
            costs[si] = np.inf
            sj = np.argmin(costs)

            self.timings["loop_selection"] += time.time() - t_sel

            if costs[sj] >= gains[si] or (si, sj) in unmatchable_pairs:
                if np.sum(costs < costs[sj]) >= self.n_clusters / 2:
                    indivisible[si] = True
                    continue
                indivisible[si] = True
                continue

            new_centers = self.cluster_centers_.copy()
            new_sse = np.inf
            si_mask = self.labels_ == si

            if np.any(si_mask):
                # Minus-Plus Phase
                t_tp = time.time()
                new_centers[sj] = X[si_mask][rs.randint(np.sum(si_mask))]
                self.timings["loop_teleport"] += time.time() - t_tp

                # TOPICAL REFINEMENT
                t_tk = time.time()
                new_centers, new_labels, new_second = self.t_k_means(
                    X, new_centers, self.labels_, self.second_centers_, si, sj
                )
                self.timings["loop_t_k_means"] += time.time() - t_tk

                # Final evaluation of trial move
                t_eval = time.time()
                # new_sse = np.sum((X - new_centers[new_labels]) ** 2)
                _, _, nd1, _ = self._get_full_metrics(X, new_centers)
                new_sse = np.sum(nd1**2)

                self.timings["loop_eval_metrics"] += time.time() - t_eval

                # 3. Acceptance Step
                t_acc = time.time()
                if new_sse < current_sse:
                    # HEURISTICS 2 & 3: Find Strong Adjacents
                    strong_si = self._get_strong_adjacents(
                        self.labels_, self.second_centers_, si
                    )
                    strong_sj = self._get_strong_adjacents(
                        self.labels_, self.second_centers_, sj
                    )

                    # Update official state
                    self.cluster_centers_ = new_centers
                    self.labels_ = new_labels
                    self.second_centers_ = new_second
                    current_sse = new_sse

                    #
                    # _, _, d1, d2 = self._get_full_metrics(X, self.cluster_centers_)
                    d1, current_sse = nd1, new_sse

                    # Markings
                    # indivisible = np.zeros(self.n_clusters, dtype=bool)
                    # irremovable = np.zeros(self.n_clusters, dtype=bool)
                    irremovable[sj] = True
                    indivisible[si] = True
                    for idx in strong_sj:
                        irremovable[idx] = True
                    for idx in strong_si:
                        indivisible[idx] = True
                else:
                    unmatchable_pairs.add((si, sj))
                    indivisible[si] = True
                self.timings["loop_acceptance_logic"] += time.time() - t_acc

        self.total_fit_time = time.time() - start_total

        # Print a quick summary of the results
        if self.printing:
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
        ("K-Means (Random)", KMeans(n_clusters=15, init="random", n_init=1)),
        ("K-Means++", KMeans(n_clusters=15, init="k-means++", n_init=1)),
        (
            "I-K-Means-+",
            IKMeansPlusMinus(n_clusters=15, random_state=1, printing=False),
        ),
    ]

    for name, model in methods:
        start = time.time()
        model.fit(X)
        elapsed = time.time() - start

        # Consistent SSE calculation
        final_labels = model.predict(X)
        centers = (
            model.cluster_centers_
            if hasattr(model, "cluster_centers_")
            else model.cluster_centers_
        )
        sse = np.sum(
            [np.sum(np.square(X[final_labels == i] - centers[i])) for i in range(15)]
        )

        print(f"{name:15} | SSE: {sse:12.2f} | Time: {elapsed:.4f}s")

    from main import run_paper_reproduction
