import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from IKMeansPlusMinus import IKMeansPlusMinus


def run_paper_reproduction():
    n_samples = 50500
    k_values = [125, 250, 500, 1000, 2000]

    # Generate dataset with high complexity to prevent early convergence
    X, _ = make_blobs(
        n_samples=n_samples, centers=max(k_values), cluster_std=1.0, random_state=42
    )

    data = []

    for k in k_values:
        print(f"Processing k = {k}...")

        # --- Standard K-means++ ---
        # Note: We use n_init=1 to match the paper's comparison logic
        start = time.time()
        km_plus = KMeans(n_clusters=k, init="k-means++", n_init=1).fit(X)
        t_km_plus = time.time() - start
        sse_km_plus = km_plus.inertia_

        # --- I-k-means-+ ---
        start = time.time()
        ikm = IKMeansPlusMinus(n_clusters=k, max_iters=10, random_state=42).fit(X)
        t_ikm = time.time() - start
        # Consistently calculate SSE
        sse_ikm = np.sum(np.square(X - ikm.cluster_centers_[ikm.predict(X)]))

        data.append(
            {
                "k": k,
                "SSE_KM_Plus": sse_km_plus,
                "SSE_IKM": sse_ikm,
                "Time_KM_Plus": t_km_plus,
                "Time_IKM": t_ikm,
                "Speed_Ratio": t_km_plus / t_ikm,
            }
        )

    return pd.DataFrame(data)


results = run_paper_reproduction()
print(results[["k", "Time_KM_Plus", "Time_IKM", "Speed_Ratio"]])
