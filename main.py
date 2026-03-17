import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from IKMeansPlusMinus import IKMeansPlusMinus

# Assuming the IKMeansPlusMinus class from the previous turn is defined here

def run_experiment(n_samples=3000, n_clusters=15, centers=15):
    # Create a complex clustering problem
    X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=1.5, random_state=42)
    
    results = []
    
    # 1. Standard K-means (Random)
    start = time.time()
    km_random = KMeans(n_clusters=n_clusters, init='random', n_init=1).fit(X)
    results.append({
        'Method': 'K-means (Random)',
        'SSE': km_random.inertia_,
        'Time (s)': time.time() - start
    })

    # 2. K-means++
    start = time.time()
    km_plus = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1).fit(X)
    results.append({
        'Method': 'K-means++',
        'SSE': km_plus.inertia_,
        'Time (s)': time.time() - start
    })

    # 3. I-k-means-+
    start = time.time()
    ikm = IKMeansPlusMinus(n_clusters=n_clusters, max_iters=20, random_state=42).fit(X)
    # Re-calculate final SSE to ensure parity
    ikm_sse = ikm._get_sse(X, ikm.labels_, ikm.cluster_centers_)
    results.append({
        'Method': 'I-k-means-+',
        'SSE': ikm_sse,
        'Time (s)': time.time() - start
    })

    return pd.DataFrame(results), X, km_plus, ikm

# Execute
df_results, X, km_plus, ikm = run_experiment()

# Display Results Table
print("--- Experimental Results ---")
print(df_results.to_string(index=False))