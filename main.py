import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
from IKMeansPlusMinus import IKMeansPlusMinus


def run_paper_reproduction():
    n_samples = 50500
    k_values = [125, 250, 500, 1000, 2000, 4000]

    X, _ = make_blobs(
        n_samples=n_samples, centers=max(k_values), cluster_std=1.0, random_state=42
    )

    data = []

    for k in k_values:
        print(f"Processing k = {k}...")

        # --- Standard K-Means (random init) ---
        start = time.time()
        km = KMeans(n_clusters=k, init="random", n_init=1).fit(X)
        t_km = time.time() - start
        sse_km = km.inertia_

        # --- Standard K-means++ ---
        start = time.time()
        km_plus = KMeans(n_clusters=k, init="k-means++", n_init=1).fit(X)
        t_km_plus = time.time() - start
        sse_km_plus = km_plus.inertia_

        # --- MiniBatchKMeans (additional sklearn algorithm) ---
        start = time.time()
        mbkm = MiniBatchKMeans(n_clusters=k, n_init=1, random_state=42).fit(X)
        t_mbkm = time.time() - start
        sse_mbkm = mbkm.inertia_

        # --- I-k-means-+ ---
        start = time.time()
        ikm = IKMeansPlusMinus(n_clusters=k, max_iters=10, random_state=42).fit(X)
        t_ikm = time.time() - start
        sse_ikm = np.sum(np.square(X - ikm.cluster_centers_[ikm.predict(X)]))

        data.append(
            {
                "k": k,
                "SSE_KM":       sse_km,
                "SSE_KM_Plus":  sse_km_plus,
                "SSE_MBK":      sse_mbkm,
                "SSE_IKM":      sse_ikm,
                "Time_KM":      t_km,
                "Time_KM_Plus": t_km_plus,
                "Time_MBK":     t_mbkm,
                "Time_IKM":     t_ikm,
            }
        )

    return pd.DataFrame(data)


def plot_results(df, figure=None):
    """
    Reproduces the runtime comparison chart from the paper and adds MiniBatchKMeans.

    Parameters
    ----------
    df      : DataFrame returned by run_paper_reproduction()
    figure  : optional matplotlib Figure to draw into.
              If None a new figure is created.
    """
    if figure is None:
        figure = plt.figure(figsize=(8, 5))

    ax = figure.add_subplot(111)
    k_vals = df["k"].values
    x_pos = np.arange(len(k_vals))  # equidistant integer positions

    ax.plot(x_pos, df["Avg_SSEDM_KM"],    marker="x", linestyle="-",  label="KM",             color="#1f77b4")
    ax.plot(x_pos, df["Avg_SSEDM_KM++"],  marker="^", linestyle="-",  label="KM++",            color="#d62728")
    ax.plot(x_pos, df["Avg_SSEDM_MBK"],   marker="s", linestyle="--", label="MiniBatchKMeans", color="#ff7f0e")
    ax.plot(x_pos, df["Avg_SSEDM_IKM-+"], marker="o", linestyle="-",  label="IKM-+",           color="#2ca02c")

    ax.set_title("Average SSEDM of Algorithms", fontsize=13)
    ax.set_xlabel("k", fontsize=11)
    ax.set_ylabel("Average SSEDM", fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"k={k}" for k in k_vals])  # original labels preserved
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    figure.tight_layout()
    return figure

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
from IKMeansPlusMinus import IKMeansPlusMinus


def run_refinement_experiment():
    n_samples = 50500
    k = 500

    refine_steps = list(range(1, 21)) + [25, 50, 100]

    X, _ = make_blobs(
        n_samples=n_samples,
        centers=k,
        cluster_std=1.0,
        random_state=42
    )

    results = []

    # -----------------------------
    # Baselines
    # -----------------------------
    print("Running baselines...")

    start = time.time()
    km = KMeans(n_clusters=k, init="random", n_init=1).fit(X)
    t_km = time.time() - start
    sse_km = km.inertia_

    start = time.time()
    km_plus = KMeans(n_clusters=k, init="k-means++", n_init=1).fit(X)
    t_km_plus = time.time() - start
    sse_km_plus = km_plus.inertia_

    start = time.time()
    mbkm = MiniBatchKMeans(n_clusters=k, n_init=1, random_state=42).fit(X)
    t_mbkm = time.time() - start
    sse_mbkm = mbkm.inertia_

    # Store baselines
    results.extend([
        {"method": "KM", "steps": 0, "time": t_km, "sse": sse_km},
        {"method": "KM++", "steps": 0, "time": t_km_plus, "sse": sse_km_plus},
        {"method": "MiniBatchKMeans", "steps": 0, "time": t_mbkm, "sse": sse_mbkm},
    ])

    # -----------------------------
    # IKM-+ refinement sweep
    # -----------------------------
    print("Running refinement sweep...")

    for steps in refine_steps:
        print(f"Refinement steps = {steps}")

        start = time.time()
        ikm = IKMeansPlusMinus(
            n_clusters=k,
            max_iters=10,
            local_refine_steps=steps,
            random_state=42
        ).fit(X)
        t_ikm = time.time() - start

        labels = ikm.predict(X)
        sse_ikm = np.sum((X - ikm.cluster_centers_[labels]) ** 2)

        results.append({
            "method": "IKM-+",
            "steps": steps,
            "time": t_ikm,
            "sse": sse_ikm
        })

    return pd.DataFrame(results)

def plot_accuracy_vs_time(df):
    fig, ax = plt.subplots(figsize=(8, 5))

    # -----------------------------
    # IKM curve
    # -----------------------------
    ikm_df = df[df["method"] == "IKM-+"].sort_values("steps")

    ax.plot(
        ikm_df["time"],
        ikm_df["sse"],
        marker="o",
        linestyle="-",
        color="#2ca02c",
        label="IKM-+"
    )

    # Annotate refinement steps
    for _, row in ikm_df.iterrows():
        ax.annotate(
            str(int(row["steps"])),
            (row["time"], row["sse"]),
            fontsize=7,
            alpha=0.7
        )

    # Baselines
    colors = {
        "KM": "#1f77b4",
        "KM++": "#d62728",
        "MiniBatchKMeans": "#ff7f0e"
    }

    baselines = df[df["method"] != "IKM-+"]

    for _, row in baselines.iterrows():
        ax.scatter(
            row["time"],
            row["sse"],
            color=colors[row["method"]],
            marker="x",
            s=80,
            label=row["method"]
        )

    # -----------------------------
    # Styling
    # -----------------------------
    ax.set_title("Accuracy vs Runtime (k = 500)", fontsize=13)
    ax.set_xlabel("Runtime (seconds)", fontsize=11)
    ax.set_ylabel("SSE (raw)", fontsize=11)

    ax.grid(True, linestyle="--", alpha=0.4)

    # Avoid duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    plt.tight_layout()
    return fig

   
if __name__ == "__main__":
    df = run_refinement_experiment()
    print(df)

    fig = plot_accuracy_vs_time(df)
    plt.show()

    # results = run_paper_reproduction()
    # print(results[["k", "Time_KM", "Time_KM_Plus", "Time_MBK", "Time_IKM"]])

    # fig = plot_results(results)
    # plt.show()