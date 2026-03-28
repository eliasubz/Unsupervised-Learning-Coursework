"""
Clustering Experiment Runner
Reproduces Tables 2 & 3 from the IKM-+ paper.

Usage:
    python experiment_runner.py                  # run everything
    python experiment_runner.py --config cfg.json  # custom config
    python experiment_runner.py --reset            # clear checkpoint and restart

Checkpointing: results are saved to results/checkpoint.json after every run.
Resume by simply re-running the script — completed runs are skipped.
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import (
    load_iris as _sklearn_load_iris,
    fetch_openml,
)
from sklearn.utils import check_random_state

warnings.filterwarnings("ignore")

from IKMeansPlusMinus import IKMeansPlusMinus
HAS_IKM = True

# ===========================================================================
# CONFIG
# ===========================================================================

DEFAULT_CONFIG = {
    "output_dir": "results_w_init",
    "checkpoint_file": "results/checkpoint.json",
    "data_dir": "data",
    "n_runs_synthetic": 15,
    "n_runs_realworld": 5,
    "ikm_max_iters": 10,
    "synthetic_datasets": ["A1", "A2", "A3", "S1", "S2", "S3", "S4", "Birch1"],
    "realworld_datasets": ["Iris", "LR", "Musk", "Statlog", "HAR", "ISOLET"],
}


# ===========================================================================
# DATASET LOADERS
# ===========================================================================

def load_franti_txt(path):
    """
    Load a Fränti benchmark .txt file — whitespace-separated x y coordinates,
    one point per line, no header.
    """
    return np.loadtxt(path, dtype=np.float64)


def load_a_series(name, data_dir="data"):
    k_map = {"A1": 20, "A2": 35, "A3": 50}
    fname = os.path.join(data_dir, f"{name.lower()}.txt")
    X = load_franti_txt(fname)
    return X, k_map[name]


def load_s_series(name, data_dir="data"):
    k_map = {"S1": 15, "S2": 15, "S3": 15, "S4": 15}
    fname = os.path.join(data_dir, f"{name.lower()}.txt")
    X = load_franti_txt(fname)
    return X, k_map[name]


def load_birch1(data_dir="data"):
    fname = os.path.join(data_dir, "birch1.txt")
    X = load_franti_txt(fname)
    return X, 100


def load_iris_dataset():
    data = _sklearn_load_iris()
    return data.astype(np.float64), 3


def load_letter_recognition():
    print("  [loader] Fetching Letter Recognition from OpenML...")
    ds = fetch_openml(name="letter", version=1, as_frame=False, parser="auto")
    X = ds.data.astype(np.float64)
    return X, 26


def load_musk():
    print("  [loader] Fetching Musk from OpenML...")
    ds = fetch_openml(name="musk", version=1, as_frame=False, parser="auto")
    X = ds.data.astype(np.float64)
    return X, 2


def load_statlog():
    print("  [loader] Fetching Statlog Shuttle from OpenML...")
    ds = fetch_openml(name="shuttle", version=1, as_frame=False, parser="auto")
    X = ds.data.astype(np.float64)
    return X, 7


def load_har():
    print("  [loader] Fetching HAR from OpenML...")
    ds = fetch_openml(name="har", version=1, as_frame=False, parser="auto")
    X = ds.data.astype(np.float64)
    return X, 6


def load_isolet():
    print("  [loader] Fetching ISOLET from OpenML...")
    ds = fetch_openml(name="isolet", version=1, as_frame=False, parser="auto")
    X = ds.data.astype(np.float64)
    return X, 26


def make_synthetic_loaders(data_dir):
    return {
        "A1":     lambda: load_a_series("A1", data_dir),
        "A2":     lambda: load_a_series("A2", data_dir),
        "A3":     lambda: load_a_series("A3", data_dir),
        "S1":     lambda: load_s_series("S1", data_dir),
        "S2":     lambda: load_s_series("S2", data_dir),
        "S3":     lambda: load_s_series("S3", data_dir),
        "S4":     lambda: load_s_series("S4", data_dir),
        "Birch1": lambda: load_birch1(data_dir),
    }

REALWORLD_LOADERS = {
    "Iris":    load_iris_dataset,
    "LR":      load_letter_recognition,
    "Musk":    load_musk,
    "Statlog": load_statlog,
    "HAR":     load_har,
    "ISOLET":  load_isolet,
}


def load_dataset(name, data_dir="data"):
    synthetic_loaders = make_synthetic_loaders(data_dir)
    if name in synthetic_loaders:
        return synthetic_loaders[name]()
    elif name in REALWORLD_LOADERS:
        return REALWORLD_LOADERS[name]()
    else:
        raise ValueError(f"Unknown dataset: {name}")





# ===========================================================================
# METRICS
# ===========================================================================

def compute_ssedm(X, labels, centers):
    """Total SSEDM."""
    return float(np.sum(np.square(X - centers[labels])))


def compute_max_partial_ssedm(X, labels, centers, k):
    """Maximum per-cluster SSEDM."""
    partials = []
    for c in range(k):
        mask = labels == c
        if np.any(mask):
            partials.append(float(np.sum(np.square(X[mask] - centers[c]))))
    return max(partials) if partials else 0.0


# ===========================================================================
# ALGORITHM RUNNERS
# ===========================================================================

def run_km(X, k, seed):
    start = time.time()
    m = KMeans(n_clusters=k, init="random", n_init=1, random_state=seed).fit(X)
    t = time.time() - start
    return m.labels_, m.cluster_centers_, t


def run_kmpp(X, k, seed):
    start = time.time()
    m = KMeans(n_clusters=k, init="k-means++", n_init=1, random_state=seed).fit(X)
    t = time.time() - start
    return m.labels_, m.cluster_centers_, t


def run_mbkm(X, k, seed):
    start = time.time()
    m = MiniBatchKMeans(n_clusters=k, n_init=1, random_state=seed).fit(X)
    t = time.time() - start
    return m.labels_, m.cluster_centers_, t


def run_ikm(X, k, seed, max_iters):
    if not HAS_IKM:
        return None, None, None
    start = time.time()
    m = IKMeansPlusMinus(n_clusters=k, max_iters=max_iters, random_state=seed).fit(X)
    t = time.time() - start
    labels = m.predict(X)
    return labels, m.cluster_centers_, t


RUNNERS = {
    "KM":    run_km,
    "KM++":  run_kmpp,
    "MBK":   run_mbkm,
}


def get_metrics(X, labels, centers, k):
    return {
        "ssedm":       compute_ssedm(X, labels, centers),
        "max_partial": compute_max_partial_ssedm(X, labels, centers, k),
    }


# ===========================================================================
# CHECKPOINTING
# ===========================================================================

def load_checkpoint(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def checkpoint_key(dataset, alg, run):
    return f"{dataset}__{alg}__{run}"


# ===========================================================================
# CORE EXPERIMENT LOOP
# ===========================================================================

def run_experiment(config):
    os.makedirs(config["output_dir"], exist_ok=True)
    checkpoint = load_checkpoint(config["checkpoint_file"])

    synthetic_results = {}
    realworld_results = {}

    # -----------------------------------------------------------------------
    # SYNTHETIC
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("SYNTHETIC DATASETS")
    print("="*60)

    for ds_name in config["synthetic_datasets"]:
        print(f"\n--- {ds_name} ---")
        try:
            X_raw, k = load_dataset(ds_name, config.get("data_dir", "data"))
        except Exception as e:
            print(f"  [SKIP] Could not load {ds_name}: {e}")
            continue

        X = X_raw
        n_runs = config["n_runs_synthetic"]
        accum = {alg: {"ssedm": [], "max_partial": [], "time": []}
                 for alg in ["KM", "KM++", "MBK", "IKM-+"]}

        # Load any already-completed runs from checkpoint
        for alg in list(accum.keys()):
            for run in range(n_runs):
                key = checkpoint_key(ds_name, alg, run)
                if key in checkpoint:
                    r = checkpoint[key]
                    accum[alg]["ssedm"].append(r["ssedm"])
                    accum[alg]["max_partial"].append(r["max_partial"])
                    accum[alg]["time"].append(r["time"])

        # Run remaining
        for run in range(n_runs):
            needs_run = [alg for alg in accum
                         if checkpoint_key(ds_name, alg, run) not in checkpoint]
            if not needs_run:
                continue

            print(f"  Run {run+1:3d}/{n_runs}  algorithms: {needs_run}")

            for alg in needs_run:
                key = checkpoint_key(ds_name, alg, run)
                try:
                    if alg == "KM":
                        labels, centers, t = run_km(X, k, run)
                    elif alg == "KM++":
                        labels, centers, t = run_kmpp(X, k, run)
                    elif alg == "MBK":
                        labels, centers, t = run_mbkm(X, k, run)
                    elif alg == "IKM-+":
                        labels, centers, t = run_ikm(X, k, run, config["ikm_max_iters"])
                        if labels is None:
                            continue

                    m = get_metrics(X, labels, centers, k)
                    accum[alg]["ssedm"].append(m["ssedm"])
                    accum[alg]["max_partial"].append(m["max_partial"])
                    accum[alg]["time"].append(t)

                    checkpoint[key] = {"ssedm": m["ssedm"],
                                       "max_partial": m["max_partial"],
                                       "time": t}
                except Exception as e:
                    print(f"    [ERROR] {alg} run {run}: {e}")

            save_checkpoint(config["checkpoint_file"], checkpoint)

        # Aggregate
        row = {"dataset": ds_name, "k": k}
        km_time = np.mean(accum["KM"]["time"]) if accum["KM"]["time"] else np.nan
        for alg in ["KM", "KM++", "MBK", "IKM-+"]:
            if accum[alg]["ssedm"]:
                row[f"MaxPartial_{alg}"] = np.mean(accum[alg]["max_partial"])
                row[f"SSEDM_{alg}"]      = np.mean(accum[alg]["ssedm"])
                row[f"Time_{alg}"]       = np.mean(accum[alg]["time"])
            else:
                row[f"MaxPartial_{alg}"] = np.nan
                row[f"SSEDM_{alg}"]      = np.nan
                row[f"Time_{alg}"]       = np.nan

        ikm_time = row.get("Time_IKM-+", np.nan)
        row["tIKM/tKM"] = ikm_time / km_time if km_time and km_time > 0 else np.nan
        synthetic_results[ds_name] = row

    # -----------------------------------------------------------------------
    # REAL-WORLD
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("REAL-WORLD DATASETS")
    print("="*60)

    for ds_name in config["realworld_datasets"]:
        print(f"\n--- {ds_name} ---")
        try:
            X_raw, k = load_dataset(ds_name, config.get("data_dir", "data"))
        except Exception as e:
            print(f"  [SKIP] Could not load {ds_name}: {e}")
            continue

        X = X_raw
        n_runs = config["n_runs_realworld"]
        accum = {alg: {"ssedm": [], "max_partial": [], "time": []}
                 for alg in ["KM", "KM++", "MBK", "IKM-+"]}

        for alg in list(accum.keys()):
            for run in range(n_runs):
                key = checkpoint_key(ds_name, alg, run)
                if key in checkpoint:
                    r = checkpoint[key]
                    accum[alg]["ssedm"].append(r["ssedm"])
                    accum[alg]["max_partial"].append(r["max_partial"])
                    accum[alg]["time"].append(r["time"])

        for run in range(n_runs):
            needs_run = [alg for alg in accum
                         if checkpoint_key(ds_name, alg, run) not in checkpoint]
            if not needs_run:
                continue

            print(f"  Run {run+1:3d}/{n_runs}  algorithms: {needs_run}")

            for alg in needs_run:
                key = checkpoint_key(ds_name, alg, run)
                try:
                    if alg == "KM":
                        labels, centers, t = run_km(X, k, run)
                    elif alg == "KM++":
                        labels, centers, t = run_kmpp(X, k, run)
                    elif alg == "MBK":
                        labels, centers, t = run_mbkm(X, k, run)
                    elif alg == "IKM-+":
                        labels, centers, t = run_ikm(X, k, run, config["ikm_max_iters"])
                        if labels is None:
                            continue

                    m = get_metrics(X, labels, centers, k)
                    accum[alg]["ssedm"].append(m["ssedm"])
                    accum[alg]["max_partial"].append(m["max_partial"])
                    accum[alg]["time"].append(t)

                    checkpoint[key] = {"ssedm": m["ssedm"],
                                       "max_partial": m["max_partial"],
                                       "time": t}
                except Exception as e:
                    print(f"    [ERROR] {alg} run {run}: {e}")
            

            save_checkpoint(config["checkpoint_file"], checkpoint)
        # After the runs loop, print live averages before aggregating
        print(f"\n  Results for {ds_name} so far:")
        for alg in ["KM", "KM++", "MBK", "IKM-+"]:
            if accum[alg]["ssedm"]:
                avg_ssedm = np.mean(accum[alg]["ssedm"])
                avg_time  = np.mean(accum[alg]["time"])
                n         = len(accum[alg]["ssedm"])
                print(f"    {alg:<8} n={n:>3}  avg SSEDM={avg_ssedm:.2E}  avg time={avg_time:.4f}s")
            else:
                print(f"    {alg:<8} no results yet")

        row = {"dataset": ds_name, "k": k}
        km_time = np.mean(accum["KM"]["time"]) if accum["KM"]["time"] else np.nan
        for alg in ["KM", "KM++", "MBK", "IKM-+"]:
            if accum[alg]["ssedm"]:
                row[f"MaxPartial_{alg}"] = np.mean(accum[alg]["max_partial"])
                row[f"SSEDM_{alg}"]      = np.mean(accum[alg]["ssedm"])
                row[f"Time_{alg}"]       = np.mean(accum[alg]["time"])
            else:
                row[f"MaxPartial_{alg}"] = np.nan
                row[f"SSEDM_{alg}"]      = np.nan
                row[f"Time_{alg}"]       = np.nan

        ikm_time = row.get("Time_IKM-+", np.nan)
        row["tIKM/tKM"] = ikm_time / km_time if km_time and km_time > 0 else np.nan
        realworld_results[ds_name] = row

    return (pd.DataFrame(list(synthetic_results.values())),
            pd.DataFrame(list(realworld_results.values())))


# ===========================================================================
# TABLE FORMATTING
# ===========================================================================

def fmt(v):
    if pd.isna(v):
        return "N/A"
    if abs(v) >= 1e6 or (abs(v) < 0.01 and v != 0):
        return f"{v:.2E}"
    return f"{v:.3f}"


import numpy as np

def print_table(df, title):
    # Updated algorithms list to match your data
    algs = ["KM", "KM++", "MBK", "IKM-+"]
    
    # Helper for consistent Scientific Notation formatting
    def fmt_sci(val):
        try:
            if pd.isna(val) or val is None:
                return "   NaN    "
            return f"{float(val):.2E}"
        except:
            return "   NaN    "

    print(f"\n{'='*160}")
    print(f" {title}")
    print(f"{'='*160}")

    # Build Header
    # Dataset (12) + 4x MaxP (11) + 4x SSEDM (11) + 4x Time (11) + Ratio (11)
    header = f"{'Dataset':<12}"
    for a in algs: header += f" MaxP_{a:<6}"
    header += " |"
    for a in algs: header += f" SSEDM_{a:<5}"
    header += " |"
    for a in algs: header += f" T_{a:<8}"
    header += " | tIKM/tKM"
    
    print(header)
    print("-" * len(header))

    for _, row in df.iterrows():
        # Start with Dataset name
        line = f"{str(row['dataset']):<12}"
        
        # 1. Max Partial SSEDMs
        for alg in algs:
            val = row.get(f'MaxPartial_{alg}', np.nan)
            line += f" {fmt_sci(val):>10}"
        
        line += " |"
        
        # 2. Total SSEDMs
        for alg in algs:
            val = row.get(f'SSEDM_{alg}', np.nan)
            line += f" {fmt_sci(val):>10}"
            
        line += " |"
        
        # 3. Runtimes
        for alg in algs:
            val = row.get(f'Time_{alg}', np.nan)
            line += f" {fmt_sci(val):>10}"
            
        line += " |"
        
        # 4. Final Ratio
        ratio = row.get('tIKM/tKM', np.nan)
        line += f" {fmt_sci(ratio):>10}"
        
        print(line)

# Usage:
# print_table(real_world_df, "Real World Datasets Performance")


def save_tables(syn_df, rw_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    syn_df.to_csv(f"{output_dir}/table_synthetic.csv", index=False)
    rw_df.to_csv(f"{output_dir}/table_realworld.csv", index=False)
    print(f"\n[Saved] {output_dir}/table_synthetic.csv")
    print(f"[Saved] {output_dir}/table_realworld.csv")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=str, default=None,
                        help="Path to JSON config file")
    parser.add_argument("--reset",   action="store_true",
                        help="Clear checkpoint and restart from scratch")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()

    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))

    if args.reset and os.path.exists(config["checkpoint_file"]):
        os.remove(config["checkpoint_file"])
        print("[Reset] Checkpoint cleared.")

    syn_df, rw_df = run_experiment(config)

    print_table(syn_df, "Table 2 — Synthetic Datasets (averaged over runs)")
    print_table(rw_df,  "Table 3 — Real-World Datasets")

    save_tables(syn_df, rw_df, config["output_dir"])


if __name__ == "__main__":
    main()
