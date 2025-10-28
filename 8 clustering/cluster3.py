# cluster2.py
# Clustering experiments: KMeans, Hierarchical (Agglomerative), DBSCAN
# Saves plots and cluster summaries next to the data file.
# Run in your venv: python cluster2.py

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional prettier plots
try:
    import seaborn as sns
    sns.set(style="whitegrid")
    HAS_SEABORN = True
except:
    HAS_SEABORN = False

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.neighbors import NearestNeighbors

# -------------- CONFIG --------------
DATA_PATH = r"D:\DATA SCIENCE\ASSIGNMENTS\8 clustering\Clustering\EastWestAirlines.csv"
# If you already have the scaled csv from earlier: set SCALED_CSV to that path; script will use it.
SCALED_CSV = Path(DATA_PATH).parent / "eastwest_scaled_numeric.csv"

OUT_DIR = Path(DATA_PATH).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------- LOAD / PREPROCESS --------------
def load_prepared():
    # Always load from the scaled CSV (skip Excel)
    scaled_df = pd.read_csv(r"D:\DATA SCIENCE\ASSIGNMENTS\8 clustering\Clustering\eastwest_scaled_numeric.csv")
    
    # Use the same data for raw_num (no scaling reversal needed for clustering visuals)
    raw_num = scaled_df.copy()
    return raw_num, scaled_df


raw_num, X_scaled_df = load_prepared()
X = X_scaled_df.values
cols = X_scaled_df.columns.tolist()

# PCA for visualization coordinates
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
print("PCA explained variance (first 2):", pca.explained_variance_ratio_)

# -------------- VISUAL: PCA scatter (no clusters) --------------
plt.figure(figsize=(7,5))
plt.scatter(X_pca[:,0], X_pca[:,1], s=10, alpha=0.6)
plt.title("PCA (2D) - raw (no clusters)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(OUT_DIR / "pca_raw.png", dpi=150)
plt.close()

# -------------- K-MEANS: Elbow + Silhouette sweep --------------
Ks = list(range(2, 11))
inertia = []
sil_scores = []

for k in Ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels))

# Plot elbow & silhouette
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(Ks, inertia, marker='o')
plt.title("KMeans Elbow (Inertia)")
plt.xlabel("k"); plt.ylabel("Inertia")
plt.subplot(1,2,2)
plt.plot(Ks, sil_scores, marker='o')
plt.title("KMeans Silhouette vs k")
plt.xlabel("k"); plt.ylabel("Silhouette score")
plt.tight_layout()
plt.savefig(OUT_DIR / "kmeans_elbow_silhouette.png", dpi=150)
plt.close()

best_k = Ks[int(np.argmax(sil_scores))]
print("KMeans: best k by silhouette in range 2-10:", best_k, "silhouette:", max(sil_scores))

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=30)
km_labels = kmeans.fit_predict(X)

# PCA plot colored by KMeans
plt.figure(figsize=(7,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=km_labels, cmap='tab10', s=12, alpha=0.7)
plt.title(f"KMeans (k={best_k}) on PCA(2)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(OUT_DIR / f"kmeans_k{best_k}_pca.png", dpi=150)
plt.close()

# Save KMeans results
pd.DataFrame({'PC1': X_pca[:,0], 'PC2': X_pca[:,1], 'kmeans_label': km_labels}).to_csv(OUT_DIR / f"kmeans_k{best_k}_pca_labels.csv", index=False)

print("KMeans cluster counts:\n", pd.Series(km_labels).value_counts())
print("KMeans silhouette:", silhouette_score(X, km_labels))

# -------------- Hierarchical (Agglomerative) --------------
# Dendrogram (on subset for readability)
sample_n = min(300, X.shape[0])
sample_idx = np.random.choice(X.shape[0], size=sample_n, replace=False)
Z = linkage(X[sample_idx], method='ward')

plt.figure(figsize=(10, 4))
dendrogram(Z, truncate_mode='level', p=5)
plt.title("Dendrogram (ward) - truncated")
plt.tight_layout()
plt.savefig(OUT_DIR / "dendrogram_ward_truncated.png", dpi=150)
plt.close()

linkages = ['ward', 'complete', 'average']
agg_results = {}
for link in linkages:
    # ward linkage requires 'euclidean' and can't be used if metric != euclidean (we have default)
    ac = AgglomerativeClustering(n_clusters=best_k, linkage=link)
    labels_ac = ac.fit_predict(X)
    s = silhouette_score(X, labels_ac)
    agg_results[link] = (labels_ac, s)
    # plot
    plt.figure(figsize=(7,5))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_ac, cmap='tab10', s=12, alpha=0.7)
    plt.title(f"Agglomerative ({link}) k={best_k} silhouette={s:.3f}")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"agg_{link}_k{best_k}_pca.png", dpi=150)
    plt.close()
    print(f"Agglomerative ({link}) silhouette: {s:.4f} counts:\n", pd.Series(labels_ac).value_counts())

# Save last agg's labels
for link in linkages:
    labels_ac, s = agg_results[link]
    pd.DataFrame({'PC1': X_pca[:,0], 'PC2': X_pca[:,1], f'agg_{link}_label': labels_ac}).to_csv(OUT_DIR / f"agg_{link}_k{best_k}_labels.csv", index=False)

# -------------- DBSCAN: choose eps by k-NN knee + sweep --------------
# compute 5-NN distances sorted to inspect knee
nbrs = NearestNeighbors(n_neighbors=5).fit(X)
distances, _ = nbrs.kneighbors(X)
kth_dist = np.sort(distances[:,4])
plt.figure(figsize=(6,4))
plt.plot(kth_dist)
plt.title("Sorted 5-NN distances (knee indicates eps)")
plt.ylabel("5-NN distance")
plt.xlabel("sorted points")
plt.tight_layout()
plt.savefig(OUT_DIR / "knn_5dist_sorted.png", dpi=150)
plt.close()

# Sweep eps and min_samples
eps_list = [0.3, 0.5, 0.7, 0.9, 1.1]
min_samples_list = [4, 6, 8]
best_db = None
best_db_score = -1
for eps in eps_list:
    for ms in min_samples_list:
        db = DBSCAN(eps=eps, min_samples=ms)
        labels_db = db.fit_predict(X)
        n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
        if n_clusters <= 1:
            score = -1
        else:
            mask = labels_db != -1
            try:
                score = silhouette_score(X[mask], labels_db[mask])
            except:
                score = -1
        print(f"DBSCAN eps={eps}, min_samples={ms} -> clusters={n_clusters}, silhouette={score:.4f}, noise={(labels_db==-1).sum()}")
        if score > best_db_score:
            best_db_score = score
            best_db = (eps, ms, labels_db)

if best_db is not None:
    eps, ms, labels_db = best_db
    print("Best DBSCAN:", eps, ms, "silhouette:", best_db_score)
    # save and plot
    plt.figure(figsize=(7,5))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_db, cmap='tab10', s=12, alpha=0.7)
    plt.title(f"DBSCAN (eps={eps}, min_samples={ms})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"dbscan_eps{eps}_ms{ms}_pca.png", dpi=150)
    plt.close()
    pd.DataFrame({'PC1':X_pca[:,0],'PC2':X_pca[:,1],'dbscan_label':labels_db}).to_csv(OUT_DIR / f"dbscan_eps{eps}_ms{ms}_labels.csv", index=False)
else:
    print("DBSCAN did not find a stable multi-cluster solution in the tried grid.")

# -------------- CLUSTER INTERPRETATION: cluster means on original-scale features --------------
def summarize_clusters(original_df, labels, name):
    # original_df should be the raw (unscaled) numeric dataframe aligned to labels rows
    df_tmp = original_df.copy().reset_index(drop=True)
    df_tmp['cluster'] = labels
    summary = df_tmp.groupby('cluster').mean().T
    summary_file = OUT_DIR / f"{name}_cluster_feature_means.csv"
    summary.to_csv(summary_file)
    print(f"Saved cluster means for {name} to {summary_file}")
    return summary

# For kmeans:
summary_km = summarize_clusters(raw_num.reset_index(drop=True).loc[X_scaled_df.index], km_labels, f"kmeans_k{best_k}")
print("\nKMeans cluster means (truncated):")
print(summary_km.head())

# For hierarchical (ward)
ward_labels, ward_s = agg_results['ward']
summary_ward = summarize_clusters(raw_num.reset_index(drop=True).loc[X_scaled_df.index], ward_labels, "agg_ward")
print("\nAgglomerative(ward) cluster means (truncated):")
print(summary_ward.head())

if best_db is not None:
    summary_db = summarize_clusters(raw_num.reset_index(drop=True).loc[X_scaled_df.index], labels_db, f"dbscan_eps{eps}_ms{ms}")
    print("\nDBSCAN cluster means (truncated):")
    print(summary_db.head())

# -------------- FINAL SUMMARY PRINT --------------
print("\n--- FINAL SUMMARY ---")
print(f"KMeans k={best_k} silhouette={silhouette_score(X, km_labels):.4f}")
for link in linkages:
    print(f"Agglomerative ({link}) silhouette={agg_results[link][1]:.4f}")
if best_db is not None:
    print(f"Best DBSCAN eps={eps}, min_samples={ms} silhouette={best_db_score:.4f}")

print("All plots and CSV outputs saved to:", OUT_DIR)
