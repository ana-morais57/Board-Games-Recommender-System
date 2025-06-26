import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
import hdbscan


def load_features(path="boardgamesdf.csv"):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ['bggid', 'name', 'avgrating', 'Cluster']]
    X = df[feature_cols]
    return df, X


def cluster_agglomerative(X, n_clusters=4):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    return labels


def cluster_hdbscan(X, min_cluster_size=15):
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(X)
    return labels


def compute_metrics(X, labels):
    if len(np.unique(labels)) <= 1:
        return None, None, None
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    return sil, db, ch


def main():
    df, X = load_features()
    # existing KMeans clusters
    if 'Cluster' in df.columns:
        km_labels = df['Cluster'].values
        sil_km, db_km, ch_km = compute_metrics(X, km_labels)
        print(f"KMeans -> silhouette: {sil_km:.4f}, DB: {db_km:.4f}, CH: {ch_km:.4f}")
    else:
        km_labels = KMeans(n_clusters=4, random_state=42).fit_predict(X)
        df['Cluster'] = km_labels
        sil_km, db_km, ch_km = compute_metrics(X, km_labels)
        print(f"KMeans -> silhouette: {sil_km:.4f}, DB: {db_km:.4f}, CH: {ch_km:.4f}")

    agg_labels = cluster_agglomerative(X)
    df['Cluster_Agglomerative'] = agg_labels
    sil_agg, db_agg, ch_agg = compute_metrics(X, agg_labels)
    print(f"Agglomerative -> silhouette: {sil_agg:.4f}, DB: {db_agg:.4f}, CH: {ch_agg:.4f}")

    hdb_labels = cluster_hdbscan(X)
    df['Cluster_HDBSCAN'] = hdb_labels
    sil_hdb, db_hdb, ch_hdb = compute_metrics(X, hdb_labels)
    print(f"HDBSCAN -> silhouette: {sil_hdb:.4f}, DB: {db_hdb:.4f}, CH: {ch_hdb:.4f}")

    df.to_csv('boardgamesdf_with_clusters.csv', index=False)


if __name__ == "__main__":
    main()
