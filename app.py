import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import hdbscan

st.title('Recommender System for Board Games')
st.write('This app provides board games recommendations based on clustering analysis')

# Load data
df = pd.read_csv("boardgamesdf.csv")

# Prepare feature matrix (exclude identifiers and rating)
feature_cols = [c for c in df.columns if c not in ['bggid', 'name', 'avgrating']]
X = df[feature_cols]

# Compute clustering labels
labels_kmeans = KMeans(n_clusters=4, random_state=42).fit_predict(X)
labels_agg = AgglomerativeClustering(n_clusters=4).fit_predict(X)
labels_hdb = hdbscan.HDBSCAN(min_cluster_size=15).fit_predict(X)

# Store labels in dataframe
df['Cluster_KMeans'] = labels_kmeans
df['Cluster_Agglomerative'] = labels_agg
df['Cluster_HDBSCAN'] = labels_hdb

# Calculate metrics

def calc_metrics(data, labels):
    if len(np.unique(labels)) <= 1:
        return -1, np.inf, -1
    sil = silhouette_score(data, labels)
    db = davies_bouldin_score(data, labels)
    ch = calinski_harabasz_score(data, labels)
    return sil, db, ch

metrics = {
    'KMeans': calc_metrics(X, labels_kmeans),
    'Agglomerative': calc_metrics(X, labels_agg),
    'HDBSCAN': calc_metrics(X, labels_hdb),
}

best_method = max(metrics, key=lambda m: metrics[m][0])

method = st.selectbox(
    'Select clustering method:',
    ['KMeans', 'Agglomerative', 'HDBSCAN'],
    index=['KMeans', 'Agglomerative', 'HDBSCAN'].index(best_method)
)

selected_name = st.selectbox('Select a board game:', df['name'].unique())

cluster_col = {
    'KMeans': 'Cluster_KMeans',
    'Agglomerative': 'Cluster_Agglomerative',
    'HDBSCAN': 'Cluster_HDBSCAN'
}[method]

if selected_name:
    selected_cluster = df.loc[df['name'] == selected_name, cluster_col].values[0]
    recommendations = df[(df[cluster_col] == selected_cluster) & (df['name'] != selected_name)]
    top_recommendations = recommendations.sort_values(by='avgrating', ascending=False).head(10)
    st.write(f"Top 10 games similar to '{selected_name}' using {method}:")
    st.dataframe(top_recommendations[['bggid', 'name', 'avgrating', 'gameweight']])

st.write('---')
st.write('Clustering metrics:')
for m, (sil, db, ch) in metrics.items():
    st.write(f"{m}: silhouette={sil:.3f}, DB={db:.3f}, CH={ch:.3f}")
