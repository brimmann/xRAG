import logging
import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# Configure logging to show progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# UMAP parameters
N_NEIGHBORS = 15
MIN_DIST = 0.1
METRIC = 'cosine'

# HDBSCAN parameters
MIN_CLUSTER_SIZE = 3


def reduce_dimensionality(embeddings, random_state=42):
    """
    Reduces the dimensionality of embeddings from high-dim to 2D using UMAP.
    This step does NOT require any labels.
    """
    logging.info("Starting UMAP dimensionality reduction...")
    reducer = umap.UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric=METRIC,
        n_components=2,
        random_state=random_state,
        verbose=True
    )
    reduced_embeddings = reducer.fit_transform(embeddings)
    logging.info("UMAP reduction complete.")
    return reduced_embeddings

def discover_clusters(embeddings_2d):
    """
    Discovers clusters in the 2D embeddings using HDBSCAN.
    HDBSCAN does not require you to specify the number of clusters beforehand.
    """
    logging.info("Starting cluster discovery with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    # The output of this function is the "discovered labels"
    labels = clusterer.fit_predict(embeddings_2d)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logging.info(f"HDBSCAN found {n_clusters} clusters and {np.sum(labels == -1)} noise points.")
    return labels

def visualize_clusters(reduced_embeddings, labels, output_filename):
    """
    Visualizes the 2D embeddings, coloring points by their discovered cluster label.
    """
    logging.info("Generating visualization...")
    # Points labeled -1 are noise; we'll plot them in grey.
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # Use a color map with enough distinct colors
    cmap = plt.cm.get_cmap('Spectral', len(unique_labels))

    plt.figure(figsize=(14, 12))
    for k in unique_labels:
        class_member_mask = (labels == k)
        xy = reduced_embeddings[class_member_mask]

        if k == -1:
            # It's a noise point
            plt.scatter(xy[:, 0], xy[:, 1], c='lightgray', s=1, alpha=0.5, label='Noise')
        else:
            # It's a core cluster point
            plt.scatter(xy[:, 0], xy[:, 1], c=[cmap(k)], s=5, alpha=0.8, label=f'Cluster {k}')

    plt.title(f'UMAP Projection with {n_clusters} Discovered Clusters (HDBSCAN)', fontsize=16)
    plt.xlabel('UMAP Component 1', fontsize=12)
    plt.ylabel('UMAP Component 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Only add a legend if there are a reasonable number of clusters
    if len(unique_labels) < 20:
        plt.legend()
        
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    logging.info(f"Cluster visualization saved to {output_filename}")