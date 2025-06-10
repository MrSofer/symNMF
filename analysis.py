# -*- coding: utf-8 -*-
"""
Analysis file to compare SymNMF and K-means clustering.

This file applies both algorithms to a given dataset and reports
the silhouette scores for comparison.
"""

import sys
import numpy as np
from sklearn.metrics import silhouette_score
import kmeans
import symnmfmodule  # Import the C extension module

# Constants for convergence
EPSILON = 1e-4
MAX_ITER = 300

np.random.seed(1234)

def parse_arguments():
    """
    Parses command line arguments for analysis.

    Expected arguments:
    1. k (int): Number of required clusters.
    2. file_name (str): Path to the input data file (.txt).

    Returns:
        tuple: (k, file_name)
    """
    # Assuming arguments are always provided and valid as per instructions
    if len(sys.argv) != 3:
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        k = int(sys.argv[1])
        file_name = sys.argv[2]
    except ValueError:
        print("An Error Has Occurred")
        sys.exit(1)

    return k, file_name


def load_data(file_name):
    """
    Loads data points from the specified file.

    Args:
        file_name (str): Path to the input data file (.txt).

    Returns:
        np.ndarray: A numpy array of data points.
    """
    try:
        # Data points are expected to be comma-separated floats
        data = np.loadtxt(file_name, delimiter=',')
        return data
    except Exception:
        # Handle potential file reading errors
        print("An Error Has Occurred")
        sys.exit(1)


def assign_clusters_symnmf(H):
    """
    Assigns clusters based on the H matrix from symNMF.

    Each data point is assigned to the cluster corresponding to the
    column with the highest value in its row of H.

    Args:
        H (np.ndarray): The optimized H matrix (n x k).

    Returns:
        np.ndarray: A numpy array of cluster assignments (n,).
    """
    # Find the index of the maximum value in each row
    cluster_assignments = np.argmax(H, axis=1)
    return cluster_assignments


def main():
    """
    Main function to perform analysis and compare SymNMF and K-means.
    """
    k, file_name = parse_arguments()
    data = load_data(file_name)
    if not (1 < k < len(data)):
        print("An Error Has Occurred")
        sys.exit(1)

    # --- Perform SymNMF Clustering ---
    try:
        W = np.array(symnmfmodule.norm(data.tolist()))

        n = data.shape[0]
        m = np.mean(W)
        upper_bound = 2 * np.sqrt(m / k)
        initial_H = np.random.uniform(0, upper_bound, size=(n, k))

        # Optimize H using the C extension
        final_H = np.array(symnmfmodule.symnmf(initial_H.tolist(), W.tolist()))

        # Assign clusters based on the final H
        symnmf_labels = assign_clusters_symnmf(final_H)

        # Calculate silhouette score for SymNMF
        # Need at least 2 clusters and more than k data points for silhouette score
        symnmf_silhouette = silhouette_score(data, symnmf_labels)

    except Exception as e:
        # Handle potential errors from C extension calls
        # print(f"Error during SymNMF: {e}") # For debugging
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        labels = kmeans.fit(data, k, MAX_ITER)
        kmeans_silhouette = silhouette_score(data, labels)

    except Exception as e:
        # Handle potential errors during K-means
        print("An Error Has Occurred")
        sys.exit(1)

    # --- Output Results ---
    # Format output to 4 decimal places
    print(f"nmf: {symnmf_silhouette:.4f}")
    print(f"kmeans: {kmeans_silhouette:.4f}")


if __name__ == "__main__":
    main()
