import numpy as np

def euclidean_distance(x, y):
    """Compute the Euclidean distance between two points."""
    return np.sqrt(np.sum((x - y) ** 2))

def agglomerative_clustering(X, n_clusters):
    """Perform agglomerative clustering on a dataset X.

    Args:
        X (ndarray): Input data of shape (n_samples, n_features).
        n_clusters (int): The number of clusters to form.

    Returns:
        ndarray: An array of length n_samples indicating the cluster
            number (0 to n_clusters-1) for each data point in X.
    """
    n_samples = X.shape[0]
    # Initialize each data point as a separate cluster
    clusters = np.arange(n_samples)
    # Compute the pairwise distances between all data points
    distances = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            distances[i,j] = euclidean_distance(X[i], X[j])
            distances[j,i] = distances[i,j]
    # Perform agglomerative clustering
    for k in range(n_samples, n_samples-n_clusters, -1):
        # Find the pair of closest clusters
        min_distance = np.inf
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                if clusters[i] != clusters[j] and distances[i,j] < min_distance:
                    min_distance = distances[i,j]
                    min_i, min_j = i, j
        # Merge the closest clusters
        clusters[clusters == clusters[min_j]] = clusters[min_i]
        # Update the distances to the new cluster
        for i in range(n_samples):
            if clusters[i] == clusters[min_i]:
                distances[i,min_i] = np.inf
                distances[min_i,i] = np.inf
            elif clusters[i] == clusters[min_j]:
                distances[i,min_i] = distances[i,min_j]
                distances[min_i,i] = distances[i,min_j]
                distances[i,min_j] = np.inf
                distances[min_j,i] = np.inf
        distances[min_i,min_i+1:] = distances[min_i+1:,min_i] = np.inf
    return clusters

# Example usage
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
n_clusters = 2

# Perform agglomerative clustering
labels = agglomerative_clustering(X, n_clusters)

# Print the resulting cluster labels
print(labels)

import matplotlib.pyplot as plt

# Plot the clustering result
plt.scatter(X[:,0], X[:,1], c=labels)
plt.title('Agglomerative Clustering Result')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
