import numpy as np
import matplotlib.pyplot as plt


def k_means(data, k, max_iterations=100):
    centroids = data[np.random.choice(data.shape[0], k, replace=False), :]
    for i in range(max_iterations):
        distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        cluster_assignments = np.argmin(distances, axis=0)
        for j in range(k):
            centroids[j, :] = np.mean(data[cluster_assignments == j, :], axis=0)
    return centroids, cluster_assignments


data = np.random.rand(100, 2)
k = 3
centroids, cluster_assignments = k_means(data, k)

# plot results
plt.scatter(data[:, 0], data[:, 1], c=cluster_assignments)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='k')
plt.show()