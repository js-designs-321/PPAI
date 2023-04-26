import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Create a random dataset
np.random.seed(42)
data = np.random.rand(100, 2)

# Create an instance of the AgglomerativeClustering class with the desired parameters
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
# ward: Minimizes the variance of the clusters being merged.
# complete: Maximizes the distance between clusters.
# average: Uses the average distance between clusters.

# Fit the data to the clustering model
agg.fit(data)

# Plot the data points with different colors for each cluster
plt.scatter(data[:, 0], data[:, 1], c=agg.labels_)
plt.show()
