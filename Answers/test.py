import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('/Users/vivekmalipatel/Downloads/Assignmnts/IntrotoMLAssignment/Question/TwoFeatures.csv')
x = data[['x1', 'x2']].values

# Rescale data
x_rescaled = x

# Implement K-Means with Manhattan distance
def kmeans_manhattan(x, k, max_iter=100):
    centroids = x[np.random.choice(x.shape[0], size=k, replace=False)]
    for i in range(max_iter):
        distances = cdist(x, centroids, 'cityblock')
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([x[labels == j].mean(axis=0) for j in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

# Discover optimal number of clusters using Elbow method
twcss = []
elbow_values = []
for k in range(1, 9):
    centroids, labels = kmeans_manhattan(x_rescaled, k)
    twcss.append(((x_rescaled - centroids[labels])**2).sum())
    elbow_values.append(twcss[0] / twcss[-1])

# Plot Elbow values
plt.plot(range(1, 9), elbow_values, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Elbow value')
plt.show()

# Find optimal number of clusters
optimal_k = elbow_values.index(max(elbow_values)) - 1
print('Optimal number of clusters:', optimal_k)

