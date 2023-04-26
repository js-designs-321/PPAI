import numpy as np

# X = np.random.rand(15, 4) * 100
X = np.random.randint(low=0, high=10, size=(15, 4))

# Step 1: Standardize the Data
X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Step 2: Compute the Covariance Matrix
cov_mat = np.cov(X_std.T)

# Step 3: Compute the Eigenvectors and Eigenvalues of the Covariance Matrix
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# Step 4: Select the Principal Components
k = 2 # number of principal components to retain
top_k_eig_vecs = eig_vecs[:, :k]

# Step 5: Project the Data onto the Principal Components
X_pca = np.dot(X_std, top_k_eig_vecs)

# Print some information about the PCA results
print("Original Dataset:")
print(X)
print("\nStandardized Dataset:")
print(X_std)
print("\nCovariance Matrix:")
print(cov_mat)
print("\nEigenvalues:")
print(eig_vals)
print("\nEigenvectors:")
print(eig_vecs)
print("\nTop", k, "Eigenvectors:")
print(top_k_eig_vecs)
print("\nPCA-Transformed Data (", k, "dimensions):")
print(X_pca)
