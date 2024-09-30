import numpy as np

class KMeansInteractive:
    def __init__(self, n_clusters=3, max_iters=300, init_method='random', tol=1e-4):
        """
        KMeans algorithm with interactive step-through capability.

        Parameters:
        - n_clusters: Number of clusters to form.
        - max_iters: Maximum number of iterations.
        - init_method: Method to initialize centroids ('random', 'farthest_first', 'kmeans++', 'manual').
        - tol: Tolerance for stopping criterion.
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None
        self.init_method = init_method
        self.tol = tol  # Tolerance for convergence
        self.prev_centroids = None  # To store centroids from the previous step
    
    def initialize_centroids(self, X):
        """Initialize centroids based on the selected method."""
        if self.init_method == 'random':
            indices = np.random.choice(len(X), self.n_clusters, replace=False)
            self.centroids = X[indices]
        elif self.init_method == 'farthest_first':
            self.centroids = self.farthest_first_initialization(X)
        elif self.init_method == 'kmeans++':
            self.centroids = self.kmeans_plus_plus_initialization(X)
        # Manual initialization should skip this step, as centroids will already be set externally
    
    def farthest_first_initialization(self, X):
        """Farthest-first initialization method for centroids."""
        centroids = [X[np.random.choice(len(X))]]
        for _ in range(1, self.n_clusters):
            distances = np.array([min(np.linalg.norm(x - c) for c in centroids) for x in X])
            next_centroid = X[np.argmax(distances)]
            centroids.append(next_centroid)
        return np.array(centroids)
    
    def kmeans_plus_plus_initialization(self, X):
        """KMeans++ initialization method for centroids."""
        centroids = [X[np.random.choice(len(X))]]
        for _ in range(1, self.n_clusters):
            distances = np.array([min(np.linalg.norm(x - c) ** 2 for c in centroids) for x in X])
            probabilities = distances / distances.sum()
            next_centroid = X[np.random.choice(len(X), p=probabilities)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def assign_clusters(self, X):
        """Assign clusters to data points based on the nearest centroid."""
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        self.labels = np.argmin(distances, axis=1)
    
    def update_centroids(self, X):
        """Update centroids based on the mean of the assigned points."""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            points_in_cluster = X[self.labels == i]
            if len(points_in_cluster) > 0:
                new_centroids[i] = points_in_cluster.mean(axis=0)
            else:
                # Optionally: Handle empty clusters (leave centroid unchanged or reinitialize)
                new_centroids[i] = self.centroids[i]  # Keep the previous centroid
        return new_centroids
    
    def has_converged(self, new_centroids):
        """Check if the algorithm has converged (centroids no longer change significantly)."""
        if self.prev_centroids is None:
            return False
        centroid_shift = np.linalg.norm(self.centroids - new_centroids, axis=None)
        return centroid_shift < self.tol
    
    def fit_step(self, X):
        """Perform one step of the KMeans algorithm."""
        if self.centroids is None:
            self.initialize_centroids(X)
        else:
            self.assign_clusters(X)
            new_centroids = self.update_centroids(X)
            
            if self.has_converged(new_centroids):
                return True  # Converged
            
            self.prev_centroids = self.centroids
            self.centroids = new_centroids
        return False  # Not yet converged
    
    def fit(self, X):
        """Run the KMeans algorithm until convergence or the maximum number of iterations."""
        for _ in range(self.max_iters):
            if self.fit_step(X):
                break