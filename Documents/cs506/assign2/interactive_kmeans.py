import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Define the KMeans class for interactive clustering
class KMeansInteractive:
    def __init__(self, n_clusters=3, max_iters=300, init_method='random', tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None
        self.init_method = init_method
        self.tol = tol  # Tolerance for convergence
        self.prev_centroids = None  # To store centroids from the previous step
    
    def initialize_centroids(self, X):
        if self.init_method == 'random':
            # Select random centroids from the data points
            indices = np.random.choice(len(X), self.n_clusters, replace=False)
            self.centroids = X[indices]
    
    def assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        self.labels = np.argmin(distances, axis=1)
    
    def update_centroids(self, X):
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            points_in_cluster = X[self.labels == i]
            if len(points_in_cluster) > 0:
                new_centroids[i] = points_in_cluster.mean(axis=0)
        return new_centroids
    
    def has_converged(self, new_centroids):
        # Check if the centroids have moved less than the tolerance (converged)
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
            
            # Check if the centroids have converged
            if self.has_converged(new_centroids):
                return True  # Converged
            self.prev_centroids = self.centroids
            self.centroids = new_centroids
        return False  # Not yet converged

# Global variables for interaction
kmeans = None
X = None
current_step = 0

# Update the plot after each step
def update_plot(X, kmeans):
    plt.clf()  # Clear the plot
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, cmap='viridis')
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='x', s=100)
    plt.title(f'KMeans Clustering Step {current_step}')
    plt.draw()

# Handle "Next Step" button click
def next_step(event):
    global current_step
    current_step += 1
    if current_step <= kmeans.max_iters:
        converged = kmeans.fit_step(X)
        update_plot(X, kmeans)
        if converged:
            print("Clustering has converged!")
            plt.title(f'KMeans Converged at Step {current_step}')
            plt.draw()  # Update the plot with the convergence message
            return
    else:
        print("Reached the maximum number of iterations. Clustering finished.")

# Set up the step-by-step visualization with a button
def stepwise_visualization(X, kmeans):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    ax.scatter(X[:, 0], X[:, 1], c='blue', marker='o')
    ax_button = plt.axes([0.8, 0.05, 0.1, 0.075])  # Button position
    btn_next = Button(ax_button, 'Next Step')

    btn_next.on_clicked(next_step)
    plt.show()

# Main function to run the interactive KMeans demo
def interactive_kmeans_demo():
    global kmeans, X
    # Generate some random data
    np.random.seed(42)
    X = np.random.rand(300, 2) * 10

    # Initialize KMeans object with random centroid initialization
    kmeans = KMeansInteractive(n_clusters=3, init_method='random')

    # Start stepwise visualization
    stepwise_visualization(X, kmeans)

if __name__ == "__main__":
    interactive_kmeans_demo()