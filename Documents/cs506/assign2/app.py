import io
from flask import Flask, render_template, jsonify, request, Response
import numpy as np
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from kmeans import KMeansInteractive  # Assuming KMeans logic is in kmeans.py

app = Flask(__name__)

# Global variables for the dataset and KMeans instance
X = np.random.rand(300, 2) * 10  # Initial random dataset
kmeans = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/initialize', methods=['POST'])
def initialize_kmeans():
    global kmeans
    data = request.json
    init_method = data['method']
    num_clusters = int(data['numClusters'])

    # Initialize KMeans
    kmeans = KMeansInteractive(n_clusters=num_clusters, init_method=init_method)
    kmeans.initialize_centroids(X)

    return jsonify({"status": f"KMeans initialized with {num_clusters} clusters using {init_method} method"})

@app.route('/step', methods=['POST'])
def step():
    global kmeans
    if kmeans is None:
        return jsonify({"status": "KMeans not initialized"})

    # Perform one step of KMeans
    converged = kmeans.fit_step(X)

    return jsonify({"converged": converged})

@app.route('/converge', methods=['POST'])
def run_to_convergence():
    global kmeans
    if kmeans is None:
        return jsonify({"status": "KMeans not initialized"})

    converged = False
    while not converged:
        converged = kmeans.fit_step(X)

    return jsonify({"converged": converged})

# Dynamic plot generation
@app.route('/plot.png')
def plot_png():
    fig = create_figure()  # Dynamically create the plot
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    global kmeans, X
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)

    if kmeans is not None and kmeans.labels is not None and kmeans.centroids is not None:
        # Plot data points and centroids only if KMeans is initialized
        axis.scatter(X[:, 0], X[:, 1], c=kmeans.labels, cmap='viridis', marker='o')
        axis.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='x', s=100)
        axis.set_title("KMeans Clustering Data")
    else:
        axis.set_title("KMeans not initialized or no data to display")

    return fig

@app.route('/new_dataset', methods=['POST'])
def new_dataset():
    global X
    # Generate a new random dataset
    X = np.random.rand(300, 2) * 10
    return jsonify({"status": "New dataset generated"})

if __name__ == '__main__':
    app.run(debug=True)