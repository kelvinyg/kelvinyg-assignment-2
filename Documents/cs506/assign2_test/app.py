import io
from flask import Flask, render_template, jsonify, request
import numpy as np
import plotly.graph_objs as go
from kmeans import KMeansInteractive  # Assuming KMeans logic is in kmeans.py

app = Flask(__name__)

# Global variables for the dataset, KMeans instance, and manual centroids
X = np.random.rand(300, 2) * 10  # Initial random dataset
kmeans = None
manual_centroids = []  # Store user-selected centroids

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/initialize', methods=['POST'])
def initialize_kmeans():
    global kmeans, manual_centroids
    data = request.json
    init_method = data['method']
    num_clusters = int(data['numClusters'])

    # Initialize KMeans
    if init_method == 'manual':
        manual_centroids = []  # Reset manual centroids when initializing manually
        kmeans = KMeansInteractive(n_clusters=num_clusters, init_method=init_method)
        return jsonify({"status": f"KMeans initialized with {num_clusters} clusters in Manual Selection mode. Please select {num_clusters} centroids."})
    else:
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

@app.route('/select_centroid', methods=['POST'])
def select_centroid():
    global manual_centroids, kmeans
    data = request.json
    x = data['x']
    y = data['y']

    if kmeans is None or kmeans.init_method != 'manual':
        return jsonify({"status": "KMeans not in manual mode or not initialized."})

    # Ensure that we are not adding more than the required number of centroids
    if len(manual_centroids) < kmeans.n_clusters:
        manual_centroids.append([x, y])

        # If all centroids are selected, initialize KMeans with these centroids
        if len(manual_centroids) == kmeans.n_clusters:
            kmeans.centroids = np.array(manual_centroids)
            return jsonify({"status": f"All {kmeans.n_clusters} centroids selected. You can now step through KMeans."})
        
        return jsonify({
            "status": f"Centroid {len(manual_centroids)} of {kmeans.n_clusters} added.",
            "num_selected": len(manual_centroids)
        })
    else:
        return jsonify({"status": "All centroids have already been selected or KMeans not initialized."})

# Dynamic plot generation
@app.route('/plot', methods=['POST'])
def plot():
    global kmeans, X
    fig = create_figure()  # Dynamically create the plot
    
    # Convert the Plotly figure to a JSON-serializable dictionary
    return jsonify(fig.to_dict())  # Convert Plotly figure to JSON serializable format

def create_figure():
    global kmeans, X, manual_centroids
    scatter_data = []

    # Plot data points, colored by their cluster assignment if available
    if kmeans is not None and kmeans.labels is not None:
        scatter_data.append(go.Scatter(
            x=X[:, 0].tolist(),  # Convert ndarray to list
            y=X[:, 1].tolist(),  # Convert ndarray to list
            mode='markers',
            marker=dict(
                size=10,
                color=kmeans.labels.tolist(),  # Color points by cluster assignment
                colorscale='Viridis',  # Use a color scale to differentiate clusters
                opacity=0.7
            ),
            name='Data Points'
        ))
    else:
        # If no clusters have been assigned yet, plot all points in the same color
        scatter_data.append(go.Scatter(
            x=X[:, 0].tolist(),
            y=X[:, 1].tolist(),
            mode='markers',
            marker=dict(size=10, color='blue', opacity=0.5),
            name='Data Points'
        ))

    # Plot selected manual centroids (in red)
    if manual_centroids:
        scatter_data.append(go.Scatter(
            x=[c[0] for c in manual_centroids],
            y=[c[1] for c in manual_centroids],
            mode='markers',
            marker=dict(size=15, color='red', symbol='x'),
            name='Selected Centroids'
        ))

    # Plot centroids (if initialized) (in black)
    if kmeans and kmeans.centroids is not None:
        scatter_data.append(go.Scatter(
            x=kmeans.centroids[:, 0].tolist(),
            y=kmeans.centroids[:, 1].tolist(),
            mode='markers',
            marker=dict(size=15, color='black', symbol='x'),
            name='Centroids'
        ))

    layout = go.Layout(
        title="KMeans Clustering Data",
        xaxis=dict(title="X"),
        yaxis=dict(title="Y"),
        clickmode='event+select'  # Enable click events
    )

    return go.Figure(data=scatter_data, layout=layout)

# Add the converge route
@app.route('/converge', methods=['POST'])
def run_to_convergence():
    global kmeans
    if kmeans is None:
        return jsonify({"status": "KMeans not initialized"})

    converged = False
    while not converged:
        converged = kmeans.fit_step(X)

    return jsonify({"converged": True})

@app.route('/new_dataset', methods=['POST'])
def new_dataset():
    global X, kmeans, manual_centroids
    # Generate a new random dataset
    X = np.random.rand(300, 2) * 10
    kmeans = None  # Reset KMeans when a new dataset is generated
    manual_centroids = []  # Reset manual centroids

    return jsonify({"status": "New dataset generated"})

if __name__ == '__main__':
    app.run(debug=True)