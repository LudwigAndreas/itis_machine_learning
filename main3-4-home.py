import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from sklearn.metrics import silhouette_score

# Generate random data points
dots = []
for i in range(15):
    dots.append([np.random.randint(0, 5), np.random.randint(0, 5)])
    dots.append([10 + np.random.randint(0, 5), np.random.randint(0, 5)])
    dots.append([np.random.randint(0, 5), 10 + np.random.randint(0, 5)])
    dots.append([10 + np.random.randint(0, 5), 10 + np.random.randint(0, 5)])
dots = np.array(dots)

# Function to compute WCSS
def compute_wcss(data, centroids, labels):
    wcss = 0
    for i, centroid in enumerate(centroids):
        cluster_points = data[labels == i]
        wcss += np.sum((cluster_points - centroid) ** 2)
    return wcss

# K-means implementation
def k_means(data, k, max_iters=100):
    np.random.seed(42)  # Ensure reproducibility
    centroids = data[np.random.choice(range(len(data)), size=k, replace=False)]
    centroids_history = [centroids.copy()]
    
    for _ in range(max_iters):
        # Step 1: Assign points to the nearest centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Step 2: Recalculate centroids
        new_centroids = []
        for i in range(k):
            cluster_points = data[labels == i]
            if len(cluster_points) == 0:
                # Handle empty cluster by reinitializing to a random point
                new_centroids.append(data[np.random.choice(range(len(data)))])
            else:
                new_centroids.append(cluster_points.mean(axis=0))
        new_centroids = np.array(new_centroids)
        
        centroids_history.append(new_centroids.copy())
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return centroids, labels, centroids_history

# Calculate WCSS and Silhouette Score for different values of k
wcss_values = []
silhouette_values = []
ks = range(2, 10)  # Start at k=2 because silhouette score is not defined for k=1
for k in ks:
    centroids, labels, _ = k_means(dots, k)
    wcss_values.append(compute_wcss(dots, centroids, labels))
    
    # Calculate Silhouette Score
    if k > 1:
        silhouette_avg = silhouette_score(dots, labels)
        silhouette_values.append(silhouette_avg)
    else:
        silhouette_values.append(-1)  # Silhouette is not defined for k=1

# Plot WCSS for the Elbow Method
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(ks, wcss_values, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")

# Plot Silhouette Scores for each k
plt.subplot(1, 2, 2)
plt.plot(ks, silhouette_values, marker='o', color='green')
plt.title("Silhouette Score")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.show()

# Choose optimal k using both methods
optimal_k_elbow = ks[np.argmin(np.diff(wcss_values)) + 1]  # Find elbow using WCSS
optimal_k_silhouette = ks[np.argmax(silhouette_values)]  # Find optimal k using silhouette score

print(f"Optimal number of clusters based on Elbow Method: {optimal_k_elbow}")
print(f"Optimal number of clusters based on Silhouette Score: {optimal_k_silhouette}")

# Choose k that is optimal in both methods (or you can choose one based on your preference)
optimal_k = optimal_k_silhouette  # You can also use `optimal_k_elbow` here if preferred

# Perform final k-means with the optimal number of clusters
centroids, labels, centroids_history = k_means(dots, optimal_k)

# Interactive plot with "Next" button
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)  # Adjust space for the button
iteration = 0

def plot_iteration(iteration):
    ax.clear()
    centroids = centroids_history[iteration]
    distances = np.linalg.norm(dots[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    for i, centroid in enumerate(centroids):
        cluster_points = dots[labels == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}")
        ax.scatter(centroid[0], centroid[1], marker='x', s=200, c='black')  # Centroid
    ax.set_title(f"Iteration {iteration + 1}")
    ax.legend()
    plt.draw()

plot_iteration(iteration)

# Callback for the "Next" button
def next_iteration(event):
    global iteration
    if iteration < len(centroids_history) - 1:
        iteration += 1
        plot_iteration(iteration)
    if iteration == len(centroids_history) - 1:
        btn_next.ax.set_visible(False)  # Hide the button when the final iteration is reached
        plt.draw()

# Add the "Next" button
ax_next = plt.axes([0.8, 0.05, 0.1, 0.075])  # Position of the button
btn_next = Button(ax_next, 'Next')
btn_next.on_clicked(next_iteration)

plt.show()
