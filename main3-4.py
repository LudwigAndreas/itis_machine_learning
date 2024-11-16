from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dots = []
    for i in range(15):
        dots.append([np.random.randint(0, 5), np.random.randint(0, 5)])
        dots.append([10 + np.random.randint(0, 5), np.random.randint(0, 5)])
        dots.append([np.random.randint(0, 5), 10 + np.random.randint(0, 5)])
        dots.append([10 + np.random.randint(0, 5), 10 + np.random.randint(0, 5)])

    dots = np.array(dots)

    # Elbow Method: Sum of Squared Errors (SSE)
    sse = []
    silhouette_scores = []
    k_range = range(1, 11)  # Test for k = 1 to k = 10

    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(dots)
        
        sse.append(kmeans.inertia_)  # Inertia gives the sum of squared errors (SSE)
        
        # Silhouette Score, only calculate for k > 1
        if k > 1:
            score = silhouette_score(dots, kmeans.labels_)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(-1)  # Silhouette score is not defined for k=1

    # Plotting Elbow Method
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(k_range, sse, marker='o')
    plt.title('Elbow Method (SSE)')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.grid(True)

    # Plotting Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(k_range[1:], silhouette_scores[1:], marker='o', color='green')
    plt.title('Silhouette Score')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Find optimal k as the one with the highest silhouette score
    optimal_k = k_range[1:][np.argmax(silhouette_scores[1:])]
    print(f"Optimal number of clusters: {optimal_k}")

    # Final clustering with optimal_k
    kmeans_optimal = KMeans(n_clusters=optimal_k)
    kmeans_optimal.fit(dots)
    
    # Visualize clustering result
    plt.scatter(dots[:, 0], dots[:, 1], c=kmeans_optimal.labels_, cmap='viridis')
    plt.title(f'KMeans Clustering (k={optimal_k})')
    plt.show()
