import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris

iris = load_iris()


dots = iris.data


# функция для расчета WCSS (внутрикластерной суммы квадратов отклонений)
def compute_wcss(data, centroids, labels):
    wcss = 0
    for i, centroid in enumerate(centroids):
        cluster_points = data[labels == i]
        wcss += np.sum((cluster_points - centroid) ** 2)
    return wcss


# Функция для выполнения кластеризации методом K-means:
def k_means(data, k, max_iters=100):
    np.random.seed(42)
    centroids = data[np.random.choice(range(len(data)), size=k, replace=False)]
    centroids_history = [centroids.copy()]

    for _ in range(max_iters):

        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = []
        for i in range(k):
            cluster_points = data[labels == i]
            if len(cluster_points) == 0:
                # случайная точка
                new_centroids.append(data[np.random.choice(range(len(data)))])
            else:
                # среднее всех точек
                new_centroids.append(cluster_points.mean(axis=0))
        # обновление центройдов
        new_centroids = np.array(new_centroids)

        centroids_history.append(new_centroids.copy())

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, labels, centroids_history

# списки для метрик
wcss_values = []
silhouette_values = []
ks = range(2, 10)
for k in ks:
    centroids, labels, _ = k_means(dots, k)
    wcss_values.append(compute_wcss(dots, centroids, labels))

    if k > 1:
        silhouette_avg = silhouette_score(dots, labels)
        silhouette_values.append(silhouette_avg)
    else:
        silhouette_values.append(-1)


# рисуем локоть
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(ks, wcss_values, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")

# рисуем sihouette_values
plt.subplot(1, 2, 2)
plt.plot(ks, silhouette_values, marker='o', color='green')
plt.title("Silhouette Score")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.show()

# точка локтя
optimal_k_elbow = ks[np.argmin(np.diff(wcss_values)) + 1]
# пик по sihouette
optimal_k_silhouette = ks[np.argmax(silhouette_values)]

print(f"Optimal number of clusters based on Elbow Method: {optimal_k_elbow}")
print(f"Optimal number of clusters based on Silhouette Score: {
      optimal_k_silhouette}")


# можем выбрать один из методов
optimal_k = optimal_k_silhouette


# вычисляем k_means
centroids, labels, centroids_history = k_means(dots, optimal_k)


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
iteration = 0

# рисуем итерацию
def plot_iteration(iteration):
    ax.clear()
    centroids = centroids_history[iteration]
    distances = np.linalg.norm(dots[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    for i, centroid in enumerate(centroids):
        cluster_points = dots[labels == i]
        ax.scatter(cluster_points[:, 0],
                   cluster_points[:, 1], label=f"Cluster {i}")
        ax.scatter(centroid[0], centroid[1], marker='x', s=200, c='black')
    ax.set_title(f"Iteration {iteration + 1}")
    ax.legend()
    plt.draw()


plot_iteration(iteration)


# для кнопки next
def next_iteration(event):
    global iteration
    if iteration < len(centroids_history) - 1:
        iteration += 1
        plot_iteration(iteration)
    if iteration == len(centroids_history) - 1:
        btn_next.ax.set_visible(False)
        plt.draw()


ax_next = plt.axes([0.8, 0.05, 0.1, 0.075])
btn_next = Button(ax_next, 'Next')
btn_next.on_clicked(next_iteration)

plt.show()
