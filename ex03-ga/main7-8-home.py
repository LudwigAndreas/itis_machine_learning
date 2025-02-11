import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# KNP (Kruskal's Minimum Spanning Tree) implementation
def kruskal_mst(graph):
    # Sort edges by weight
    sorted_edges = sorted(graph.edges(data=True), key=lambda x: x[2]['weight'])
    mst = nx.Graph()
    
    # Create subsets for union-find to detect cycles
    parent = {}
    rank = {}
    
    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])  # Path compression
        return parent[node]

    def union(node1, node2):
        root1 = find(node1)
        root2 = find(node2)
        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            elif rank[root1] < rank[root2]:
                parent[root1] = root2
            else:
                parent[root2] = root1
                rank[root1] += 1

    # Initialize union-find sets
    for node in graph.nodes:
        parent[node] = node
        rank[node] = 0

    # Add edges to MST, ensuring no cycles
    for u, v, data in sorted_edges:
        if find(u) != find(v):
            mst.add_edge(u, v, weight=data['weight'])
            union(u, v)

    return mst

def remove_longest_edges(mst, k):
    # Get edges sorted by weight in descending order
    edges = sorted(mst.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    
    # Remove k-1 longest edges
    edges_to_remove = edges[:k-1]
    for u, v, _ in edges_to_remove:
        mst.remove_edge(u, v)
    
    return mst, edges_to_remove

def main():
    n = 10  # Number of nodes
    k = 2   # Number of clusters
    dist = np.zeros((n, n))
    g = nx.Graph()

    # Generate random weighted graph
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.random() > 0.4:
                dist[i][j] = np.random.randint(10, 100)
                g.add_edge(i, j, weight=dist[i][j])

    # Calculate Minimum Spanning Tree using Kruskal's algorithm
    mst = kruskal_mst(g)

    # Visualization: Original graph with MST highlighted
    positions = nx.spring_layout(g)  # Layout for nodes
    plt.figure(figsize=(10, 7))
    nx.draw(g, positions, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
    edge_labels = nx.get_edge_attributes(g, 'weight')
    nx.draw_networkx_edge_labels(g, positions, edge_labels=edge_labels)
    nx.draw_networkx_edges(g, positions, edgelist=mst.edges(data=True), edge_color='red', width=2)
    plt.title("Graph with Kruskal's MST Highlighted")
    plt.show()

    # Remove k-1 longest edges to form k clusters
    clustered_mst, removed_edges = remove_longest_edges(mst, k)

    # Visualization: Clusters after removing edges
    plt.figure(figsize=(10, 7))
    nx.draw(clustered_mst, positions, with_labels=True, node_color='lightblue', node_size=500, edge_color='red', width=2)
    plt.title(f"Graph with {k} Clusters (Removed {k-1} Longest Edges)")
    plt.show()

    # Print removed edges
    print(f"Removed edges to form {k} clusters: {removed_edges}")

if __name__ == '__main__':
    main()
