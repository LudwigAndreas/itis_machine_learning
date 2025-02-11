import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# KNP algorithm
def remove_longest_edges(mst, k):
    edges = sorted(mst.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    
    edges_to_remove = edges[:k-1]
    for u, v, _ in edges_to_remove:
        mst.remove_edge(u, v)
    
    return mst, edges_to_remove

def main():
    n = 10
    dist = np.zeros((n, n))
    g = nx.Graph()

    for i in range(n):
        for j in range(i + 1, n):
            if np.random.random() > 0.4:
                dist[i][j] = np.random.randint(10, 100)
                g.add_edge(i, j, weight=dist[i][j])

    positions = nx.spring_layout(g)
    nx.draw(g, positions, with_labels=True)
    edge_labels = nx.get_edge_attributes(g, 'weight')
    nx.draw_networkx_edge_labels(g, positions, edge_labels=edge_labels)
    plt.show()

    minimal_tree = nx.minimum_spanning_tree(g)
    nx.draw(minimal_tree, positions, edge_color = "red")
    plt.show()

    clustered_mst, removed_edges = remove_longest_edges(minimal_tree, 2)

    nx.draw(clustered_mst, positions, edge_color='red')
    plt.show()



if __name__ == '__main__':
    main()