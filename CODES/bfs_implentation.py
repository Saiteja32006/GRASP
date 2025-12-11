import numpy as np
from scipy.sparse import csr_matrix
from collections import deque

def burkhardt_bfs(A_csr: csr_matrix, source: int):
    """
    Optimal algebraic BFS for sparse graphs (Burkhardt's method).
    Returns: Array of distances from the source node.
    """
    n = A_csr.shape[0]
    visited = np.zeros(n, dtype=bool)
    distance = np.full(n, -1, dtype=int)
    
    visited[source] = True
    distance[source] = 0
    frontier = [source]
    
    while frontier:
        next_frontier = []
        for u in frontier:
            # Get neighbors from CSR format: indices[indptr[u]:indptr[u+1]]
            neighbors = A_csr.indices[A_csr.indptr[u]:A_csr.indptr[u+1]]
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    distance[v] = distance[u] + 1
                    next_frontier.append(v)
        frontier = next_frontier
    return distance