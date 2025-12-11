import numpy as np
from scipy.sparse import csr_matrix
import random

def burkhardt_bfs(A_csr: csr_matrix, source: int):
    """Burkhardt's optimal BFS implementation"""
    n = A_csr.shape[0]
    visited = np.zeros(n, dtype=bool)
    distance = np.full(n, -1, dtype=int)
    
    visited[source] = True
    distance[source] = 0
    frontier = [source]
    
    while frontier:
        next_frontier = []
        for u in frontier:
            neighbors = A_csr.indices[A_csr.indptr[u]:A_csr.indptr[u+1]]
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    distance[v] = distance[u] + 1
                    next_frontier.append(v)
        frontier = next_frontier
    return distance

def grasp_traversal(A_csr: csr_matrix, source: int, iterations: int = 10, alpha: float = 0.3):
    n = A_csr.shape[0]
    best_path = []
    best_coverage = 0

    for _ in range(iterations):
        visited = np.zeros(n, dtype=bool)
        unvisited = set(range(n))
        current_path = []
        visited[source] = True
        unvisited.remove(source)
        current_node = source
        current_path.append(current_node)

        while unvisited:
            neighbors = A_csr.indices[A_csr.indptr[current_node]:A_csr.indptr[current_node+1]]
            active_neighbors = [v for v in neighbors if v in unvisited]
            
            if not active_neighbors:
                if unvisited:
                    current_node = next(iter(unvisited))
                    current_path.append(current_node)
                    visited[current_node] = True
                    unvisited.remove(current_node)
                continue
            
            scores = {
                v: sum(1 for w in A_csr.indices[A_csr.indptr[v]:A_csr.indptr[v+1]] if w in unvisited)
                for v in active_neighbors
            }
            
            sorted_nodes = sorted(scores.items(), key=lambda x: -x[1])
            rcl_size = max(1, int(len(sorted_nodes) * alpha))
            rcl = [v for v, _ in sorted_nodes[:rcl_size]]
            
            current_node = random.choice(rcl)
            current_path.append(current_node)
            visited[current_node] = True
            unvisited.remove(current_node)
            
        coverage = len(current_path)
        if coverage > best_coverage:
            best_path = current_path.copy()
            best_coverage = coverage
            
    return best_path, best_coverage

def standard_grasp_traversal(A_csr: csr_matrix, source: int, iterations: int = 10, alpha: float = 0.3):
    n = A_csr.shape[0]
    best_path = []
    best_coverage = 0

    for _ in range(iterations):
        visited = np.zeros(n, dtype=bool)
        current_path = [source]
        visited[source] = True
        current_node = source

        while True:
            neighbors = A_csr.indices[A_csr.indptr[current_node]:A_csr.indptr[current_node+1]]
            active_neighbors = [v for v in neighbors if not visited[v]]
            
            if not active_neighbors:
                break
                
            scores = {v: len(A_csr.indices[A_csr.indptr[v]:A_csr.indptr[v+1]]) for v in active_neighbors}
            sorted_nodes = sorted(scores.items(), key=lambda x: -x[1])
            rcl_size = max(1, int(len(sorted_nodes) * alpha))
            rcl = [v for v, _ in sorted_nodes[:rcl_size]]
            current_node = random.choice(rcl)
            current_path.append(current_node)
            visited[current_node] = True
            
        coverage = len(current_path)
        if coverage > best_coverage:
            best_path = current_path.copy()
            best_coverage = coverage
            
    return best_path, best_coverage