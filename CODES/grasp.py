"""
Optimal Algebraic BFS (Burkhardt) + GRASP Adaptation for Sparse Graphs
Author: A Sai Teja, Donil Jaison, M Jaswanth
"""
import numpy as np
from scipy.sparse import csr_matrix
import random
import matplotlib.pyplot as plt
import networkx as nx

# ---------------------------
# 1. Burkhardt's Optimal BFS
# ---------------------------

def burkhardt_bfs(A_csr: csr_matrix, source: int):
    """
    Implements the optimal algebraic BFS from Burkhardt's paper.
    Uses CSR format for efficient sparse matrix operations.
    
    Args:
        A_csr: Adjacency matrix in Compressed Sparse Row format
        source: Starting node for BFS
        
    Returns:
        Array of distances from source node
    """
    n = A_csr.shape[0]
    visited = np.zeros(n, dtype=bool)
    distance = np.full(n, -1, dtype=int)
    
    # Initialize with source node
    visited[source] = True
    distance[source] = 0
    frontier = [source]
    
    while frontier:
        next_frontier = []
        for u in frontier:
            # Get neighbors using CSR structure
            start_ptr = A_csr.indptr[u]
            end_ptr = A_csr.indptr[u+1]
            neighbors = A_csr.indices[start_ptr:end_ptr]
            
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    distance[v] = distance[u] + 1
                    next_frontier.append(v)
        frontier = next_frontier
        
    return distance

# ---------------------------
# 2. GRASP Adaptation
# ---------------------------

def grasp_traversal(A_csr: csr_matrix, source: int, iterations: int = 10, alpha: float = 0.3):
    """
    GRASP-based graph traversal with Burkhardt-inspired masking
    
    Args:
        A_csr: Adjacency matrix in CSR format
        source: Starting node
        iterations: Number of GRASP iterations
        alpha: Top percentage for Restricted Candidate List (RCL)
        
    Returns:
        Tuple of (best path as list of nodes, coverage as number of nodes visited)
    """
    n = A_csr.shape[0]
    best_path = []
    best_coverage = 0

    for _ in range(iterations):
        visited = np.zeros(n, dtype=bool)
        unvisited = set(range(n))  # Track unvisited nodes
        current_path = []
        visited[source] = True
        unvisited.remove(source)
        current_node = source
        current_path.append(current_node)

        while unvisited:
            # Get neighbors, masking visited nodes
            neighbors = A_csr.indices[A_csr.indptr[current_node]:A_csr.indptr[current_node+1]]
            active_neighbors = [v for v in neighbors if v in unvisited]
            
            if not active_neighbors:
                # If no unvisited neighbors, pick a new unvisited node to continue
                if unvisited:
                    current_node = next(iter(unvisited))  # Pick the first unvisited node
                    current_path.append(current_node)
                    visited[current_node] = True
                    unvisited.remove(current_node)
                continue
                
            # Greedy evaluation: Node with most unvisited neighbors
            scores = {
                v: sum(1 for w in A_csr.indices[A_csr.indptr[v]:A_csr.indptr[v+1]] if w in unvisited)
                for v in active_neighbors
            }
            
            # Build Restricted Candidate List (RCL)
            sorted_nodes = sorted(scores.items(), key=lambda x: -x[1])
            rcl_size = max(1, int(len(sorted_nodes) * alpha))
            rcl = [v for v, _ in sorted_nodes[:rcl_size]]
            
            # Randomized selection
            current_node = random.choice(rcl)
            current_path.append(current_node)
            visited[current_node] = True
            unvisited.remove(current_node)
            
        # Update best solution based on coverage
        coverage = len(current_path)
        if coverage > best_coverage:
            best_path = current_path.copy()
            best_coverage = coverage
            
    return best_path, best_coverage

# ---------------------------
# 3. Test Case & Visualization
# ---------------------------

def create_sample_graph():
    """Create a simple sparse graph for testing"""
    edges = [
        (0, 1), (0, 2),
        (1, 3),
        (2, 3), (2, 4),
        (3, 4)
    ]
    n_nodes = 5
    rows, cols = zip(*edges)
    data = np.ones(len(rows), dtype=int)
    return csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

def visualize_graph(A_csr, path):
    """Visualize graph with highlighted path"""
    G = nx.DiGraph()
    G.add_edges_from(zip(*A_csr.nonzero()))
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue')
    
    # Highlight path
    path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='red')
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
    plt.show()

# ---------------------------
# Main Execution
# ---------------------------

if __name__ == "__main__":
    # Create sample graph
    A_csr = create_sample_graph()
    
    # Run Burkhardt's BFS
    bfs_distances = burkhardt_bfs(A_csr, source=0)
    print("Burkhardt BFS Distances:", bfs_distances)
    
    # Run GRASP traversal
    grasp_path, grasp_coverage = grasp_traversal(A_csr, source=0, iterations=5)
    print("GRASP Path:", grasp_path)
    print("GRASP Coverage:", grasp_coverage, "nodes")
    
    # Visualize results
    visualize_graph(A_csr, grasp_path)