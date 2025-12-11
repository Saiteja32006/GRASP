import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx

def create_sample_graph(directed=True):
    """
    Creates a 5-node sample graph in CSR format.

    Args:
        directed (bool): If True, creates a directed graph; else undirected.

    Returns:
        csr_matrix: Adjacency matrix in CSR format.
    """
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    if not directed:
        edges += [(v, u) for u, v in edges]  # Add reverse edges

    rows, cols = zip(*edges)
    data = np.ones(len(rows), dtype=int)
    return csr_matrix((data, (rows, cols)), shape=(5, 5))


def csr_to_networkx(A_csr, directed=True):
    """
    Converts a CSR adjacency matrix to a NetworkX graph.

    Args:
        A_csr (csr_matrix): Sparse adjacency matrix.
        directed (bool): Whether to create a directed or undirected graph.

    Returns:
        networkx.Graph or networkx.DiGraph: The constructed graph.
    """
    G = nx.DiGraph() if directed else nx.Graph()
    rows, cols = A_csr.nonzero()
    G.add_edges_from(zip(rows, cols))
    return G
