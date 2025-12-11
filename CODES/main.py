from algorithms import burkhardt_bfs, grasp_traversal
from graph_utils import create_sample_graph

def main():
    A_csr = create_sample_graph()
    
    print("=== Burkhardt BFS ===")
    print("Distances:", burkhardt_bfs(A_csr, 0))
    
    print("\n=== GRASP Path ===")
    print("Path:", grasp_traversal(A_csr, 0))

if __name__ == "__main__":
    main()