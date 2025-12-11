import matplotlib
matplotlib.use('TkAgg')

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
import multiprocessing as mp
from algorithms import burkhardt_bfs, grasp_traversal, standard_grasp_traversal
from graph_utils import csr_to_networkx, create_sample_graph
from scipy.sparse import csr_matrix
from memory_profiler import memory_usage
import psutil

class GraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Graph Algorithm Visualizer")

        self.graphs = []
        self.current_graph_idx = -1
        self.A_csr = None
        self.pos = None
        self.process = psutil.Process()  # For memory usage tracking

        # Control frame for buttons and input
        control_frame = ttk.Frame(root)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.algo_var = tk.StringVar(value="bfs")
        ttk.Radiobutton(control_frame, text="Burkhardt BFS", variable=self.algo_var, value="bfs").pack(side=tk.LEFT)
        ttk.Radiobutton(control_frame, text="Optimized GRASP", variable=self.algo_var, value="grasp").pack(side=tk.LEFT)
        ttk.Radiobutton(control_frame, text="Standard GRASP", variable=self.algo_var, value="standard_grasp").pack(side=tk.LEFT)

        ttk.Button(control_frame, text="Run Algorithm", command=self.run_algorithm).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Run Experiments", command=self.run_experiments).pack(side=tk.LEFT)

        # Nodes input
        ttk.Label(control_frame, text="Nodes (max 100):").pack(side=tk.LEFT, padx=5)
        self.nodes_entry = ttk.Entry(control_frame, width=5)
        self.nodes_entry.pack(side=tk.LEFT)
        self.nodes_entry.insert(0, "5")  # Default value

        # Number of Edges input (new)
        ttk.Label(control_frame, text="Edges (max depends on nodes):").pack(side=tk.LEFT, padx=5)
        self.edges_num_entry = ttk.Entry(control_frame, width=5)
        self.edges_num_entry.pack(side=tk.LEFT)
        self.edges_num_entry.insert(0, "5")  # Default value

        # Updated buttons to use number of nodes and edges
        ttk.Button(control_frame, text="Run Exp (Custom Nodes & Edges)", command=self.run_experiments_custom_nodes_edges).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Create & Run (Custom Nodes & Edges)", command=self.create_and_run_custom_graph_edges).pack(side=tk.LEFT)

        self.figure = plt.Figure(figsize=(12, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.output = tk.Text(root, height=15)
        self.output.pack(side=tk.BOTTOM, fill=tk.X)

        self.output.insert(tk.END, "Enter number of nodes and edges to create a random graph, then run experiments or algorithms.\n")

    def generate_random_graph(self, n_nodes, num_edges):
        """Generate a random graph with n_nodes nodes and approximately num_edges edges."""
        if n_nodes < 1:
            n_nodes = 1
            messagebox.showwarning("Warning", "Number of nodes set to 1 (minimum).")
        if n_nodes > 100:
            n_nodes = 100
            messagebox.showwarning("Warning", "Max nodes is 100. Set to 100.")

        # Maximum possible edges for n_nodes (complete graph: n * (n-1) / 2)
        max_edges = (n_nodes * (n_nodes - 1)) // 2
        if num_edges < 0:
            num_edges = 0
            messagebox.showwarning("Warning", "Number of edges set to 0 (minimum).")
        if num_edges > max_edges:
            num_edges = max_edges
            messagebox.showwarning("Warning", f"Max edges for {n_nodes} nodes is {max_edges}. Set to {max_edges}.")

        # Generate a random graph using Erdos-Renyi model, adjusting probability to match num_edges
        p = (2 * num_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
        G = nx.erdos_renyi_graph(n_nodes, p, directed=False)

        # Adjust the number of edges to be as close as possible to num_edges
        current_edges = G.number_of_edges()
        if current_edges < num_edges:
            # Add edges
            possible_edges = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes) if not G.has_edge(i, j)]
            np.random.shuffle(possible_edges)
            for i in range(min(num_edges - current_edges, len(possible_edges))):
                u, v = possible_edges[i]
                G.add_edge(u, v)
        elif current_edges > num_edges:
            # Remove edges
            edges = list(G.edges())
            np.random.shuffle(edges)
            for i in range(current_edges - num_edges):
                u, v = edges[i]
                G.remove_edge(u, v)

        edges = list(G.edges())
        if not edges:  # Ensure at least one edge if num_edges > 0
            edges = [(0, 1 % n_nodes)]
        rows, cols = zip(*edges)
        data = np.ones(len(rows), dtype=int)
        return csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    def run_experiments_custom_nodes_edges(self):
        try:
            n_nodes = int(self.nodes_entry.get())
            num_edges = int(self.edges_num_entry.get())
            if n_nodes < 1 or n_nodes > 100:
                messagebox.showerror("Error", "Number of nodes must be between 1 and 100.")
                return
            max_edges = (n_nodes * (n_nodes - 1)) // 2
            if num_edges < 0 or num_edges > max_edges:
                messagebox.showerror("Error", f"Number of edges must be between 0 and {max_edges} for {n_nodes} nodes.")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for nodes and edges.")
            return

        self.output.delete(1.0, tk.END)
        self.output.insert(tk.END, f"Running Experiments on Random Graph with {n_nodes} Nodes and {num_edges} Edges\n\n")

        graph = self.generate_random_graph(n_nodes, num_edges)
        actual_edges = graph.nnz // 2  # Actual number of edges after generation
        self.output.insert(tk.END, f"Generated Graph with {actual_edges} Edges\n")
        results, paths, metrics = self.run_single_graph_experiments(graph, n_nodes)
        graph_types = [f"Random Graph ({n_nodes} nodes)"]

        for algo, result in results.items():
            time, accuracy, ops = result[0]
            path = paths[algo][0]
            self.output.insert(tk.END, f"{algo}: Time = {time:.6f}s, Accuracy = {accuracy:.2f}%, Operations = {ops:.0f}, Path = {path}\n")

        # Output additional metrics
        self.output_metrics(metrics)

        self.plot_results(results, graph_types)

    def create_and_run_custom_graph_edges(self):
        try:
            n_nodes = int(self.nodes_entry.get())
            num_edges = int(self.edges_num_entry.get())
            if n_nodes < 1 or n_nodes > 100:
                messagebox.showerror("Error", "Number of nodes must be between 1 and 100.")
                return
            max_edges = (n_nodes * (n_nodes - 1)) // 2
            if num_edges < 0 or num_edges > max_edges:
                messagebox.showerror("Error", f"Number of edges must be between 0 and {max_edges} for {n_nodes} nodes.")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for nodes and edges.")
            return

        self.A_csr = self.generate_random_graph(n_nodes, num_edges)
        self.graphs = [self.A_csr]  # Update graphs list
        self.current_graph_idx = 0
        G = csr_to_networkx(self.A_csr)
        for i in range(self.A_csr.shape[0]):
            G.add_node(i)
        try:
            self.pos = nx.spring_layout(G, k=1/np.sqrt(self.A_csr.shape[0]), iterations=100)
        except:
            self.output.insert(tk.END, "Spring layout failed, using circular layout instead.\n")
            self.pos = nx.circular_layout(G)
        self.draw_default_graph()
        actual_edges = self.A_csr.nnz // 2
        self.output.delete(1.0, tk.END)
        self.output.insert(tk.END, f"Created Random Graph with {n_nodes} nodes and {actual_edges} edges\n")
        self.run_algorithm()

    def draw_default_graph(self):
        if self.A_csr is None:
            return
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        G = csr_to_networkx(self.A_csr)
        for i in range(self.A_csr.shape[0]):
            G.add_node(i)
        nodes = list(range(self.A_csr.shape[0]))
        nx.draw(G, self.pos, ax=ax, with_labels=True, node_color='lightblue', nodelist=nodes)
        self.canvas.draw()

    def run_experiments(self):
        if not self.graphs:
            messagebox.showinfo("No Graphs", "Please create a graph using 'Create & Run (Custom Nodes & Edges)' first.")
            return

        self.output.delete(1.0, tk.END)
        self.output.insert(tk.END, "Running Experiments on All Created Graphs...\n\n")

        results = {'BFS': [], 'Optimized GRASP': [], 'Standard GRASP': []}
        paths = {'BFS': [], 'Optimized GRASP': [], 'Standard GRASP': []}
        metrics_list = []
        graph_types = [f"Custom Graph {i+1}" for i in range(len(self.graphs))]

        for idx, graph in enumerate(self.graphs):
            n_nodes = graph.shape[0]
            actual_edges = graph.nnz // 2
            self.output.insert(tk.END, f"Graph {idx + 1} - {n_nodes} Nodes, {actual_edges} Edges\n")
            result, path, metrics = self.run_single_graph_experiments(graph, n_nodes)
            metrics_list.append(metrics)
            for algo in results:
                results[algo].append(result[algo][0])
                paths[algo].append(path[algo][0])
                time, accuracy, ops = result[algo][0]
                self.output.insert(tk.END, f"{algo}: Time = {time:.6f}s, Accuracy = {accuracy:.2f}%, Operations = {ops:.0f}, Path = {path[algo][0]}\n")
            self.output.insert(tk.END, "\n")

        # Output averaged metrics across all graphs
        avg_metrics = self.average_metrics(metrics_list)
        self.output_metrics(avg_metrics)

        self.plot_results(results, graph_types)

    def run_single_graph_experiments(self, graph, n_nodes):
        """Run experiments on a single graph and collect metrics."""
        results = {'BFS': [], 'Optimized GRASP': [], 'Standard GRASP': []}
        paths = {'BFS': [], 'Optimized GRASP': [], 'Standard GRASP': []}
        source = 0
        runs = 5

        # Sequential runs
        bfs_times, bfs_ops, bfs_coverage, bfs_paths = [], [], 0, []
        grasp_times, grasp_ops, grasp_coverage, grasp_paths = [], [], 0, []
        std_grasp_times, std_grasp_ops, std_grasp_coverage, std_grasp_paths = [], [], 0, []

        # Parallel runs
        bfs_times_par, grasp_times_par, std_grasp_times_par = [], [], []

        # Memory usage tracking
        mem_before = self.process.memory_info().rss / 1024 / 1024  # MB

        for _ in range(runs):
            # Sequential BFS
            ops_count = 0
            start_time = time.time()
            distances, bfs_path = self.bfs_with_path(graph, source)
            bfs_times.append(time.time() - start_time)
            bfs_coverage = np.sum(distances >= 0)
            bfs_paths.append(bfs_path)
            for u in range(n_nodes):
                ops_count += len(graph.indices[graph.indptr[u]:graph.indptr[u+1]])
            bfs_ops.append(ops_count)

            # Sequential Optimized GRASP
            ops_count = 0
            start_time = time.time()
            path, coverage = grasp_traversal(graph, source)
            grasp_times.append(time.time() - start_time)
            grasp_coverage = coverage
            grasp_paths.append(path)
            for u in path:
                ops_count += len(graph.indices[graph.indptr[u]:graph.indptr[u+1]])
            grasp_ops.append(ops_count)

            # Sequential Standard GRASP
            ops_count = 0
            start_time = time.time()
            path_std, coverage_std = standard_grasp_traversal(graph, source)
            std_grasp_times.append(time.time() - start_time)
            std_grasp_coverage = coverage_std
            std_grasp_paths.append(path_std)
            for u in path_std:
                ops_count += len(graph.indices[graph.indptr[u]:graph.indptr[u+1]])
            std_grasp_ops.append(ops_count)

        # Parallel execution
        pool = mp.Pool(processes=3)
        for _ in range(runs):
            # Parallel BFS
            start_time = time.time()
            pool.apply_async(burkhardt_bfs, args=(graph, source)).get()
            bfs_times_par.append(time.time() - start_time)

            # Parallel Optimized GRASP
            start_time = time.time()
            pool.apply_async(grasp_traversal, args=(graph, source)).get()
            grasp_times_par.append(time.time() - start_time)

            # Parallel Standard GRASP
            start_time = time.time()
            pool.apply_async(standard_grasp_traversal, args=(graph, source)).get()
            std_grasp_times_par.append(time.time() - start_time)
        pool.close()
        pool.join()

        # Memory usage after
        mem_after = self.process.memory_info().rss / 1024 / 1024  # MB
        space_usage = mem_after - mem_before

        # Compute averages
        bfs_time = np.mean(bfs_times)
        bfs_accuracy = (bfs_coverage / n_nodes) * 100
        bfs_operations = np.mean(bfs_ops)
        best_bfs_path = bfs_paths[0]

        grasp_time = np.mean(grasp_times)
        grasp_accuracy = (grasp_coverage / n_nodes) * 100
        grasp_operations = np.mean(grasp_ops)
        best_grasp_path = grasp_paths[np.argmax([len(p) for p in grasp_paths])]

        std_grasp_time = np.mean(std_grasp_times)
        std_grasp_accuracy = (std_grasp_coverage / n_nodes) * 100
        std_grasp_operations = np.mean(std_grasp_ops)
        best_std_grasp_path = std_grasp_paths[np.argmax([len(p) for p in std_grasp_paths])]

        # Parallel times
        bfs_time_par = np.mean(bfs_times_par)
        grasp_time_par = np.mean(grasp_times_par)
        std_grasp_time_par = np.mean(std_grasp_times_par)

        # Solution quality (based on path length and coverage)
        bfs_quality = bfs_coverage / n_nodes + len(best_bfs_path) / n_nodes
        grasp_quality = grasp_coverage / n_nodes + len(best_grasp_path) / n_nodes
        std_grasp_quality = std_grasp_coverage / n_nodes + len(best_std_grasp_path) / n_nodes

        # Scalability (time increase per node, simplified)
        scalability_bfs = bfs_time / n_nodes
        scalability_grasp = grasp_time / n_nodes
        scalability_std_grasp = std_grasp_time / n_nodes

        results['BFS'].append((bfs_time, bfs_accuracy, bfs_operations))
        results['Optimized GRASP'].append((grasp_time, grasp_accuracy, grasp_operations))
        results['Standard GRASP'].append((std_grasp_time, std_grasp_accuracy, std_grasp_operations))

        paths['BFS'].append(best_bfs_path)
        paths['Optimized GRASP'].append(best_grasp_path)
        paths['Standard GRASP'].append(best_std_grasp_path)

        metrics = {
            'space_usage': space_usage,
            'time_complexity': {'BFS': 'O(V + E)', 'Optimized GRASP': 'O(V * E)', 'Standard GRASP': 'O(V * E)'},
            'operations': {'BFS': bfs_operations, 'Optimized GRASP': grasp_operations, 'Standard GRASP': std_grasp_operations},
            'solution_quality': {'BFS': bfs_quality, 'Optimized GRASP': grasp_quality, 'Standard GRASP': std_grasp_quality},
            'scalability': {'BFS': scalability_bfs, 'Optimized GRASP': scalability_grasp, 'Standard GRASP': scalability_std_grasp},
            'sequential_times': {'BFS': bfs_time, 'Optimized GRASP': grasp_time, 'Standard GRASP': std_grasp_time},
            'parallel_times': {'BFS': bfs_time_par, 'Optimized GRASP': grasp_time_par, 'Standard GRASP': std_grasp_time_par},
        }

        return results, paths, metrics

    def average_metrics(self, metrics_list):
        """Compute average metrics across multiple graphs."""
        if not metrics_list:
            return {}

        avg_metrics = {
            'space_usage': np.mean([m['space_usage'] for m in metrics_list]),
            'time_complexity': metrics_list[0]['time_complexity'],  # Same for all
            'operations': {
                'BFS': np.mean([m['operations']['BFS'] for m in metrics_list]),
                'Optimized GRASP': np.mean([m['operations']['Optimized GRASP'] for m in metrics_list]),
                'Standard GRASP': np.mean([m['operations']['Standard GRASP'] for m in metrics_list]),
            },
            'solution_quality': {
                'BFS': np.mean([m['solution_quality']['BFS'] for m in metrics_list]),
                'Optimized GRASP': np.mean([m['solution_quality']['Optimized GRASP'] for m in metrics_list]),
                'Standard GRASP': np.mean([m['solution_quality']['Standard GRASP'] for m in metrics_list]),
            },
            'scalability': {
                'BFS': np.mean([m['scalability']['BFS'] for m in metrics_list]),
                'Optimized GRASP': np.mean([m['scalability']['Optimized GRASP'] for m in metrics_list]),
                'Standard GRASP': np.mean([m['scalability']['Standard GRASP'] for m in metrics_list]),
            },
            'sequential_times': {
                'BFS': np.mean([m['sequential_times']['BFS'] for m in metrics_list]),
                'Optimized GRASP': np.mean([m['sequential_times']['Optimized GRASP'] for m in metrics_list]),
                'Standard GRASP': np.mean([m['sequential_times']['Standard GRASP'] for m in metrics_list]),
            },
            'parallel_times': {
                'BFS': np.mean([m['parallel_times']['BFS'] for m in metrics_list]),
                'Optimized GRASP': np.mean([m['parallel_times']['Optimized GRASP'] for m in metrics_list]),
                'Standard GRASP': np.mean([m['parallel_times']['Standard GRASP'] for m in metrics_list]),
            },
        }
        return avg_metrics

    def output_metrics(self, metrics):
        """Output additional metrics to the text area."""
        self.output.insert(tk.END, "\n--- Additional Metrics ---\n")
        self.output.insert(tk.END, f"Space Usage: {metrics['space_usage']:.2f} MB\n")
        self.output.insert(tk.END, "Time Complexity:\n")
        for algo, comp in metrics['time_complexity'].items():
            self.output.insert(tk.END, f"  {algo}: {comp}\n")
        self.output.insert(tk.END, "Operations:\n")
        for algo, ops in metrics['operations'].items():
            self.output.insert(tk.END, f"  {algo}: {ops:.0f}\n")
        self.output.insert(tk.END, "Solution Quality (Coverage + Path Length):\n")
        for algo, quality in metrics['solution_quality'].items():
            self.output.insert(tk.END, f"  {algo}: {quality:.2f}\n")
        self.output.insert(tk.END, "Scalability (Time per Node):\n")
        for algo, scale in metrics['scalability'].items():
            self.output.insert(tk.END, f"  {algo}: {scale:.6f}s/node\n")
        self.output.insert(tk.END, "Sequential Runtime:\n")
        for algo, time in metrics['sequential_times'].items():
            self.output.insert(tk.END, f"  {algo}: {time:.6f}s\n")
        self.output.insert(tk.END, "Parallel Runtime:\n")
        for algo, time in metrics['parallel_times'].items():
            self.output.insert(tk.END, f"  {algo}: {time:.6f}s\n")
        self.output.insert(tk.END, "\n")

    def bfs_with_path(self, A_csr, source):
        n = A_csr.shape[0]
        visited = np.zeros(n, dtype=bool)
        distance = np.full(n, -1, dtype=int)
        path = []

        visited[source] = True
        distance[source] = 0
        frontier = [source]
        path.append(source)

        while frontier:
            next_frontier = []
            for u in frontier:
                neighbors = A_csr.indices[A_csr.indptr[u]:A_csr.indptr[u+1]]
                for v in neighbors:
                    if not visited[v]:
                        visited[v] = True
                        distance[v] = distance[u] + 1
                        path.append(v)
                        next_frontier.append(v)
            frontier = next_frontier
        return distance, path

    def plot_results(self, results, graph_types):
        self.figure.clf()
        axes = self.figure.subplots(1, 3)
        ax1, ax2, ax3 = axes

        if not results['BFS']:
            return

        bar_width = 0.25
        x = np.arange(len(results['BFS']))
        ax1.bar(x - bar_width, [r[0] for r in results['BFS']], bar_width, label='BFS', color='blue')
        ax1.bar(x, [r[0] for r in results['Optimized GRASP']], bar_width, label='Optimized GRASP', color='green')
        ax1.bar(x + bar_width, [r[0] for r in results['Standard GRASP']], bar_width, label='Standard GRASP', color='orange')
        ax1.set_xlabel('Graph')
        ax1.set_ylabel('Time (s)')
        ax1.set_title('Speed Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(graph_types, rotation=45, ha='right')
        ax1.legend()

        ax2.bar(x - bar_width, [r[1] for r in results['BFS']], bar_width, label='BFS', color='blue')
        ax2.bar(x, [r[1] for r in results['Optimized GRASP']], bar_width, label='Optimized GRASP', color='green')
        ax2.bar(x + bar_width, [r[1] for r in results['Standard GRASP']], bar_width, label='Standard GRASP', color='orange')
        ax2.set_xlabel('Graph')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(graph_types, rotation=45, ha='right')
        ax2.legend()

        ax3.bar(x - bar_width, [r[2] for r in results['BFS']], bar_width, label='BFS', color='blue')
        ax3.bar(x, [r[2] for r in results['Optimized GRASP']], bar_width, label='Optimized GRASP', color='green')
        ax3.bar(x + bar_width, [r[2] for r in results['Standard GRASP']], bar_width, label='Standard GRASP', color='orange')
        ax3.set_xlabel('Graph')
        ax3.set_ylabel('Operations')
        ax3.set_title('Operations Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(graph_types, rotation=45, ha='right')
        ax3.legend()

        self.figure.tight_layout()
        self.canvas.draw()

    def run_algorithm(self):
        if self.A_csr is None:
            messagebox.showinfo("No Graph", "Please create a graph using 'Create & Run (Custom Nodes & Edges)' first.")
            return

        self.figure.clf()
        ax = self.figure.add_subplot(111)
        G = csr_to_networkx(self.A_csr)
        for i in range(self.A_csr.shape[0]):
            G.add_node(i)
        source = 0

        rows, cols = self.A_csr.nonzero()
        edges = list(zip(rows, cols))
        print(f"Graph Edges: {edges}")

        algo = self.algo_var.get()
        if algo == "bfs":
            self.output.delete(1.0, tk.END)
            visited_nodes = []
            distances, path = self.bfs_with_path(self.A_csr, source)
            self.output.insert(tk.END, f"BFS Path: {path}\nDistances: {distances}\n")
            self.bfs_with_animation(self.A_csr, source, G, ax, visited_nodes)
        elif algo == "grasp":
            path, coverage = grasp_traversal(self.A_csr, source)
            self.output.delete(1.0, tk.END)
            self.output.insert(tk.END, f"Optimized GRASP Path: {path}\nCoverage: {coverage} nodes\n")
            self.grasp_with_animation(self.A_csr, path, G, ax)
        else:
            path, coverage = standard_grasp_traversal(self.A_csr, source)
            self.output.delete(1.0, tk.END)
            self.output.insert(tk.END, f"Standard GRASP Path: {path}\nCoverage: {coverage} nodes\n")
            self.grasp_with_animation(self.A_csr, path, G, ax)

    def bfs_with_animation(self, A_csr, source, G, ax, visited_nodes):
        n = A_csr.shape[0]
        visited = np.zeros(n, dtype=bool)
        distance = np.full(n, -1, dtype=int)
        visited[source] = True
        distance[source] = 0
        frontier = [source]
        edges = []

        while frontier:
            next_frontier = []
            for u in frontier:
                visited_nodes.append(u)
                ax.clear()
                nodes = list(range(A_csr.shape[0]))
                nx.draw(G, self.pos, ax=ax, with_labels=True, node_color='lightblue', nodelist=nodes)
                nx.draw_networkx_nodes(G, self.pos, nodelist=visited_nodes, node_color='red', ax=ax)
                nx.draw_networkx_edges(G, self.pos, edgelist=edges, edge_color='red', width=2, ax=ax)
                self.canvas.draw()
                self.root.update()
                time.sleep(0.5)

                neighbors = A_csr.indices[A_csr.indptr[u]:A_csr.indptr[u+1]]
                for v in neighbors:
                    if not visited[v]:
                        visited[v] = True
                        distance[v] = distance[u] + 1
                        edges.append((u, v))
                        next_frontier.append(v)
            frontier = next_frontier

    def grasp_with_animation(self, A_csr, path, G, ax):
        visited_nodes = []
        for i in range(len(path)):
            visited_nodes.append(path[i])
            ax.clear()
            nodes = list(range(A_csr.shape[0]))
            nx.draw(G, self.pos, ax=ax, with_labels=True, node_color='lightblue', nodelist=nodes)
            nx.draw_networkx_nodes(G, self.pos, nodelist=visited_nodes, node_color='green', ax=ax)
            if i > 0:
                path_edges = []
                jump_edges = []
                for j in range(i):
                    u, v = path[j], path[j+1]
                    neighbors = A_csr.indices[A_csr.indptr[u]:A_csr.indptr[u+1]]
                    if v in neighbors:
                        path_edges.append((u, v))
                    else:
                        jump_edges.append((u, v))
                nx.draw_networkx_edges(G, self.pos, edgelist=path_edges, edge_color='green', width=2, ax=ax)
                nx.draw_networkx_edges(G, self.pos, edgelist=jump_edges, edge_color='orange', width=2, style='dashed', ax=ax)
            self.canvas.draw()
            self.root.update()
            time.sleep(0.5)

if __name__ == "__main__":
    root = tk.Tk()
    app = GraphApp(root)
    root.mainloop()