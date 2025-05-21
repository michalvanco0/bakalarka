import random
from collections import Counter

import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import analysis
import text_processing
from config_setter import load_config
from analysis import compute_slope
from scipy.stats import poisson, geom
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QScrollArea
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

NODE_COLOR = 'black'
EDGE_COLOR = 'gray'
PUNC_NODE_COLOR = 'red'
PUNC_EDGE_COLOR = 'salmon'


def build_network(tokens):
    try:
        config = load_config()
        weighted = config["weighted"]
        directed = config["directed"]
        G = nx.DiGraph() if directed else nx.Graph()
        for i in range(len(tokens) - 1):
            if weighted:
                if G.has_edge(tokens[i], tokens[i + 1]):
                    G[tokens[i]][tokens[i + 1]]['weight'] += 1
                else:
                    G.add_edge(tokens[i], tokens[i + 1], weight=1)
            else:
                G.add_edge(tokens[i], tokens[i + 1])
        return G
    except Exception as e:
        print(e)
        return None


def plot_networks(Gs, titles):
    config = load_config()
    pattern = config["punctuation_pattern"]
    show_labels = config["show_labels"]
    min_degree = config["min_degree"]
    count_graphs = len(Gs)
    fig, axs = plt.subplots(count_graphs, 1, figsize=(8, 4 * count_graphs))

    try:
        if count_graphs == 1:
            axs = [axs]

        for i, G in enumerate(Gs):
            degrees = dict(G.degree())
            filtered_nodes = {node for node, degree in degrees.items() if degree >= min_degree}
            G = G.subgraph(filtered_nodes)

            pos = nx.spring_layout(G, seed=42, scale=1.3, k=0.3)
            punctuation_nodes = [node for node in G.nodes if node in pattern]
            word_nodes = [node for node in G.nodes if node not in pattern]

            punctuation_edges = [(u, v) for u, v in G.edges if u in pattern or v in pattern]
            word_edges = [(u, v) for u, v in G.edges if u not in pattern and v not in pattern]

            node_sizes = {node: degree * 50 for node, degree in G.degree()}

            nx.draw(G, pos, with_labels=show_labels, nodelist=word_nodes, node_color=NODE_COLOR, edgelist=word_edges,
                    edge_color=EDGE_COLOR, node_size=[node_sizes[node] for node in word_nodes],
                    ax=axs[i], font_color='gray')
            nx.draw(G, pos, with_labels=show_labels, nodelist=punctuation_nodes, node_color=PUNC_NODE_COLOR,
                    edgelist=punctuation_edges, edge_color=PUNC_EDGE_COLOR,
                    node_size=[node_sizes[node] for node in punctuation_nodes], ax=axs[i], font_color='gray')

            axs[i].set_title(titles[i])

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(e)
    return fig


def log_binning(G, bin_count=35):
    data = [d for n, d in G.degree()]
    max_exp = np.log10(max(data))
    min_exp = np.log10(min(data))
    bins = np.logspace(min_exp, max_exp, bin_count)

    bin_means = (bins[:-1] + bins[1:]) / 2.0
    bin_counts, _ = np.histogram(data, bins=bins)

    return bin_means, bin_counts


def plot_digree_distribution(G1, G2, binned=True, xscale='log', yscale='log'):
    degrees_with = analysis.get_degrees(G1)
    degrees_without = analysis.get_degrees(G2)
    title = "Node degree distribution"
    if binned:
        title = "Node degree distribution (log-binned)"
        x_with, y_with = analysis.log_bin_degrees(degrees_with)
        x_without, y_without = analysis.log_bin_degrees(degrees_without)
    else:
        x_with, counts_with = np.unique(degrees_with, return_counts=True)
        x_without, counts_without = np.unique(degrees_without, return_counts=True)
        y_with = counts_with / counts_with.sum()
        y_without = counts_without / counts_without.sum()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x_with, y_with, label="include punctuation", color=NODE_COLOR, marker='o')
    ax.scatter(x_without, y_without, label="ignore punctuation", color=PUNC_NODE_COLOR, marker='o')

    slope_with, intercept_with = analysis.get_slope(G1, fit_range=(0.2, 0.8))
    slope_without, intercept_without = analysis.get_slope(G2, fit_range=(0.2, 0.8))

    if slope_with is not None:
        x_log = np.log10(np.array(x_with)[(np.array(x_with) > 0) & (np.array(y_with) > 0)])
        fit_min, fit_max = np.percentile(x_log, [20, 80])
        x_fit_with = np.logspace(fit_min, fit_max, 100)
        y_fit_with = 10 ** (intercept_with + slope_with * np.log10(x_fit_with))
        ax.plot(x_fit_with, y_fit_with, NODE_COLOR, linestyle='-',
                label=f"Fit (with punct): {slope_with:.2f}")

    if slope_without is not None:
        x_log = np.log10(np.array(x_without)[(np.array(x_without) > 0) & (np.array(y_without) > 0)])
        fit_min, fit_max = np.percentile(x_log, [20, 80])
        x_fit_without = np.logspace(fit_min, fit_max, 100)
        y_fit_without = 10 ** (intercept_without + slope_without * np.log10(x_fit_without))
        ax.plot(x_fit_without, y_fit_without, PUNC_NODE_COLOR, linestyle=':',
                label=f"Fit (without punct): {slope_without:.2f}")

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel("Node degree")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.set_title(title)
    ax.grid(True)
    plt.show()
    return fig


def fit_convergence_analysis(full_text, step_size=1000, max_words=None):
    from analysis import get_degrees, log_bin_degrees
    words = text_processing.tokenize_text(full_text)
    results = []

    if max_words is None:
        max_words = len(words)

    for n in range(step_size, min(len(words), max_words) + 1, step_size):
        sliced_tokens = words[:n]
        G = build_network(sliced_tokens)
        degrees = get_degrees(G)
        bin_centers, counts = log_bin_degrees(degrees)
        if len(bin_centers) > 1 and len(counts) > 1:
            slope, _ = compute_slope(bin_centers, counts)
            results.append((n, slope))
        else:
            results.append((n, None))

    return results


def plot_fit_convergence(results):
    if results is None or len(results) < 2:
        return
    xs, ys = zip(*[(x, y) for x, y in results if y is not None])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, ys, marker='o', linestyle='-', color='navy')
    ax.set_title("Power-Law")
    ax.set_xlabel("Word count")
    ax.set_ylabel("Power-law slope")
    ax.grid(True)
    ax.axhline(ys[-1], color='red', linestyle='--', label=f"Final Fit ≈ {ys[-1]:.2f}")
    ax.legend()
    plt.show()
    return fig


# def add_node(G, new_node, alpha=0):
#     degrees = np.array([G.degree(n) for n in G.nodes()])
#     degrees = np.power(degrees, alpha)
#     probs = degrees / degrees.sum()
#     nodes = list(G.nodes())
#     chosen = np.random.choice(nodes, p=probs)
#
#     neighbors = list(G.neighbors(chosen))
#     second = random.choice(neighbors) if neighbors else random.choice(nodes)
#
#     G.add_node(new_node)
#     G.add_edge(new_node, chosen)
#     G.add_edge(new_node, second)

def add_node(G, new_node, alpha=1.4, m=2):
    degrees = np.array([G.degree(node) for node in G.nodes()])
    probs = degrees ** alpha
    probs = probs / probs.sum()
    targets = np.random.choice(list(G.nodes()), size=m, replace=False, p=probs)
    for target in targets:
        G.add_edge(new_node, target)


def dorogov_model(nodes_count):
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 0)

    for _ in range(3, nodes_count):
        add_node(G, new_node=_)

    return G


def generate_models(nodes_count):
    models = {
        "Barabási-Albert": nx.barabasi_albert_graph(nodes_count, 2),
        # "Watts-Strogatz": nx.watts_strogatz_graph(nodes_count, 4, 0.3),
        "Powerlaw Cluster": nx.powerlaw_cluster_graph(nodes_count, 2, 0.5),
        "Dorogovtsev-Mendes": dorogov_model(nodes_count)
        # "Dorogovtsev-Mendes": nx.dorogovtsev_goltsev_mendes_graph(nodes_count)
    }
    return models


def plot_weibull_fit(ks, freqs, q, beta):
    model_probs = analysis.weibull_pmf(ks, q, beta)

    fig, ax = plt.subplots()
    ax.plot(ks, freqs, 'bo-', label='Empirical distribution')
    ax.plot(ks, model_probs, 'r--', label=f'Weibull fit\np={1 - q:.3f}, β={beta:.3f}')
    ax.set_title("in-between punctuation word count distribution")
    ax.set_xlabel("in-between punctuation word count")
    ax.set_ylabel("Probability")
    ax.legend()
    plt.grid()
    plt.show()
    return fig


def plot_distribution_comparisons(ks, freqs, q, beta):
    model_weibull = analysis.weibull_pmf(ks, q, beta)

    lambda_poisson = analysis.poisson_lambda(ks, freqs)
    model_poisson = poisson.pmf(ks, mu=lambda_poisson)

    p_geom = analysis.geometric_p(ks, freqs)
    model_geom = geom.pmf(ks, p=p_geom)

    fig, ax = plt.subplots()
    ax.plot(ks, freqs, 'bo-', label='Empirical distribution')
    ax.plot(ks, model_weibull, 'r--', label=f'Weibull fit\np={1 - q:.3f}, β={beta:.3f}')
    ax.plot(ks, model_poisson, 'g-.', label=f'Poisson fit\nλ={lambda_poisson:.2f}')
    ax.plot(ks, model_geom, 'm:', label=f'Geometric fit\np={p_geom:.2f}')

    ax.set_title("in-between punctuation word count comparison")
    ax.set_xlabel("in-between punctuation word count")
    ax.set_ylabel("Probability")
    ax.legend()
    plt.grid()
    plt.show()
    return fig


def degree_histogram_log(G, bin_count=35):
    degrees = [d for _, d in G.degree()]
    if not degrees:
        return np.array([]), np.array([])

    max_exp = np.log10(max(degrees))
    min_exp = np.log10(min(d for d in degrees if d > 0))  # avoid log10(0)
    bins = np.logspace(min_exp, max_exp, bin_count)

    bin_counts, edges = np.histogram(degrees, bins=bins)
    bin_centers = np.sqrt(edges[:-1] * edges[1:])

    total = bin_counts.sum()
    probs = bin_counts / total if total > 0 else np.zeros_like(bin_counts)

    mask = probs > 0
    return bin_centers[mask], probs[mask]


def fit_power_law(x, y, min_k=5):
    mask = (x >= min_k) & (y > 0)
    if not np.any(mask):
        return None, None
    log_x = np.log10(x[mask])
    log_y = np.log10(y[mask])
    coeffs = np.polyfit(log_x, log_y, 1)
    return coeffs[0], coeffs[1]


class ModelPlotWindow(QWidget):
    def __init__(self, Gs):
        super().__init__()
        self.setWindowTitle("Degree Distributions of Models")
        self.setMinimumSize(800, 600)
        self.canvas = FigureCanvas(plt.Figure(figsize=(8, 6)))
        self.ax = self.canvas.figure.subplots()
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.set_xlabel("Degree (log)")
        self.ax.set_ylabel("Probability (log)")
        self.ax.set_title("Degree Distributions (log-log)")
        self.plots = {}

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        checkbox_widget = QWidget()
        checkbox_layout = QVBoxLayout(checkbox_widget)

        for label, G in Gs.items():
            degrees = analysis.get_degrees(G)
            x, y = analysis.log_bin_degrees(degrees)
            x = np.array(x, dtype=float)
            y = np.array(y, dtype=float)

            mask = (x >= 5) & (y > 0)
            if not np.any(mask):
                slope, intercept = None, None
            else:
                slope, intercept = analysis.get_slope(G, fit_range=(0.2, 0.8))

            plot = self.ax.scatter(x, y, label=label, alpha=0.7)

            fit_line = None
            if slope is not None:
                mask = (x > 0) & (y > 0)
                x_log = np.log10(x[mask])
                p_min, p_max = np.percentile(x_log, [20, 80])
                x_fit = np.logspace(p_min, p_max, 100)
                y_fit = 10 ** (intercept + slope * np.log10(x_fit))
                fit_line, = self.ax.plot(x_fit, y_fit, '--', label=f"{label} fit", alpha=0.7)

            self.plots[label] = (plot, fit_line)

            cb_label = f"{label} ({slope:.2f})" if slope is not None else label
            cb = QCheckBox(cb_label)
            cb.setChecked(True)
            cb.stateChanged.connect(lambda state, l=label: self.toggle_visibility(l))
            checkbox_layout.addWidget(cb)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(checkbox_widget)
        layout.addWidget(scroll)

        self.canvas.draw()
        self.setLayout(layout)

    def toggle_visibility(self, label):
        scatter, fit_line = self.plots[label]
        visible = not scatter.get_visible()
        scatter.set_visible(visible)
        if fit_line:
            fit_line.set_visible(visible)
        self.canvas.draw()


def show_model_plot_window(Gs):
    window = ModelPlotWindow(Gs)
    window.show()
    return window


def compute_slopes(Gs):
    slopes = {}
    for label, G in Gs.items():
        slope = analysis.get_slope(G)[0]
        slopes[label] = slope
    return slopes
