import random
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pyvis.network import Network
from scipy.optimize import minimize
from scipy.stats import weibull_min

import analysis
import text_processing
from config_setter import load_config
from analysis import compute_slope

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
                    ax=axs[i])
            nx.draw(G, pos, with_labels=show_labels, nodelist=punctuation_nodes, node_color=PUNC_NODE_COLOR,
                    edgelist=punctuation_edges, edge_color=PUNC_EDGE_COLOR,
                    node_size=[node_sizes[node] for node in punctuation_nodes], ax=axs[i])

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


def plot_logbining(G1, G2):
    degrees_with = analysis.get_degrees(G1)
    degrees_without = analysis.get_degrees(G2)
    bin_centers_with, hist_with = analysis.log_bin_degrees(degrees_with)
    bin_centers_without, hist_without = analysis.log_bin_degrees(degrees_without)

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.plot(bin_centers_with, hist_with, label="S interpunkciou", color=NODE_COLOR, marker='o')
    ax.plot(bin_centers_without, hist_without, label="Bez interpunkcie", color=PUNC_NODE_COLOR, marker='o')

    slope_with, intercept_with = compute_slope(bin_centers_with, hist_with)
    slope_without, intercept_without = compute_slope(bin_centers_without, hist_without)

    ax.plot(bin_centers_with, 10 ** (intercept_with + slope_with * np.log10(bin_centers_with)), NODE_COLOR,
            label=f"Fit S interpunkciou: {slope_with:.2f}", linestyle=':')
    ax.plot(bin_centers_without, 10 ** (intercept_without + slope_without * np.log10(bin_centers_without)),
            PUNC_NODE_COLOR, label=f"Fit Bez interpunkcie: {slope_without:.2f}", linestyle=':')

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Stupeň uzla")
    ax.set_ylabel("Pravdepodobnosť")
    ax.legend()
    ax.set_title("Log-binned Degree Distribution")
    ax.grid(True)

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
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, marker='o', linestyle='-', color='navy')
    plt.title("Power-Law Fit Convergence")
    plt.xlabel("Number of Words in Text")
    plt.ylabel("Power-Law Slope")
    plt.grid(True)
    plt.axhline(ys[-1], color='red', linestyle='--', label=f"Final Fit ≈ {ys[-1]:.2f}")
    plt.legend()
    plt.show()


def add_node(G, new_node, edges):
    edge = random.choice(edges)
    u, v = edge
    G.add_node(new_node)
    G.add_edge(new_node, u)
    G.add_edge(new_node, v)


def dorogov_model(nodes_count):
    G = nx.Graph()
    G.add_node(0)
    G.add_node(1)
    G.add_edge(0, 1)
    edges = list(G.edges())
    for _ in range(nodes_count):
        new_node = len(G.nodes())
        add_node(G, new_node, edges)
    return G


def get_punctuation_distances(tokens, punctuation_set):
    distances = []
    last_index = None
    for i, token in enumerate(tokens):
        if token in punctuation_set:
            if last_index is not None:
                dist = i - last_index - 1
                if dist > 0:
                    distances.append(dist)
            last_index = i
    return distances


def weibull_like_pmf(k, q, beta):
    return q ** (k ** beta) - q ** ((k + 1) ** beta)


def fit_weibull_like_model(distances):
    if not distances:
        return None, None, None

    counts = Counter(distances)
    ks = np.array(sorted(counts.keys()))
    freqs = np.array([counts[k] for k in ks], dtype=float)
    freqs /= freqs.sum()  # Normalize

    def loss(params):
        q, beta = params
        if not (0 < q < 1) or beta <= 0:
            return np.inf
        model = weibull_like_pmf(ks, q, beta)
        return np.sum((freqs - model) ** 2)

    result = minimize(loss, [0.5, 1.0], bounds=[(1e-5, 1-1e-5), (1e-2, 10)])
    if result.success:
        q_fit, beta_fit = result.x
        return ks, freqs, q_fit, beta_fit
    else:
        return ks, freqs, None, None


def plot_weibull_like_fit(ks, freqs, q, beta):
    model_probs = weibull_like_pmf(ks, q, beta)

    fig, ax = plt.subplots()
    ax.plot(ks, freqs, 'bo-', label='Empirická distribúcia')
    ax.plot(ks, model_probs, 'r--', label=f'Weibull-like fit\nq={q:.3f}, β={beta:.3f}')
    ax.set_title("Distribúcia vzdialeností medzi interpunkciami")
    ax.set_xlabel("Počet slov medzi interpunkciami (k)")
    ax.set_ylabel("Pravdepodobnosť")
    ax.legend()
    plt.grid()
    plt.show()
    return fig