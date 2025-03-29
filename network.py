import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from config_setter import load_config
from analysis import compute_slope

NODE_COLOR = 'black'
EDGE_COLOR = 'gray'
PUNC_NODE_COLOR = 'red'
PUNC_EDGE_COLOR = 'salmon'


def build_network(tokens):
    G = nx.DiGraph()
    for i in range(len(tokens) - 1):
        G.add_edge(tokens[i], tokens[i + 1])
    return G


# def plot_network(G, title):
#     fig, ax = plt.subplots(figsize=(10, 5))
#     pos = nx.spring_layout(G, seed=42)
#     nx.draw(G, pos, with_labels=False, node_color=NODE_COLOR, edge_color=EDGE_COLOR, node_size=5)
#     ax.set_title(title)
#     plt.show()
#     return fig


def plot_networks(Gs, titles):
    config = load_config()
    pattern = config["punctuation_pattern"]
    count_graphs = len(Gs)
    fig, axs = plt.subplots(count_graphs, 1, figsize=(8, 4 * count_graphs))

    if count_graphs == 1:
        axs = [axs]

    for i, G in enumerate(Gs):
        pos = nx.spring_layout(G, seed=42, scale=1.3, k=0.3)
        punctuation_nodes = [node for node in G.nodes if node in pattern]
        word_nodes = [node for node in G.nodes if node not in pattern]

        punctuation_edges = [(u, v) for u, v in G.edges if u in pattern or v in pattern]
        word_edges = [(u, v) for u, v in G.edges if u not in pattern and v not in pattern]

        nx.draw(G, pos, with_labels=False, nodelist=word_nodes, node_color=NODE_COLOR, edgelist=word_edges,
                edge_color=EDGE_COLOR, node_size=5, ax=axs[i])
        nx.draw(G, pos, with_labels=False, nodelist=punctuation_nodes, node_color=PUNC_NODE_COLOR,
                edgelist=punctuation_edges, edge_color=PUNC_EDGE_COLOR, node_size=5, ax=axs[i])

        axs[i].set_title(titles[i])

    plt.tight_layout()
    plt.show()
    return fig


def log_binning(G, bin_count=35):
    data = [d for n, d in G.degree()]
    max_exp = np.log10(max(data))
    min_exp = np.log10(min(data))
    bins = np.logspace(min_exp, max_exp, bin_count)

    bin_means = (bins[:-1] + bins[1:]) / 2.0
    bin_counts, _ = np.histogram(data, bins=bins)

    return bin_means, bin_counts


def plot_logbining(bin_centers_with, bin_centers_without, hist_with, hist_without):
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers_with, hist_with, label="S interpunkciou", color=NODE_COLOR, marker='o')
    plt.plot(bin_centers_without, hist_without, label="Bez interpunkcie", color=PUNC_NODE_COLOR, marker='o')

    slope_with, intercept_with = compute_slope(bin_centers_with, hist_with)
    slope_without, intercept_without = compute_slope(bin_centers_without, hist_without)

    plt.plot(bin_centers_with, 10 ** (intercept_with + slope_with * np.log10(bin_centers_with)), NODE_COLOR,
             label=f"Fit S interpunkciou: {slope_with:.2f}")
    plt.plot(bin_centers_without, 10 ** (intercept_without + slope_without * np.log10(bin_centers_without)),
             PUNC_NODE_COLOR, label=f"Fit Bez interpunkcie: {slope_without:.2f}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Stupeň uzla (Degree)")
    plt.ylabel("Pravdepodobnosť (Probability)")
    plt.legend()
    plt.title("Log-binned Degree Distribution")
    plt.grid(True)
    plt.show()


def save_graph(fig, file):
    fig.savefig(file, format('png'), dpi=300)
