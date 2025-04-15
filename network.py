import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pyvis.network import Network

import analysis
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
    show_weights = config["weighted"]
    weights = []
    count_graphs = len(Gs)
    fig, axs = plt.subplots(count_graphs, 1, figsize=(8, 4 * count_graphs))

    # print("before try")
    try:
        if count_graphs == 1:
            axs = [axs]

        for i, G in enumerate(Gs):
            pos = nx.spring_layout(G, seed=42, scale=1.3, k=0.3)
            punctuation_nodes = [node for node in G.nodes if node in pattern]
            word_nodes = [node for node in G.nodes if node not in pattern]

            punctuation_edges = [(u, v) for u, v in G.edges if u in pattern or v in pattern]
            word_edges = [(u, v) for u, v in G.edges if u not in pattern and v not in pattern]

            nx.draw(G, pos, with_labels=show_labels, nodelist=word_nodes, node_color=NODE_COLOR, edgelist=word_edges,
                    edge_color=EDGE_COLOR, node_size=5, ax=axs[i])
            nx.draw(G, pos, with_labels=show_labels, nodelist=punctuation_nodes, node_color=PUNC_NODE_COLOR,
                    edgelist=punctuation_edges, edge_color=PUNC_EDGE_COLOR, node_size=5, ax=axs[i])

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
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers_with, hist_with, label="S interpunkciou", color=NODE_COLOR, marker='o')
    plt.plot(bin_centers_without, hist_without, label="Bez interpunkcie", color=PUNC_NODE_COLOR, marker='o')

    slope_with, intercept_with = compute_slope(bin_centers_with, hist_with)
    slope_without, intercept_without = compute_slope(bin_centers_without, hist_without)

    plt.plot(bin_centers_with, 10 ** (intercept_with + slope_with * np.log10(bin_centers_with)), NODE_COLOR,
             label=f"Fit S interpunkciou: {slope_with:.2f}", linestyle=':')
    plt.plot(bin_centers_without, 10 ** (intercept_without + slope_without * np.log10(bin_centers_without)),
             PUNC_NODE_COLOR, label=f"Fit Bez interpunkcie: {slope_without:.2f}", linestyle=':',)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Stupeň uzla")
    plt.ylabel("Pravdepodobnosť")
    plt.legend()
    plt.title("Log-binned Degree Distribution")
    plt.grid(True)
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

