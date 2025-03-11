import matplotlib.pyplot as plt
import networkx as nx


def build_network(tokens):
    G = nx.DiGraph()
    for i in range(len(tokens) - 1):
        G.add_edge(tokens[i], tokens[i + 1])
    return G

def plot_network(G, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=False, node_color='black', edge_color='gray', node_size=5)
    ax.set_title(title)
    return fig

def save_graph(fig, file):
    fig.savefig(file, format('png'), dpi=300)
