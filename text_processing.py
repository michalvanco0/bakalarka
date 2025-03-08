import networkx as nx
import matplotlib.pyplot as plt
import re

ALL_PUNCTUATION = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'


def preprocess_text(text, keep_punctuation=True, custom_punctuation=" "):
    pattern = f"[{re.escape(custom_punctuation)}]"

    tokens = []
    words = re.split(r'\s+', text)
    for word in words:
        if keep_punctuation:
            split_parts = re.split(f'({pattern})', word, flags=re.IGNORECASE)
            tokens.extend([part for part in split_parts if part])
        else:
            cleaned_word = re.sub(pattern, '', word, flags=re.IGNORECASE)
            if cleaned_word:
                tokens.append(cleaned_word)
    # print(tokens)
    return tokens


def build_network(tokens):
    G = nx.DiGraph()
    for i in range(len(tokens) - 1):
        G.add_edge(tokens[i], tokens[i + 1])
    return G


# same edges
def jaccard_similarity(G1, G2):
    edges1, edges2 = set(G1.edges()), set(G2.edges())
    intersection = len(edges1 & edges2)
    union = len(edges1 | edges2)
    # return nx.jaccard_coefficient(G1, G2) if union else 0
    return intersection / union if union else 0


# list
def degree_distribution(G):
    return sorted([d for _, d in G.degree()], reverse=True)


def graph_density(G):
    return nx.density(G)


def average_path_length(G):
    if nx.is_weakly_connected(G):
        return nx.average_shortest_path_length(G)
    return float('inf')


def clustering_coefficient(G):
    return nx.average_clustering(G.to_undirected())


def compare_networks(G1, G2):
    print("Network with Punctuation")
    print(f"Nodes: {len(G1.nodes())}, Edges: {len(G1.edges())}")
    print(f"Density: {graph_density(G1):.4f}")
    # print(f"Average Path Length: {average_path_length(G1):.2f}")
    print(f"Clustering Coefficient: {clustering_coefficient(G1):.4f}")

    print("\nNetwork without Punctuation")
    print(f"Nodes: {len(G2.nodes())}, Edges: {len(G2.edges())}")
    print(f"Density: {graph_density(G2):.4f}")
    # print(f"Average Path Length: {average_path_length(G2):.2f}")
    print(f"Clustering Coefficient: {clustering_coefficient(G2):.4f}")

    print(f"\nJaccard Similarity (Edges): {jaccard_similarity(G1, G2):.4f}")


def plot_network(G, title):
    plt.figure(figsize=(10, 5))
    pos = nx.spring_layout(G, seed=20)
    nx.draw(G, pos, with_labels=False, node_color='black', edge_color='gray', node_size=5)
    plt.title(title)
    plt.show()


def read_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        # cleaned_text = re.search(r'\*\*\* START.*?\*\*\*(.*?)\*\*\* END', text, re.DOTALL)
        cleaned_text = re.search(r'(.*?)\*\*\* END', text, re.DOTALL)
        return cleaned_text.group(1).strip() if cleaned_text else "dot"


if __name__ == "__main__":
    text = read_text_from_file("texts\\example.txt")

    custom_punctuation = ALL_PUNCTUATION
    tokens_with_punct = preprocess_text(text, keep_punctuation=True, custom_punctuation=custom_punctuation)
    tokens_without_punct = preprocess_text(text, keep_punctuation=False, custom_punctuation=custom_punctuation)

    G_with_punct = build_network(tokens_with_punct)
    G_without_punct = build_network(tokens_without_punct)

    compare_networks(G_with_punct, G_without_punct)

    plot_network(G_with_punct, "Word Network WITH Punctuation")
    plot_network(G_without_punct, "Word Network WITHOUT Punctuation")
